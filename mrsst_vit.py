import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
import torch.functional as F
import torch.nn.functional as FN
from torchsummary import summary


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+torch.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))


def gelu(x):
    return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*np.power(x,3))))


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(FN.softplus(x)))
        return x



# 获取相邻波段像素
def te_gain_neighbor_band(x_train, band, neighbor_band, patch=9):
    # (b,patch,patch,band)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[2], x_train.shape[3], band)
    nn = neighbor_band // 2
    z = int((patch / 2)) + 1
    x_temp = x_train[:, z - 1:z, z - 1:z, :]
    x_train_reshape = x_temp.reshape(x_train.shape[0], 1, band)
    x_train_band = torch.tensor(np.zeros((x_train.shape[0], neighbor_band, band), dtype=float)).cuda()
    x_train_band[:, nn:(nn + 1), :] = x_train_reshape
    x_train_band[:, (nn - 1):nn, :] = x_train_reshape
    for i in range(nn - 1):
        x_train_band[:, i:(i + 1), :(nn - i)] = x_train_reshape[:, 0:1, (band - nn + i):]
        x_train_band[:, i:(i + 1), (nn - i):] = x_train_reshape[:, 0:1, :(band - nn + i)]
    for i in range(nn - 1):
        x_train_band[:, (nn + 1 + i):(nn + 2 + i), (band - i - 1):] = x_train_reshape[:, 0:1, :(i + 1)]
        x_train_band[:, (nn + 1 + i):(nn + 2 + i), :(band - i - 1)] = x_train_reshape[:, 0:1, (i + 1):]  # 64,7,200
    x_train_band = x_train_band.reshape(x_train_band.shape[0], band, x_train_band.shape[1])  # 64,200,8
    return x_train_band.to(torch.float32)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode):
        super().__init__()
        self.layers1 = nn.Sequential(nn.Conv1d(num_channel + 1, num_channel + 1, kernel_size=1, padding=0)
                                     )
        self.layers2 = nn.Sequential(nn.Conv1d(num_channel + 1, num_channel + 1, kernel_size=1, padding=0)
                                     )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))

        self.mode = mode
        self.skipcat = nn.ModuleList([])
        self.fucat = nn.ModuleList([])
        for _ in range(depth - 2):
            self.skipcat.append(nn.Sequential(nn.Conv2d(num_channel + 1, num_channel + 1, [1, 2], 1, 0)

                                              )
                                )
            self.fucat.append(nn.Sequential(nn.Conv2d(num_channel + 1, num_channel + 1, [1, 2], 1, 0)

                                            )
                              )


    def forward(self, x, mask=None):
        if self.mode == 'VIT':
            for attn, ff in self.layers:
                x = attn(x, mask=mask)
                x = ff(x)

        elif self.mode == 'MICF':
            last_output = []
            nl = 0
            for attn, ff in self.layers:
                last_output.append(x)
                if nl > 1:
                    x1 = self.fucat[nl - 2](
                        torch.cat([(last_output[nl - 1]).unsqueeze(3), (last_output[nl - 2]).unsqueeze(3)],
                                  dim=3)).squeeze(3)
                    x1 = self.layers1(x1)
                    x = self.skipcat[nl - 2](torch.cat([x.unsqueeze(3), x1.unsqueeze(3)], dim=3)).squeeze(3)

                x = attn(x, mask=mask)
                x = ff(x)
                nl += 1

        return x


class ViT(nn.Module):
    def __init__(self, image_size, near_band, num_patches, num_classes, dim, depth, heads, mlp_dim, pool='cls',
                 channels=1, dim_head=16, dropout=0., emb_dropout=0., mode='ViT'):
        super().__init__()

        patch_dim = image_size ** 2 * near_band

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, mode)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, mask=None):
        # patchs[batch, patch_num, patch_size*patch_size*c]  [batch,200,145*145]
        # x = rearrange(x, 'b c h w -> b c (h w)')

        ## embedding every patch vector to embedding size: [batch, patch_num, embedding_size]
        x = self.patch_to_embedding(x)  # [b,n,dim]
        b, n, _ = x.shape

        # add position embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # [b,1,dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [b,n+1,dim]
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        x = self.transformer(x, mask)

        # classification: using cls_token output
        x = self.to_latent(x[:, 0])

        # MLP classification layer
        return self.mlp_head(x)



# 获取相邻波段像素
def gain_neighbor_band_pixel(x_train, band, neighbor_band, patch=9):
    #相邻波段必须是奇数
    pixel_number = neighbor_band +1
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[2], x_train.shape[3], band)
    # (batch_size,patch,patch,band)
    pad = int((pixel_number / 2) -1)
    z = int((patch / 2)) + 1
    x_temp = x_train[:, z-1:z, z-1:z, :] #patch 中心像素点
    x_train_reshape = x_temp.reshape(x_train.shape[0], 1, band)
    x_zero = torch.tensor(np.zeros((x_train.shape[0], 1, pad)),dtype=torch.float32).cuda()
    x_train_band = torch.tensor(np.zeros((x_train.shape[0] , pixel_number, band)),dtype=torch.float32).cuda()
    x_train_band[:, pad:pad+1, :] = x_train_reshape
    x_train_band[:, pad+1:pad+2, :] = x_train_reshape
    x_train_reshape = torch.cat((x_zero,x_train_reshape,x_zero),dim = 2)


    for i in range(band):
        for j in range(pad):
            x_train_band[:, pad + 2 + j:pad + 3 + j, i:i + 1] =x_train_reshape[:,0:1,pad + j + i +1 :pad + j +i + 2]  #右边填充
            x_train_band[:, pad  - j -1 :pad - j, i:i + 1] =x_train_reshape[:,0:1,pad +i - j - 1 :pad +i - j]

    x_train_band = x_train_band.reshape(x_train_band.shape[0], band, x_train_band.shape[1])

    return x_train_band.to(torch.float32)



class CFPE_module(nn.Module):

    def __init__(self, neighbor_band, num_patches,):
        super().__init__()
        self.conv1d_1 = nn.Sequential(
            nn.Conv1d(in_channels=num_patches, out_channels=num_patches, kernel_size=1, padding=0),
            nn.BatchNorm1d(num_patches),
            Mish()
        )

        # 光谱、空间卷积

        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1, 1, 1), padding=0, stride=(1, 1, 1)),
            nn.BatchNorm3d(1),
            Mish(),
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(neighbor_band, 5, 5),
                      padding=(int((neighbor_band - 1) / 2), 2, 2), stride=(1, 1, 1)),
            nn.BatchNorm3d(1),
            Mish()

        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1, 1, 1), padding=0, stride=(1, 1, 1)),
            nn.BatchNorm3d(1),
            Mish(),
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(neighbor_band, 3, 3),
                      padding=(int((neighbor_band - 1) / 2), 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(1),
            Mish()
        )


        self.depth_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=num_patches, out_channels=num_patches, kernel_size=(1, 1), stride=1, padding=0,
                      groups=num_patches),
            Mish(),
            nn.Conv2d(in_channels=num_patches, out_channels=num_patches, kernel_size=(5, 5), stride=1, padding=2,
                      groups=num_patches),
            Mish()
        )

        self.depth_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=num_patches, out_channels=num_patches, kernel_size=(1, 1), stride=1, padding=0,
                      groups=num_patches),
            Mish(),
            nn.Conv2d(in_channels=num_patches, out_channels=num_patches, kernel_size=(3, 3), stride=1, padding=1,
                      groups=num_patches),
            Mish()
        )

    def forward(self, x,):
        x_3d = torch.unsqueeze(x, 1)
        x_3d_1 = self.conv3d_1(x_3d)
        x_3d_2 = self.conv3d_2(x_3d)
        x_3d = x_3d_1 + x_3d_2
        x_3d = torch.squeeze(x_3d, 1)

        # 空间通道
        x_spa_1 = self.depth_conv1(x)
        x_spa_2 = self.depth_conv2(x)
        x_spa = x_spa_1 + x_spa_2


        # x= (x_spa + x_3d) * 0.7 + x
        x = x_spa +x_3d
        return x



class MRTokenG(nn.Module):

    def __init__(self, dim,patch,band, neighbor_band, num_patches):
        super().__init__()
        self.fc_layers= nn.Linear(patch * patch, dim)
        self.num_patches = num_patches
        self.neighbor_band = neighbor_band
        self.patch = patch
        self.pooling =nn.AdaptiveAvgPool2d(1)
        self.spe_para = nn.Sequential(
            nn.Linear(band, 64),
            nn.Linear(64, band),
            nn.Softmax(dim=1)
        )
        self.dw_conv1 =nn.Sequential(
            nn.Conv2d(in_channels=num_patches, out_channels=num_patches, kernel_size=(1, 1), stride=1, padding=0,
                      groups=num_patches),
            Mish()
        )


    def forward(self, x_raw,x_fe):
        x_spe = te_gain_neighbor_band(x_raw, band=self.num_patches, neighbor_band=self.neighbor_band,
                                   patch=self.patch)  # 获取邻波段的像素
        x_spe = x_spe.unsqueeze(3)
        x_adaptive_spe = self.dw_conv1(x_spe)
        x_adaptive_spe = x_adaptive_spe.squeeze(3)
        # x_temp = self.pooling(x_raw)
        # x_temp = x_temp.squeeze(-1).squeeze(-1)
        # x_spe_para = self.spe_para(x_temp)
        # x_spe_para = x_spe_para.unsqueeze(-1)
        # x_spe *= x_spe_para
        x_fe = x_fe.flatten(2)
        x_fe = self.fc_layers(x_fe)
        x = torch.cat((x_fe, x_adaptive_spe),2)


        return x




class mrsst(nn.Module):

    def __init__(self, patch, neighbor_band, num_patches, num_classes, dim, depth, heads, mlp_dim, pool='cls',
                 channels=1, dim_head=16, dropout=0., emb_dropout=0., mode='MICF'):
        super().__init__()
        self.cfpe =CFPE_module(neighbor_band=neighbor_band,num_patches=num_patches)
        self.TG = MRTokenG(dim =dim,band=num_patches,patch=patch,neighbor_band=neighbor_band,num_patches=num_patches)
        self.L = num_patches
        self.cT = dim+neighbor_band


        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim+neighbor_band))
        # self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim+neighbor_band))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim+neighbor_band, depth, heads, dim_head, mlp_dim, dropout, num_patches, mode)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim+neighbor_band),
            nn.Linear(dim+neighbor_band, num_classes)
        )

        self.num_patches = num_patches
        self.neighbor_band = neighbor_band
        self.patch = patch

    def forward(self, x, mask=None, ):
        x =torch.squeeze(x,1)
        x= x.permute(0,3,1,2)
        x_encoded =self.cfpe(x)
        x = self.TG(x,x_encoded)
        b, n, _ = x.shape

        # 位置嵌入
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # [b,1,dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [b,n+1,dim]
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        x = self.transformer(x, mask)

        # classification: using cls_token output
        x = self.to_latent(x[:, 0])

        # MLP classification layer
        return self.mlp_head(x)




# patch - patchsize
# num_patches - Number of spectral channels
# neighbor_band - Number of neighbor_band (M)
# mode -Convolutional fusion module(VIT OR MICF)
net =mrsst(patch=15,neighbor_band=5,num_patches=30,num_classes=16,dim=256,depth=7,heads=8,mlp_dim=64,dropout=0,emb_dropout=0
           ,mode='MICF')
net = net.cuda()
input = torch.randn(size=(64, 1, 15,15,30)).cuda()
y = net(input)