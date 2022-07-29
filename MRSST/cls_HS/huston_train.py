import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
from mrsst_vit import mrsst
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import spectral
import time
import argparse



parser = argparse.ArgumentParser("HSI")
parser.add_argument('--PCA_band', type=int, default=30, help='number of PCA')
parser.add_argument('--mode', choices=['VIT', 'CAF','MICF'], default='MICF', help='mode choice')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_ratio', type=int, default=5, help='number of test')
parser.add_argument('--patches', type=int, default=1, help='number of patches')
parser.add_argument('--neibor_bands', type=int, default=1, help='number of related band')
parser.add_argument('--epoches', type=int, default=300, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--dim', type=int, default=64, help='dim of transformer')
parser.add_argument('--depth', type=int, default=5, help='dim of depth')
parser.add_argument('--head', type=int, default=8, help='dim of head')
parser.add_argument('--head_dim', type=int, default=64, help='dim of head_dim')
args = parser.parse_args()
args.mode = 'MICF'
args.neibor_bands= 7 #分组波段数
args.patches = 15
args.test_ratio = 0.95
args.PCA_band = 15
args.dim = 156
args.depth = 7
 #patchesXpatchesXband的窗口大小为输入
args.epoches = 150
class_num = 16
args.batch_size = 64
args.head= 8
args.head_dim = 64



#目前最新 tesnet4 在改tesnet5

def loadData():
    # 读入数据
    data = sio.loadmat('../data/Houston.mat')['Houston']
    labels = sio.loadmat('../data/Houston_gt.mat')['Houston_gt']

    return data, labels

# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):

    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX

# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):

    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX

# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):

    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1

    return patchesData, patchesLabels


def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=testRatio,
                                                        random_state=randomState,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test

args.batch_size = 64

def create_data_loader():
    # 地物类别
    # class_num = 16
    # 读入数据
    X, y = loadData()
    # 用于测试样本的比例
    test_ratio = args.test_ratio
    # 每个像素周围提取 patch 的尺寸
    patch_size = args.patches
    # 使用 PCA 降维，得到主成分的数量
    pca_components = args.PCA_band

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA: ', X_pca.shape)

    print('\n... ... create data cubes ... ...')
    X_pca, y = createImageCubes(X_pca, y, windowSize=args.patches)
    print('Data cube X shape: ', X_pca.shape)
    print('Data cube y shape: ', y.shape)

    print('\n... ... create train & test data ... ...')
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y, test_ratio)
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)

    # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
    X = X_pca.reshape(-1, patch_size, patch_size, pca_components)
    Xtrain = Xtrain.reshape(-1, pca_components,patch_size,patch_size)
    Xtest = Xtest.reshape(-1, pca_components,patch_size,patch_size)
    print('transpose: Xtrain shape: ', Xtrain.shape)
    print('transpose: Xtest  shape: ', Xtest.shape)

    # # 为了适应 pytorch 结构，数据要做 transpose
    # X = X.transpose(0, 4, 3, 1, 2)
    # Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    # Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    # print('after transpose: Xtrain shape: ', Xtrain.shape)
    # print('after transpose: Xtest  shape: ', Xtest.shape)

    # 创建train_l oader和 test_loader
    X = TestDS(X, y)
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               drop_last=True
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               drop_last=True
                                              )
    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=0,
                                                drop_last=True
                                              )

    return train_loader, test_loader, all_data_loader, y

""" Training dataset"""

class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):

        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):

        # 返回文件数据的数目
        return self.len

""" Testing dataset"""

class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):

        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):

        # 返回文件数据的数目
        return self.len
# patches= 30;

def train(train_loader, epochs):

    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device(0)
    # 网络放到GPU上 网络在这
    net =mrsst(patch=args.patches, neighbor_band=args.neibor_bands,
                num_patches=args.PCA_band,
                num_classes=class_num,
                dim=args.dim,
                depth=args.depth,
                heads=args.head,
                mlp_dim=args.head_dim,
                dropout=0.1,
                emb_dropout=0.1,
                mode=args.mode).cuda()
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 初始化优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # 开始训练
    total_loss = 0
    for epoch in range(epochs):
        net.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # 正向传播 +　反向传播 + 优化
            # 通过输入得到预测的输出
            outputs = net(data)
            # 计算损失函数
            loss = criterion(outputs, target)
            # 优化器梯度归零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (epoch + 1,
                                                                         total_loss / (epoch + 1),
                                                                         loss.item()))

    print('Finished Training')

    return net, device

def tes(device, net, test_loader):
    count = 0
    # 模型测试
    net.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test

def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test):

    target_names = ['1', 'Corn-2', 'Corn-3', '4'
        , 'Grass-5', '6trees', '7-pasture-mowed',
                    'Hay-8', '9', '10-notill', '11-mintill',
                    '12-clean', '13', '14', '15-Grass-Trees-Drives']
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100

if __name__ == '__main__':

    train_loader, test_loader, all_data_loader, y_all= create_data_loader()
    tic1 = time.perf_counter()
    net, device = train(train_loader, epochs=args.epoches)
    # 只保存模型参数
    torch.save(net.state_dict(),'HS_MRSST.pth')
    toc1 = time.perf_counter()
    tic2 = time.perf_counter()
    y_pred_test, y_test = tes(device, net, test_loader)
    toc2 = time.perf_counter()
    # 评价指标
    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
    classification = str(classification)
    Training_Time = toc1 - tic1
    Test_time = toc2 - tic2
    file_name = "HS_MRSST.txt"
    with open(file_name, 'w') as x_file:
        x_file.write('train ratio:{},epoch:{},PCA:{}'.format(args.test_ratio,args.epoches,args.PCA_band))
        x_file.write('\n')
        x_file.write('{} Training_Time (s)'.format(Training_Time))
        x_file.write('\n')
        x_file.write('{} Test_time (s)'.format(Test_time))
        x_file.write('\n')
        x_file.write('{} Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('{} Each accuracy (%)'.format(each_acc))
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))


