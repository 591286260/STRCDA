import numpy as np
import pandas as pd
from tqdm import trange
import torch
import torch.nn as nn
import scipy.sparse as sp

from my_code.rwr import SimtoRWR

# from codes.pca import reduction


path='../dataset/1-CircR2Disease(561-100-607)/'
# path = '../dataset/2-CircAtlas 2.0(708-117-775)/'
# path = '../dataset/3-Circ2Disease(234-60-254)/'
# path = '../dataset/4-CircRNADisease(286-48-304)/'
input_dim = 128         ##128,128,124,112
output_dim = 661       ##661,825,294,334
# n_cir=561
# n_dis=100

def Makeadj(AM):
    adj = []
    for i in trange(AM.shape[0]):
        for j in range(AM.shape[1]):
            adj_inner = []
            if AM[i][j] == 1:
                adj_inner.append(i + 1)
                adj_inner.append(j + 1)
                adj_inner.append(1)
                adj.append(adj_inner)
    return np.array(adj)

def normalize_features(feat):    ###对一个特征矩阵进行归一化
    degree = np.asarray(feat.sum(1)).flatten()
    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf
    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    return feat_norm

class LinearExpansion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearExpansion, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

###水平拼接
def heteg(SC, SD, AM):
    reSC = np.hstack((SC, AM))
    reSD = np.hstack((SD, AM.T))
    return reSC, reSD


def prepareData(path):
    # Reading data from disk
    path_am = path+'AM.csv'
    path_cs = path + 'RNA_sim.csv'
    path_ds = path + 'Disease_sim.csv'
    AM = pd.read_csv(path_am, header=None).values
    CS = pd.read_csv(path_cs, header=None).values
    DS = pd.read_csv(path_ds, header=None).values

    # Using RWR to calculate CRS and DRS
    CRS, DRS = SimtoRWR(CS, DS, path)
    # print(CRS.shape)#561*561
    # print(DRS.shape)#100*100

    # Matrix Splicing
    reSC, reSD = heteg(CRS, DRS, AM)

    # array = np.zeros((n_cir, n_dis))
    # reSC1, reSD1 = heteg(CRS, DRS, array)
    # f=np.vstack((reSC1,reSD1))
    # np.savetxt('../dataset/save/feature_rwr_sim.csv', f, delimiter=',', fmt='%f')

    feature=np.vstack((reSC,reSD))
    np.savetxt(path+'feature.csv', feature, delimiter=',', fmt='%f')
    print("随机游走完成")



# Preparing data for training
prepareData(path)





