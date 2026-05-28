```python
import numpy as np
import pandas as pd
from tqdm import trange
import torch
import torch.nn as nn
import scipy.sparse as sp

from my_code.rwr import SimtoRWR

# Dataset path
path = '../dataset/1-CircR2Disease(561-100-607)/'
# path = '../dataset/2-CircAtlas 2.0(708-117-775)/'
# path = '../dataset/3-Circ2Disease(234-60-254)/'
# path = '../dataset/4-CircRNADisease(286-48-304)/'

# Feature dimensions for different datasets
input_dim = 128
output_dim = 661


def Makeadj(AM):
    """
    Convert the circRNA-disease association matrix into edge-list format.

    Parameters
    ----------
    AM : ndarray
        Binary circRNA-disease association matrix.

    Returns
    -------
    ndarray
        Edge list in the format [circRNA_index, disease_index, association_label].
    """
    adj = []

    for i in trange(AM.shape[0]):
        for j in range(AM.shape[1]):
            if AM[i][j] == 1:
                adj.append([i + 1, j + 1, 1])

    return np.array(adj)


def normalize_features(feat):
    """
    Row-normalize the input feature matrix.

    Parameters
    ----------
    feat : scipy.sparse matrix
        Input feature matrix.

    Returns
    -------
    scipy.sparse matrix
        Row-normalized feature matrix.
    """
    degree = np.asarray(feat.sum(1)).flatten()

    # Avoid division by zero
    degree[degree == 0.] = np.inf

    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])

    feat_norm = degree_inv_mat.dot(feat)

    return feat_norm


class LinearExpansion(nn.Module):
    """
    Linear feature expansion module.

    This layer maps input features into a target-dimensional space.
    """

    def __init__(self, input_dim, output_dim):
        super(LinearExpansion, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """Apply linear transformation."""
        return self.linear(x)


def heteg(SC, SD, AM):
    """
    Construct heterogeneous feature representations.

    The circRNA feature matrix is formed by concatenating circRNA
    similarity and the association matrix. The disease feature matrix
    is formed by concatenating disease similarity and the transposed
    association matrix.

    Parameters
    ----------
    SC : ndarray
        circRNA similarity matrix.
    SD : ndarray
        Disease similarity matrix.
    AM : ndarray
        circRNA-disease association matrix.

    Returns
    -------
    tuple
        Concatenated circRNA and disease feature matrices.
    """
    reSC = np.hstack((SC, AM))
    reSD = np.hstack((SD, AM.T))

    return reSC, reSD


def prepareData(path):
    """
    Prepare feature matrices for STRCDA.

    This function loads the circRNA-disease association matrix,
    circRNA similarity matrix, and disease similarity matrix. It then
    applies random walk with restart to refine similarity profiles,
    constructs heterogeneous node features, and saves the final feature
    matrix for subsequent graph autoencoder training.
    """

    # Load input matrices
    path_am = path + 'AM.csv'
    path_cs = path + 'RNA_sim.csv'
    path_ds = path + 'Disease_sim.csv'

    AM = pd.read_csv(path_am, header=None).values
    CS = pd.read_csv(path_cs, header=None).values
    DS = pd.read_csv(path_ds, header=None).values

    # Refine circRNA and disease similarity matrices using RWR
    CRS, DRS = SimtoRWR(CS, DS, path)

    # Construct heterogeneous feature matrices
    reSC, reSD = heteg(CRS, DRS, AM)

    # Combine circRNA and disease representations
    feature = np.vstack((reSC, reSD))

    # Save the generated feature matrix
    np.savetxt(
        path + 'feature.csv',
        feature,
        delimiter=',',
        fmt='%f'
    )

    print("Random walk with restart completed.")


# Generate features for graph autoencoder training
prepareData(path)
```
