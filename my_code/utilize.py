import numpy as np

def read_txt(path):
    with open(path, 'r', newline='') as txt_file:
        md_data = []
        reader = txt_file.readlines()
        for row in reader:
            line = row.split(',')
            row = []
            for k in line:
                row.append(float(k))
            md_data.append(row)
        md_data = np.array(md_data)
        return md_data

#get the features of each node by integrating three different network
def get_feature(M_M, M_D,D_D):
    H1 = np.hstack(( M_M, M_D))
    H2 = np.hstack((M_D.transpose(),D_D))
    H = np.vstack((H1,H2))
    print('The shape of H', H.shape)
    return H

#find the circRNA-disease part
def find_C_D (edges):
    m_d = []
    for i in range(edges.shape[0]):
        if edges[i,0]<607 and edges[i,1] >606:#577,576
            m_d.append(edges[i,:])
        elif edges[i,0]>606 and edges[i,1]<607:#576,577
            m_d.append(edges[i,:])
    return m_d

