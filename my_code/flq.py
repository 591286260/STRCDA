import json
import random
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve,auc
import lightgbm as lgb
import xgboost as xgb
from my_code.poltting import poltting

path = '../dataset/1-CircR2Disease(561-100-607)/'
# path = '../dataset/2-CircAtlas 2.0(708-117-775)/'
# path = '../dataset/3-Circ2Disease(234-60-254)/'
# path = '../dataset/4-CircRNADisease(286-48-304)/'

num_circ = 561
num_dis = 100
con = 607

# 加载特征数据和标签数据

# X1 = np.loadtxt(path + '0-emb_train_rwr')  # 特征数据661*64
# data_rec_adj = np.loadtxt(path + '0-data_rec_adj_rwr')  # 特征数据
# X1 = np.loadtxt(path + '1-emb_train')  # 特征数据661*64
# data_rec_adj = np.loadtxt(path + '1-data_rec_adj')  # 特征数据
X1 = np.loadtxt('0-emb_train_rwr')  # 特征数据661*64
data_rec_adj = np.loadtxt('0-data_rec_adj_rwr')  # 特征数据
print(X1.shape)
print(data_rec_adj.shape)

X1 = np.array(X1)
X2 = np.array(data_rec_adj).reshape(num_dis+num_circ, num_dis+num_circ)
X = np.concatenate((X1, X2), axis=1)
# X = X1


print(X.shape)
AM = np.loadtxt(path + 'AM.csv', delimiter=',')  # 标签数据

y = []
for i in range(con):
    y.append(1)
for i in range(con):
    y.append(0)

circfea = X[:num_circ]
disfea = X[num_circ:]
# print(circfea.shape)
# print(disfea.shape)
fea = []
for i in range(num_circ):
    for j in range(num_dis):
        # 确保它们是 numpy 数组
        circfea[i] = np.array(circfea[i])
        disfea[j] = np.array(disfea[j])
        fea.append(np.hstack([circfea[i], disfea[j]]))##56100*128
feature = []
for i in range(num_circ):
    for j in range(num_dis):
        if AM[i, j] == 1:
            feature.append(fea[i*num_dis+j])
ijdx = []
while len(feature) < 2*con:
    random_i = random.randint(0, num_circ-1)
    random_j = random.randint(0, num_dis-1)
    ij = random_i*num_dis + random_j
    if (ij not in ijdx) and (AM[random_i, random_j] == 0):
        feature.append(fea[random_i*num_dis + random_j])
        ijdx.append(ij)

# 查看数据形状，确保正确加载
# print(f"特征数据形状: {len(feature)}")


X_train=np.array(feature)
y_train=np.array(y)

# l1 = sys.argv[1]
# l1 = int(l1)
# print("l1", l1)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)#42
fpr_list, tpr_list, auc_all=[],[],[]
test_labels, score = [], []
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train,y_train)):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

    # model = RandomForestClassifier(n_estimators=5, random_state=42)#42,,198RF
    # model = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,)#GBDT
    model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.01, max_depth=6, subsample=0.8, colsample_bytree=0.8, eval_metric='error')#XGB(0.0001)
    # model = lgb.LGBMClassifier(n_estimators=10, learning_rate=0.1)#LGBM
    # model = AdaBoostClassifier(n_estimators=50, learning_rate=0.5, random_state=42)#AdaBoost
    # model = BaggingClassifier(n_estimators=5)#Bagging



    model.fit(X_train_fold, y_train_fold)

    # 预测概率值
    y_prob = model.predict_proba(X_val_fold)[:, 1]

    # 计算 ROC 曲线和 AUC
    fpr, tpr, _ = roc_curve(y_val_fold, y_prob)
    auc_score = auc(fpr, tpr)

    # 存储结果
    test_labels.append(y_val_fold)
    score.append(y_prob)

    fpr_list.append(fpr)
    tpr_list.append(tpr)
    auc_all.append(auc_score)
    print(f"Fold {fold + 1} AUC: {auc_score:.4f}")

# 计算所有折的平均 AUC
mean_auc = np.mean(auc_all)
print(f"Average AUC across all folds: {mean_auc:.4f}")


# with open(path + '11-test_labels.txt', 'w') as file:
#     for item in test_labels:
#         file.write(f"{item}\n")
# with open(path + '11-score.txt', 'w') as file:
#     for item in score:
#         file.write(f"{item}\n")

poltting(test_labels, score, path, 5)


# # case study
# fea=np.array(fea)
# y_pre = model.predict_proba(fea)[:, 1]
# print(len(y_pre))
# # 重塑为 561 行 100 列
# y_pre_reshaped = y_pre.reshape(561, 100)
#
# # 将结果保存为 DataFrame
# y_pre_df = pd.DataFrame(y_pre_reshaped)
# y_pre_df = y_pre_df.round(4)
# # 保存到 Excel
# y_pre_df.to_excel("case_study.xlsx", index=False, header=False)
