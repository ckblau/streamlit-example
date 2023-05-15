import numpy as np
import pandas as pd
import torch
from torch import nn

from utils import train_model, pred

# 2344个训练集样本 400个测试集样本
train = pd.read_csv('./data/train_dropped.csv')
test = pd.read_csv('./data/test_dropped.csv')

all_features = pd.concat((train.loc[:, 'MS SubClass':'Sale Condition'],
                          test.loc[:, 'MS SubClass':'Sale Condition']))

# 取出所有的数值特征
numeric_feats = all_features.dtypes[all_features.dtypes != "object"].index
# 减去均值，除以方差
all_features[numeric_feats] = all_features[numeric_feats].apply(lambda x: (x - x.mean()) / (x.std()))
# 对预测的价格取 log
train['SalePrice'] = np.log(train['SalePrice'])
# 对非数值进行编码
non_numeric = all_features.dtypes[all_features.dtypes == "object"].index
for attr in non_numeric:
    all_features[attr] = pd.factorize(all_features[attr])[0]
# 将none填充为列均值
all_features = all_features.fillna(all_features.mean())
feat_dim = all_features.shape[1]

num_train = int(0.9 * train.shape[0])  # 划分训练样本和验证集样本
indices = np.arange(train.shape[0])
np.random.shuffle(indices)  # shuffle 顺序
train_indices = indices[:num_train]
valid_indices = indices[num_train:]

# 提取训练集和验证集的特征
train_features = all_features.iloc[train_indices].values.astype(np.float32)
train_features = torch.from_numpy(train_features)
valid_features = all_features.iloc[valid_indices].values.astype(np.float32)
valid_features = torch.from_numpy(valid_features)
train_valid_features = all_features[:train.shape[0]].values.astype(np.float32)
train_valid_features = torch.from_numpy(train_valid_features)

# 提取训练集和验证集的label
train_labels = train['SalePrice'].values[train_indices, None].astype(np.float32)
train_labels = torch.from_numpy(train_labels)
valid_labels = train['SalePrice'].values[valid_indices, None].astype(np.float32)
valid_labels = torch.from_numpy(valid_labels)
train_valid_labels = train['SalePrice'].values[:, None].astype(np.float32)
train_valid_labels = torch.from_numpy(train_valid_labels)
# 提取测试集的特征
test_features = all_features[train.shape[0]:].values.astype(np.float32)
test_features = torch.from_numpy(test_features)


def get_model():
    _net = nn.Sequential(
        nn.Linear(feat_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.ReLU()
    )
    return _net


# 可以调整的超参
batch_size = 128
epochs = 100
lr = 0.01
wd = 0
use_gpu = False

net = get_model()
train_model(net, train_features, train_labels, valid_features, valid_labels, epochs,
            batch_size, lr, wd, use_gpu)

pred(net, test, test_features)
