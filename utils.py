# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


def encode(df, norm):
    mapping = {}
    for attr in norm:
        df[attr], mapping[attr] = pd.factorize(df[attr])
    return df, mapping


def decode(df, norm, mapping):
    for attr in norm:
        df[attr] = mapping[attr][df[attr]]
    return df


def get_data(x, y, batch_size, shuffle):
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size, shuffle=shuffle)


def train_model(model, x_train, y_train, x_valid, y_valid, epochs, batch_size, lr, weight_decay, use_gpu):
    if use_gpu:
        model = model.cuda()
    metric_log = defaultdict(list)

    train_data = get_data(x_train, y_train, batch_size, True)
    if x_valid is not None:
        valid_data = get_data(x_valid, y_valid, batch_size, False)
    else:
        valid_data = None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    for e in range(epochs):
        # 训练模型
        model.train()
        for data in train_data:
            x, y = data
            if use_gpu:
                x = x.cuda()
                y = y.cuda()
            # forward
            out = model(x)
            loss = criterion(out, y)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        metric_log['train_rmse'].append(get_rmse(model, x_train, y_train, use_gpu))

        # 测试模型
        if x_valid is not None:
            metric_log['valid_rmse'].append(get_rmse(model, x_valid, y_valid, use_gpu))
            print_str = 'epoch: {}, train rmse: {:.3f}, valid rmse: {:.3f}' \
                .format(e + 1, metric_log['train_rmse'][-1], metric_log['valid_rmse'][-1])
        else:
            print_str = 'epoch: {}, train rmse: {:.3f}'.format(e + 1, metric_log['train_rmse'][-1])
        if (e + 1) % 10 == 0:
            print(print_str)
            print()
    torch.save(model, 'model_dict.pt')
    # 可视化
    figsize = (10, 5)
    plt.figure(figsize=figsize)
    plt.plot(metric_log['train_rmse'], color='red', label='train')
    if valid_data is not None:
        plt.plot(metric_log['valid_rmse'], color='blue', label='valid')
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()


def get_rmse(model, feature, label, use_gpu):
    if use_gpu:
        feature = feature.cuda()
        label = label.cuda()
    model.eval()
    mse_loss = nn.MSELoss()
    with torch.no_grad():
        output = model(feature)
    # clipped_pred = output.clamp(1, float('inf'))
    rmse = (mse_loss(output, label)).sqrt()
    return rmse.item()


def pred(net, test_data, test_features):
    net = net.eval()
    net = net.cpu()
    with torch.no_grad():
        preds = net(test_features)
    preds = np.exp(preds.numpy())
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


def data_processed(data: dict):
    mean_data = pd.read_csv('./data/AmesHousing_dropped_1.csv')
    # feat_dim = 47
    # 取出所有的数值特征
    numeric_feats = mean_data.dtypes[mean_data.dtypes != "object"].index
    # 减去均值，除以方差
    mean_data[numeric_feats] = mean_data[numeric_feats].apply(lambda x: (x - x.mean()) / (x.std()))
    # 对非数值进行编码
    non_numeric = mean_data.dtypes[mean_data.dtypes == "object"].index
    mean_data, mapping = encode(mean_data, non_numeric)
    # 将none填充为列均值
    mean_data = mean_data.mean()
    # 将待预测数据中非数值属性转化为数值属性
    for attr in non_numeric:
        if data.get(attr) is None:
            data[attr] = mean_data[attr]
        if data[attr] is not None:
            for i in range(len(mapping[attr])):
                if mapping[attr][i] == data[attr]:
                    data[attr] = i
                    break
    # 将待预测数据中缺省值填充为均值
    for attr in numeric_feats:
        if data[attr] == -1:
            data[attr] = mean_data[attr]
    # 将待预测数据转为tensor类型
    tmp = []
    for value in data.values():
        tmp.append(value)
    data = torch.tensor(np.array(tmp, dtype=np.float32))

    return data
