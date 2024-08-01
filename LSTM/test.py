import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


# 读取数据并只提取前1000行
df = pd.read_csv('data.csv', usecols=['date', 'close'], nrows=10)

# 数据标准化
scaler = StandardScaler()
df['close'] = scaler.fit_transform(df[['close']])

# 按时间顺序划分数据集
train_size = int(len(df) * 0.8)
val_size = int(len(df) * 0.1)

train = df[:train_size]
val = df[train_size:train_size + val_size]
test = df[train_size + val_size:]

# 将数据转换为张量
X_train = torch.FloatTensor(train[['close']].values).view(-1, 1, 1)
y_train = torch.FloatTensor(train[['close']].values)
X_val = torch.FloatTensor(val[['close']].values).view(-1, 1, 1)
y_val = torch.FloatTensor(val[['close']].values)
X_test = torch.FloatTensor(test[['close']].values).view(-1, 1, 1)
y_test = torch.FloatTensor(test[['close']].values)
print(type(X_train),X_train.shape)
print(type(y_train),y_train.shape)

