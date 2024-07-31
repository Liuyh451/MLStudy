import pandas as pd
import matplotlib.pyplot as plt
import torch

# 读取同目录下的CSV文件
df = pd.read_csv('data.csv')
# 提取' date' 和 'close'列
data = df[['date', 'close']]

# 显示新的DataFrame的前几行
print(data.head())
print(data.shape)

# # 设置 'date' 列为索引
# data.set_index('date', inplace=True)
#
# # 绘制折线图
# plt.figure(figsize=(10, 5))
# plt.plot(data.index, data['close'], marker='o', linestyle='-', color='b')
#
# # 添加标题和标签
# plt.title('Close Prices Over Time')
# plt.xlabel('Date')
# plt.ylabel('Close Price')
#
# # 显示图表
# plt.grid(True)
# plt.show()
# 测试集划分
# 不能改变数据集的时间顺序，所以不能使用train_test_split函数
timeseries = data[["close"]].values.astype('float32')
train_size = int(len(timeseries) * 0.7)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]
print(train.shape, test.shape)
# 判断数据集是否为张量
is_tensor_train = torch.is_tensor(train)
# print(is_tensor_train)
# 把数据集转为张量
train_tensor = torch.FloatTensor(train).view(-1, train.shape[0], 1)
# view 方法用于对 tensor 进行重新形状（reshape），而不改变其数据,-1这个参数表示自动推断维度的大小,并将 tensor 变为三维
test_tensor = torch.FloatTensor(test).view(-1, test.shape[0], 1)


def MSE(Y_ture, Y_predict):
    return ((Y_ture - Y_predict) ** 2).sum() / Y_ture.shape[0]


# 偏方：求训练（测试）集与训练（测试）集数据的平均数之间的mse，loss值应小于这个数，即l网络的预测效果要比直接求均值好
print("mse_train", MSE(train, train.mean()), "mse_test", MSE(test, test.mean()))


def get_data():
    return train_tensor, test_tensor,test
