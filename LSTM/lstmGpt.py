import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 读取数据并只提取前1000行
df = pd.read_csv('data.csv', usecols=['date', 'close'])

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


# 定义LSTM网络结构
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=1000, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 构建全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化h0，c0
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # 提取最后一个时间步的隐藏状态
        out = self.fc(out[:, -1, :])
        return out


# 实例化模型、损失函数和优化器
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 1500
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()  # 训练模式
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()  # 评估模式
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        val_losses.append(val_loss.item())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Val Loss: {val_loss.item()}')

# 绘制训练误差和验证误差曲线
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 评估模型在测试集上的性能
model.eval()  # 评估模式
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f'Test Loss: {test_loss.item()}')


# 预测部分
def predict(model_para, input_data):
    #这里修改名字为了不让形参和上面的实例model重名，导致内层变量覆盖或“遮蔽”外层变量，从而可能引起混淆或意外的行为
    model_para.eval()
    with torch.no_grad():
        predictions = model_para(input_data)
    return predictions


# 使用模型进行预测
predicted_values = predict(model, X_test).numpy()

# # 绘制预测结果与真实值对比图
# plt.plot(y_test.flatten(), label='True Values')
# plt.plot(predicted_values.flatten(), label='Predicted Values')
# plt.xlabel('Time Step')
# plt.ylabel('Value')
# plt.legend()
# plt.show()
# 将标准化后的预测结果逆标准化
test_outputs_original = scaler.inverse_transform(test_outputs)
y_test_original = scaler.inverse_transform(y_test.numpy())

# 绘制预测结果与真实值对比图
plt.title("cell1000,1 layer,1500epc")
plt.plot(y_test_original, label='True Values')
plt.plot(test_outputs_original, label='Predicted Values')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.show()
# 绘制预测结果与真实值差值图
plt.title("cell1000,1 layer,1500epc")
plt.plot(y_test_original - test_outputs_original, label='Diff')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.show()
