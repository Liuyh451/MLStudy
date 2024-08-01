import dataPrepro
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# import data from datapreprocess
train, test, test_or = dataPrepro.get_data()
print(train.shape, test.shape)  # batch_size,time_step_input_dimension


# 网络结构
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        # input_size：每个时间步输入的特征数。
        # hidden_size：LSTM单元的隐藏状态维度。
        # num_layers：堆叠LSTM层的数量。
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)

    # 定义前向传播
    def forward(self, x):
        h0 = torch.rand(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(x.device)
        c0 = torch.rand(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(x.device)
        # lstm层输出的是output，hn和cn,这里需要的是output
        output, (_, _) = self.lstm(x, (h0.detach(), c0.detach()))
        # 格式为[batch_size, seq_len, hidden_size], 取我们有用的维度
        out = self.fc(output[:, :, :])
        return out


def MSE(Y_ture, Y_predict):
    return ((Y_ture - Y_predict) ** 2).sum() / Y_ture.shape[0]


# 超参数设置
input_size = 1
hidden_size = 100
num_layers = 5
output_size = 1
learning_rate = 0.01
num_epochs = 1500
# 实例化模型
model = LSTMModel(input_size, hidden_size, num_layers, output_size)  # 实例化
# 定义损失函数
criterion = nn.MSELoss(reduction="mean")
# 定义优化算法
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 记录训练误差和验证误差
train_losses = []
val_losses = []
# 开始进行训练的循环
for epoch in range(num_epochs):
    #model.train()
    outputs = model(train)
    optimizer.zero_grad()  # 将梯度清0
    loss = criterion(outputs, train[:, :, :])
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    if (epoch + 1) % 100 == 0:
        print(f'Epoch[{epoch + 1}/{num_epochs}],Loss:{loss.item()}')
        # 验证阶段
    #model.eval()
    # with torch.no_grad():
    #     val_outputs = model(val)
    #     val_loss = criterion(val_outputs, val)
    #     val_losses.append(val_loss.item())
# test_outputs = model(test).detach().numpy()
# print(MSE(test_or, test_outputs))
# # 绘制训练误差和验证误差曲线
# plt.plot(train_losses, label='Train Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# 最终评估模型在测试集上的性能
model.eval()
with torch.no_grad():
    test_outputs = model(test).detach().numpy()
    test_loss = MSE(test_or, test_outputs)
    print(f'Test Loss: {test_loss}')


# 预测部分
def predict(model, input_data):
    model.eval()
    with torch.no_grad():
        predictions = model(input_data)
    return predictions


# 使用模型进行预测
predicted_values = predict(model, test)

# 转换为numpy数组并进行处理
predicted_values = predicted_values.numpy()
# 绘制预测结果与真实值对比图
plt.plot(test_or.flatten(), label='True Values')
plt.plot(predicted_values.flatten(), label='Predicted Values')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.show()
