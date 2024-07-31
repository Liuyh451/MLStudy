import dataPrepro
import torch
import torch.nn as nn

# import data from datapreprocess
train, test,test_or = dataPrepro.get_data()
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
learning_rate = 0.1
num_epochs = 300
# 实例化模型
model = LSTMModel(input_size, hidden_size, num_layers, output_size)  # 实例化
# 定义损失函数
criterion = nn.MSELoss(reduction="mean")
# 定义优化算法
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 开始进行训练的循环
for epoch in range(num_epochs):
    outputs = model(train)
    optimizer.zero_grad()  # 将梯度清0
    loss = criterion(outputs, train[:, :, :])
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 150 == 0:
        print(f'Epoch[{epoch + 1}/{num_epochs}],Loss:{loss.item()}')
print("eval", model.eval())
test_outputs = model(test).detach().numpy()
print(MSE(test_or, test_outputs))
