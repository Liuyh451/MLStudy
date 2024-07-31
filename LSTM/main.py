import dataPrepro
import torch
import torch.nn as nn

# import data from datapreprocess
train, test = dataPrepro.get_data()
print(train.shape, test.shape)  # batch_size,time_step_input_dimension


# 网络结构
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)
