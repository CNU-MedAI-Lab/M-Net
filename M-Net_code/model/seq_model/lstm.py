import torch
import torch.nn as nn


class SameInOutLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SameInOutLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM层
        out, _ = self.lstm(x, (h0, c0))

        # 全连接层
        out = self.fc(out)
        return out


# 示例用法
input_size = 80*80
hidden_size = int(input_size/4)
num_layers = 5
output_size = input_size  # 输出维度与输入维度相同

# 构建模型
model = SameInOutLSTM(input_size, hidden_size, num_layers, output_size)

# 示例输入数据
batch_size = 1
seq_len = 15
input_data = torch.randn(batch_size, seq_len, input_size)

# 前向传播
print(input_data.shape)
output = model(input_data)
print("Output shape:", output.shape)  # 应与输入数据的形状相同
