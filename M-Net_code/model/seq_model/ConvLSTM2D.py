import torch
import torch.nn as nn
import numpy as np


# 定义ConvLSTM单元
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # Concatenate input and previous hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


# 定义ConvLSTM层
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([ConvLSTMCell(input_dim if i == 0 else hidden_dim,
                                                  hidden_dim, kernel_size)
                                     for i in range(num_layers)])

    def forward(self, input_tensor):
        batch_size, seq_len, _, height, width = input_tensor.size()

        h, c = self.init_hidden(batch_size, (height, width))

        output_inner = []
        for t in range(seq_len):
            for layer_idx in range(self.num_layers):
                h[layer_idx], c[layer_idx] = self.layers[layer_idx](
                    input_tensor[:, t, :, :, :],
                    (h[layer_idx], c[layer_idx])
                )
            output_inner.append(h[-1])

        return torch.stack(output_inner, dim=1)

    def init_hidden(self, batch_size, image_size):
        # 使用列表保存 h 和 c 的状态
        h, c = [], []
        for i in range(self.num_layers):
            h_i, c_i = self.layers[i].init_hidden(batch_size, image_size)
            h.append(h_i)
            c.append(c_i)
        return h, c


# 分块函数，将图像分块
def split_image_to_patches(image, patch_size):
    batch_size, channels, height, width = image.shape
    patches = []
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = image[:, :, i:i + patch_size, j:j + patch_size]
            patches.append(patch)
    return torch.stack(patches, dim=1)  # 在时间维度上堆叠块

def restore_image_from_patches(patches, original_image_size, patch_size):
    batch_size, seq_len, channels, patch_height, patch_width = patches.shape
    original_height, original_width = original_image_size
    restored_image = torch.zeros(batch_size, channels, original_height, original_width)

    patch_idx = 0
    for i in range(0, original_height, patch_size):
        for j in range(0, original_width, patch_size):
            restored_image[:, :, i:i+patch_size, j:j+patch_size] = patches[:, patch_idx, :, :, :]
            patch_idx += 1

    return restored_image



# 模型测试
if __name__ == "__main__":
    # 假设输入图像尺寸为 (batch_size, channels, height, width)
    batch_size = 192
    channels = 15
    height, width = 20, 20
    patch_size = 5

    # 生成随机图像数据
    input_image = torch.rand(batch_size, channels, height, width)

    # 分块图像
    patches = split_image_to_patches(input_image, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4)

    # 将分块图像转为时间序列 (batch_size, seq_len, channels, height, width)
    # patches 本身已经是 (batch_size, seq_len, channels, height, width)

    # 定义ConvLSTM模型
    input_dim = 16
    hidden_dim = (20//5)*(20//5)
    kernel_size = 3
    num_layers = 1
    model = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers)
    ln1 = nn.LayerNorm(hidden_dim)

    # 前向传播
    print("Input shape:", input_image.shape, patches.shape)
    output = model(patches)
    print("ConvLSTM output shape:", output.shape)
    output = output.permute(0, 1, 3, 4, 2)
    output = ln1(output)
    print("ConvLSTM output shape:", output.shape)
    output = output.permute(0, 4, 1, 2, 3)
    # 恢复分块后的图像
    restored_image = restore_image_from_patches(output, (height, width), patch_size)
    print("Restored image shape:", restored_image.shape)
