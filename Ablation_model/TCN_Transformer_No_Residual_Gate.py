import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# 定义一个用于修剪输出的类，以确保输出尺寸与输入相同
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


# 定义一个时间块(Temporal Block)，包含两个膨胀卷积层
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# 定义TCN网络结构
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.1):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation,
                                        padding=padding, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN_Transformer_No_Residual_Gate(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=12, site=134, his=60,
                 num_channels=[64, 64, 64, 64], kernel_size=5,
                 nhead=2, transformer_hidden_dim=64, num_transformer_layers=1):
        super(TCN_Transformer_No_Residual_Gate, self).__init__()

        self.num_sites = site
        self.output_dim = output_dim
        self.tcn_output_dim = num_channels[-1]

        # 共享的 TCN 网络
        self.tcn = TemporalConvNet(
            num_inputs=input_dim,
            num_channels=num_channels,
            kernel_size=kernel_size
        )

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=site,
            nhead=nhead if site % nhead == 0 else max(1, site // 2),
            dim_feedforward=transformer_hidden_dim
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer,
                                                      num_layers=num_transformer_layers)

        # 输出头（输入维度为 tcn_output_dim）
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.tcn_output_dim, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            ) for _ in range(site)
        ])

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        B, N, T, F = x.shape

        # Step 1: TCN 处理时序特征
        x = x.view(B * N, T, F).transpose(1, 2)  # [B*N, F, T]
        tcn_out = self.tcn(x)
        tcn_out = tcn_out[:, :, -1]
        tcn_out = tcn_out.view(B, N, -1)  # [B, N, tcn_output_dim]

        # Step 2: Transformer 学习跨站点特征关系
        transformer_input = tcn_out.permute(2, 0, 1)  # [Dim, B, N]
        transformer_out = self.transformer_encoder(transformer_input)
        transformer_out = transformer_out.permute(1, 2, 0)  # [B, N, Dim]

        # Step 3: 直接使用 Transformer 输出给到多头
        final_features = transformer_out

        outputs = []
        for i in range(N):
            out_i = self.output_heads[i](final_features[:, i, :])
            outputs.append(out_i.unsqueeze(1))

        final_output = torch.cat(outputs, dim=1)

        return final_output

