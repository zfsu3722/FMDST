import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# 定义一个用于修剪输出的类，以确保输出尺寸与输入相同
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size  # 输入参数：chomp_size (int)

    def forward(self, x):
        # 输入: (batch_size * num_stations, channels, seq_len + padding)
        # 输出: (batch_size * num_stations, channels, seq_len) 去除多余的padding部分
        return x[:, :, :-self.chomp_size].contiguous()

# 定义一个时间块(Temporal Block)，包含两个膨胀卷积层
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # 第一层膨胀卷积
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)  # 输入: (batch_size * num_stations, n_inputs, seq_len)
        self.chomp1 = Chomp1d(padding)  # 输入: (batch_size * num_stations, n_outputs, seq_len + padding)
        self.relu1 = nn.ReLU()  # 输入: (batch_size * num_stations, n_outputs, seq_len)
        self.dropout1 = nn.Dropout(dropout)  # 输入: (batch_size * num_stations, n_outputs, seq_len)

        # 第二层膨胀卷积
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)  # 输入: (batch_size * num_stations, n_outputs, seq_len)
        self.chomp2 = Chomp1d(padding)  # 输入: (batch_size * num_stations, n_outputs, seq_len + padding)
        self.relu2 = nn.ReLU()  # 输入: (batch_size * num_stations, n_outputs, seq_len)
        self.dropout2 = nn.Dropout(dropout)  # 输入: (batch_size * num_stations, n_outputs, seq_len)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None  # 如果输入输出通道数不同，则需要下采样层
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)  # 初始化conv1权重
        self.conv2.weight.data.normal_(0, 0.01)  # 初始化conv2权重
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)  # 初始化downsample权重

    def forward(self, x):
        out = self.net(x)  # 输入: (batch_size * num_stations, n_inputs, seq_len)
        res = x if self.downsample is None else self.downsample(x)  # 输入: (batch_size * num_stations, n_inputs, seq_len)
        return self.relu(out + res)  # 输出: (batch_size * num_stations, n_outputs, seq_len)

# 定义TCN网络结构
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i  # 膨胀系数随层数指数增长
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation  # 计算填充大小
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation,
                                       padding=padding, dropout=dropout))  # 每个TemporalBlock处理输入并输出到下一个block
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)  # 输入: (batch_size * num_stations, num_inputs, seq_len)
                                # 输出: (batch_size * num_stations, num_channels[-1], seq_len)




class TCN_No_site_Transformer(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=1, site=134, his=60,
                 num_channels=[64, 64, 64, 64,64], kernel_size=3,
                 nhead=2, transformer_hidden_dim=64, num_transformer_layers=1):
        super(TCN_No_site_Transformer, self).__init__()

        self.num_sites = site
        self.output_dim = output_dim

        # 共享的 TCN 网络
        self.tcn = TemporalConvNet(
            num_inputs=input_dim,
            num_channels=num_channels,
            kernel_size=kernel_size
        )


        # Transformer Encoder Layer for learning relationships within features (i.e., across sites per feature)
        encoder_layer_features = TransformerEncoderLayer(
            d_model=site,
            nhead=nhead,
            dim_feedforward=transformer_hidden_dim
        )
        self.transformer_encoder_features = TransformerEncoder(encoder_layer_features, num_layers=num_transformer_layers)

        # 为每个站点定义独立的输出头
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_channels[-1] , 64),
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
        """
        :param x: shape (batch_size, num_stations, seq_len, input_dim)
        :return: shape (batch_size, num_stations, output_dim)
        """
        B, N, T, F = x.shape  # N 是站点数

        # 调整输入形状并经过TCN
        x = x.view(B * N, T, F).transpose(1, 2)  # -> (B*N, F, T)
        tcn_out = self.tcn(x)  # -> (B*N, C, T)

        # 取最后一个时间步
        tcn_out = tcn_out[:, :, -1]  # -> (B*N, C)

        # reshape 回 (B, N, C)
        tcn_out = tcn_out.view(B, N, -1)  # -> (B, N, C)

        # Transformer 2：学习特征间（跨站点）关系
        # 转置为 [C, B, N]，让Transformer处理"站点维度"
        transformer_input_features = tcn_out.permute(2, 0, 1)  # -> (C, B, N)
        transformer_out_features = self.transformer_encoder_features(transformer_input_features)
        transformer_out_features = transformer_out_features.permute(1, 2, 0)  # -> (B, N, C)

        # 合并两个 Transformer 的输出
        combined_representation = transformer_out_features

        # 每个站点使用自己的输出头
        outputs = []
        for i in range(N):
            out_i = self.output_heads[i](combined_representation[:, i, :])  # (B, output_dim)
            outputs.append(out_i.unsqueeze(1))  # (B, 1, output_dim)

        final_output = torch.cat(outputs, dim=1)  # (B, N, output_dim)

        return final_output