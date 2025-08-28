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




class TCN_Transformer_No_multi_head(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=1, site=134, his=60,
                 num_channels=[64, 64, 64, 64,64], kernel_size=3,
                 nhead=2, transformer_hidden_dim=64, num_transformer_layers=1):
        super(TCN_Transformer_No_multi_head, self).__init__()

        self.num_sites = site
        self.output_dim = output_dim

        # 共享的 TCN 网络
        self.tcn = TemporalConvNet(
            num_inputs=input_dim,
            num_channels=num_channels,
            kernel_size=kernel_size
        )

        # Transformer Encoder Layer for learning relationships between sites
        encoder_layer_sites = TransformerEncoderLayer(
            d_model=num_channels[-1],
            nhead=nhead,
            dim_feedforward=transformer_hidden_dim
        )
        self.transformer_encoder_sites = TransformerEncoder(encoder_layer_sites, num_layers=num_transformer_layers)

        # Transformer Encoder Layer for learning relationships within features (i.e., across sites per feature)
        encoder_layer_features = TransformerEncoderLayer(
            d_model=site,
            nhead=nhead,
            dim_feedforward=transformer_hidden_dim
        )
        self.transformer_encoder_features = TransformerEncoder(encoder_layer_features, num_layers=num_transformer_layers)

        # ✅ 共享输出头（不再是每个站点一个）
        self.shared_output_head = nn.Sequential(
            nn.Linear(num_channels[-1] * 2, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

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

        # Transformer 1：学习站点间关系
        transformer_input_sites = tcn_out.transpose(0, 1)  # -> (N, B, C)
        transformer_out_sites = self.transformer_encoder_sites(transformer_input_sites)
        transformer_out_sites = transformer_out_sites.transpose(0, 1)  # -> (B, N, C)

        # Transformer 2：学习特征间（跨站点）关系
        transformer_input_features = tcn_out.permute(2, 0, 1)  # -> (C, B, N)
        transformer_out_features = self.transformer_encoder_features(transformer_input_features)
        transformer_out_features = transformer_out_features.permute(1, 2, 0)  # -> (B, N, C)

        # 合并两个 Transformer 的输出
        combined_representation = torch.cat([
            transformer_out_sites,
            transformer_out_features
        ], dim=-1)  # -> (B, N, 2*C)

        # ✅ 使用共享的输出头处理每个站点的表示
        # combined_representation: (B, N, 2*C)
        # 将站点维度展开成 batch 维度的一部分
        B, N, _ = combined_representation.shape
        combined_representation = combined_representation.view(B * N, -1)  # -> (B*N, 2*C)
        outputs = self.shared_output_head(combined_representation)  # -> (B*N, output_dim)
        outputs = outputs.view(B, N, -1)  # -> (B, N, output_dim)

        return outputs