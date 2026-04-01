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


class TCN_Transformer(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=1, site=134, his=60,
                 num_channels=[64, 64, 64, 64], kernel_size=5,
                 nhead=2, transformer_hidden_dim=64, num_transformer_layers=1):
        # 注意：移除了 short_mem_num, long_mem_num, mem_dim 参数
        super(TCN_Transformer, self).__init__()

        self.num_sites = site
        self.output_dim = output_dim
        self.tcn_output_dim = num_channels[-1]

        # 共享的 TCN 网络
        self.tcn = TemporalConvNet(
            num_inputs=input_dim,
            num_channels=num_channels,
            kernel_size=kernel_size
        )

        # Transformer Encoder（仅保留跨站点特征关系学习）
        # 逻辑说明：
        # 输入经过 permute(2, 0, 1) 后形状为 [tcn_output_dim, B, site]。
        # Transformer 的 d_model 对应输入的最后一个维度，即 site (num_stations)。
        # 序列长度 (Sequence Length) 为 tcn_output_dim。
        encoder_layer= TransformerEncoderLayer(
            d_model=site,
            nhead=nhead if site % nhead == 0 else max(1, site // 2),  # 防止 nhead 无法整除 site
            dim_feedforward=transformer_hidden_dim
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer,
                                                               num_layers=num_transformer_layers)

        # === 门控残差机制 ===
        # 移除 memory 后，Transformer 的输出维度变回 tcn_output_dim (因为 permute 回去后最后一维是 tcn_output_dim)
        combined_dim = self.tcn_output_dim

        # 将 TCN 输出投影到 combined_dim 维度
        # 由于此时输入输出维度相同，这一步主要提供非线性变换，也可简化，但保留以维持架构一致性
        self.tcn_to_combined_proj = nn.Sequential(
            nn.Linear(self.tcn_output_dim, combined_dim),
            nn.LayerNorm(combined_dim),
            nn.ReLU()
        )

        # 门控网络：输入是 Transformer 输出 + 原始 TCN 特征
        # 输入维度: combined_dim (Transformer部分) + tcn_output_dim (原始TCN部分)
        self.gate_controller = nn.Sequential(
            nn.Linear(combined_dim + self.tcn_output_dim, combined_dim),
            nn.Sigmoid()
        )

        # 输出头（输入维度为 combined_dim）
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(combined_dim, 64),
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

        # 门控偏置初始化为负值，让网络初始时更倾向于使用 Transformer 路径
        if isinstance(self.gate_controller[0], nn.Linear):
            init.constant_(self.gate_controller[0].bias, -2.0)

    def extract_features(self, x):
        """
        返回用于 FFT 的时序特征
        shape: [B, N, T, 64]

        这里直接使用 TCN 在每个时间步的输出特征
        """
        B, N, T, F = x.shape

        # [B, N, T, F] -> [B*N, F, T]
        x = x.view(B * N, T, F).transpose(1, 2)

        # TCN 输出 [B*N, 64, T]
        tcn_out = self.tcn(x)

        # -> [B*N, T, 64]
        tcn_out = tcn_out.transpose(1, 2)

        # -> [B, N, T, 64]
        tcn_out = tcn_out.view(B, N, T, -1)

        return tcn_out

    def forward(self, x):
        """
        :param x: shape (batch_size, num_stations, seq_len, input_dim)
        :return: shape (batch_size, num_stations, output_dim)
        """
        B, N, T, F = x.shape

        # Step 1: TCN 处理时序特征
        x = x.view(B * N, T, F).transpose(1, 2)  # [B*N, F, T]
        tcn_out = self.tcn(x)
        tcn_out = tcn_out[:, :, -1]
        tcn_out = tcn_out.view(B, N, -1)  # [B, N, tcn_output_dim]

        # === 移除 Step 2: 查询全局记忆 ===
        # 直接使用 tcn_out 作为后续输入
        h_base = tcn_out

        # Step 3: Transformer 学习跨站点特征关系
        # 输入变换: [B, N, Dim] -> [Dim, B, N]
        # 此时 Sequence Length = Dim (tcn_output_dim), Feature Dim = N (site)
        transformer_input = h_base.permute(2, 0, 1)

        transformer_out = self.transformer_encoder(transformer_input)

        # 变回 [B, N, Dim]
        transformer_out = transformer_out.permute(1, 2, 0)

        # Step 4: 门控残差融合
        # 投影 TCN 特征
        tcn_residual = self.tcn_to_combined_proj(tcn_out)  # [B, N, tcn_output_dim]

        # 计算门控信号
        gate_input = torch.cat([transformer_out, tcn_out], dim=-1)
        gate = self.gate_controller(gate_input)  # [B, N, tcn_output_dim]

        # 融合
        gated = gate * tcn_residual + (1 - gate) * transformer_out

        # Step 5: 每个站点使用独立输出头预测
        outputs = []
        for i in range(N):
            out_i = self.output_heads[i](gated[:, i, :])
            outputs.append(out_i.unsqueeze(1))

        final_output = torch.cat(outputs, dim=1)

        return final_output

