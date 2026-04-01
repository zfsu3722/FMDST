import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


# 定义一个时间块(Temporal Block)，包含两个膨胀卷积层
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
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
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
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



class LSTM_TCN_Transformer(nn.Module):
    def __init__(self,
                 input_dim=7,
                 # --- 用户指定超参数 ---
                 lstm_hidden_dim=32,
                 tcn_num_channels=[64, 64, 64, 64],
                 mem_dim=32,
                 transformer_hidden_dim=128,
                 nhead=2,
                 dropout=0.1,
                 # --- 其他固定/默认参数 ---
                 output_dim=12,
                 num_sites=134,
                 his=60,
                 num_transformer_layers=1,
                 lstm_num_layers=2,
                 mem_num=20,
                 tcn_kernel_size=5):
        super(LSTM_TCN_Transformer, self).__init__()

        self.num_sites = num_sites
        self.lstm_hidden_dim = lstm_hidden_dim
        self.tcn_output_dim = tcn_num_channels[-1]  # 64
        self.mem_dim = mem_dim

        # 1. 计算基础融合维度：LSTM(32) + Memory(32) + TCN(64) = 128
        self.base_dim = lstm_hidden_dim + mem_dim + self.tcn_output_dim

        # 2. 【修改】最终维度 = 仅 Feature Transformer 输出 (不再 * 2)
        self.final_dim = self.base_dim

        # ====== 1. LSTM 分支 (含 Memory) ======
        self.lstm_original = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True
        )

        # Memory 模块参数
        self.mem_num = mem_num
        self.memory_params = nn.ParameterDict({
            'Memory': nn.Parameter(torch.randn(mem_num, mem_dim)),
            'Wq': nn.Parameter(torch.randn(lstm_hidden_dim, mem_dim))
        })
        init.xavier_normal_(self.memory_params['Memory'])
        init.xavier_normal_(self.memory_params['Wq'])

        # ====== 2. TCN 分支 ======
        self.tcn = TemporalConvNet(
            num_inputs=input_dim,
            num_channels=tcn_num_channels,
            kernel_size=tcn_kernel_size,
            dropout=dropout
        )

        # ====== 3. 单 Transformer 编码器 (仅 Feature/Channel) ======

        # 检查 nhead 是否能整除 num_sites
        if num_sites % nhead != 0:
            nhead_feat = max(1, num_sites // 2)
            print(
                f"Warning: num_sites ({num_sites}) not divisible by nhead ({nhead}). Adjusting feature encoder nhead to {nhead_feat}")
        else:
            nhead_feat = nhead

        # Encoder: 特征间关系 (Channel)
        # d_model = num_sites (134)
        encoder_layer_features = TransformerEncoderLayer(
            d_model=num_sites,
            nhead=nhead_feat,
            dim_feedforward=transformer_hidden_dim
        )
        self.transformer_encoder_features = TransformerEncoder(encoder_layer_features,
                                                               num_layers=num_transformer_layers)

        # ====== 4. 门控残差机制 (Gated Residual) ======

        # 4.1 投影层：将原始三路拼接特征 (base_dim) 映射到 final_dim (base_dim)
        self.residual_proj = nn.Sequential(
            nn.Linear(lstm_hidden_dim + self.tcn_output_dim, self.final_dim),
            nn.LayerNorm(self.final_dim),
            nn.ReLU()
        )

        # 4.2 门控控制器
        # 输入维度：final_dim (Transformer) + base_dim (Original) = 2 * base_dim
        # 输出维度：final_dim
        self.gate_controller = nn.Sequential(
            nn.Linear(self.final_dim + self.base_dim, self.final_dim),
            nn.Sigmoid()
        )

        # ====== 5. 输出头 (恢复为多输出头 ModuleList) ======
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.final_dim, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            ) for _ in range(num_sites)
        ])

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        init.xavier_normal_(param)
                    elif 'bias' in name:
                        init.constant_(param, 0)

        if len(self.gate_controller) > 0 and isinstance(self.gate_controller[0], nn.Linear):
            init.constant_(self.gate_controller[0].bias, -2.0)

    def query_memory(self, h_t):
        query = torch.matmul(h_t, self.memory_params['Wq'])
        att_score = torch.softmax(torch.matmul(query, self.memory_params['Memory'].t()), dim=-1)
        value = torch.matmul(att_score, self.memory_params['Memory'])
        return value

    def forward(self, x):
        B, N, T, F = x.shape

        # =======================
        # Step 1: 并行提取三大特征
        # =======================

        # --- A. LSTM 分支 ---
        x_lstm = x.view(B * N, T, F)
        out_ori, _ = self.lstm_original(x_lstm)
        h_lstm = out_ori[:, -1].view(B, N, -1)  # [B, N, 32]

        # --- B. Memory 分支 ---
        h_mem = self.query_memory(h_lstm)  # [B, N, 32]

        # --- C. TCN 分支 ---
        x_tcn = x.view(B * N, T, F).transpose(1, 2)
        tcn_out = self.tcn(x_tcn)
        h_tcn = tcn_out[:, :, -1].view(B, N, -1)  # [B, N, 64]

        lstm_tcn_out = torch.cat([h_lstm,  h_tcn], dim=-1)
        # =======================
        # Step 2: 三路特征拼接
        # =======================
        combined_input = torch.cat([h_lstm, h_mem, h_tcn], dim=-1)  # [B, N, 128]

        # =======================
        # Step 3: 单 Transformer 编码 (仅 Feature/Channel)
        # =======================

        # Input: [B, N, D] -> Transpose to [N, B, D] for batch_first=False
        trans_in_features = combined_input.permute(2, 0, 1)

        # Forward
        trans_out_features = self.transformer_encoder_features(trans_in_features)

        # Output: [N, B, D] -> Transpose back to [B, N, D]
        transformer_repr = trans_out_features.permute(1, 2, 0)  # [B, N, 128]

        # =======================
        # Step 4: 门控残差融合
        # =======================

        # 4.1 投影原始特征
        projected_residual = self.residual_proj(lstm_tcn_out)

        # 4.2 计算门控信号
        gate_input = torch.cat([transformer_repr, projected_residual], dim=-1)
        gate = self.gate_controller(gate_input)  # [B, N, 128]

        # 4.3 动态融合
        gated_features = gate * projected_residual + (1 - gate) * transformer_repr

        # =======================
        # Step 5: 输出预测 (恢复多输出头循环)
        # =======================
        outputs = []
        for i in range(N):
            # 使用每个站点独立的头
            out_i = self.output_heads[i](gated_features[:, i, :])
            outputs.append(out_i.unsqueeze(1))

        final_output = torch.cat(outputs, dim=1)  # [B, N, output_dim]

        return final_output


