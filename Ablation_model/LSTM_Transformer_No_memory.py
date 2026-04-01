import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer



class LSTM_Transformer_No_Memory(nn.Module):
    def __init__(self, input_dim=7, lstm_hidden_dim=32, transformer_hidden_dim=64,
                 output_dim=1, num_layers=2, num_sites=134, his=60, nhead=2, num_transformer_layers=1):
        super(LSTM_Transformer_No_Memory, self).__init__()

        self.num_sites = num_sites
        self.seq_len = his
        self.lstm_hidden_dim = lstm_hidden_dim

        # LSTM 编码时序特征
        self.lstm_original = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_dim,
                                     num_layers=num_layers, batch_first=True)

        # 移除 Memory 模块后，维度直接为 lstm_hidden_dim
        final_dim = lstm_hidden_dim

        # Transformer Encoder（仅保留特征间关系学习）
        encoder_layers = TransformerEncoderLayer(d_model=num_sites, nhead=nhead,
                                                          dim_feedforward=transformer_hidden_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                               num_layers=num_transformer_layers)

        # === 门控残差机制 ===
        # 将原始LSTM输出投影到final_dim维度 (此时输入输出维度相同，主要提供非线性)
        self.lstm_residual_proj = nn.Sequential(
            nn.Linear(lstm_hidden_dim, final_dim),
            nn.LayerNorm(final_dim),
            nn.ReLU()
        )

        # 门控控制器：根据Transformer输出和原始LSTM特征生成门控信号
        self.gate_controller = nn.Sequential(
            nn.Linear(final_dim + lstm_hidden_dim, final_dim),
            nn.Sigmoid()
        )

        # 每个站点一个输出头（输入维度为 final_dim）
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(final_dim, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            ) for _ in range(num_sites)
        ])

        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() >= 2:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

        # 门控偏置初始化为负值
        gate_linear = self.gate_controller[0]
        if isinstance(gate_linear, nn.Linear):
            nn.init.constant_(gate_linear.bias, -1.0)

    def forward(self, x):
        B, N, T, _ = x.shape  # Batch, Num Sites, Time Steps, Features

        # Step 1: LSTM 处理时序特征
        x_all = x.view(B * N, T, -1)
        out_ori, _ = self.lstm_original(x_all)  # [B*N, T, lstm_hidden_dim]
        h_ori = out_ori[:, -1]  # 取最后时刻 [B*N, lstm_hidden_dim]

        # Step 2: 恢复批次和站点维度 - [B, N, lstm_hidden_dim]
        h_t = h_ori.view(B, N, -1)  # **保存原始LSTM特征用于后续门控**

        # === 移除 Step 3 & 4: 查询全局记忆与拼接 ===
        # 直接使用 h_t 作为 combined 特征
        combined = h_t

        # Step 5: Transformer 学习特征间关系（跨站点特征维度）
        transformer_input = combined.permute(2, 0, 1)  # [combined_dim, B, N]
        transformer_out = self.transformer_encoder(transformer_input)
        transformer_out = transformer_out.permute(1, 2, 0)  # [B, N, combined_dim]

        # Step 6: 门控残差融合（核心）
        # 投影原始LSTM特征到final_dim维度
        h_t_residual = self.lstm_residual_proj(h_t)  # [B, N, final_dim]

        # 计算门控信号：基于Transformer输出和原始LSTM特征共同决定
        gate_input = torch.cat([transformer_out, h_t], dim=-1)  # [B, N, final_dim + lstm_hidden_dim]
        gate = self.gate_controller(gate_input)  # [B, N, final_dim]，值域 0-1

        # 融合策略
        gated = gate * h_t_residual + (1 - gate) * transformer_out

        # Step 7: 每个站点使用自己的输出头进行预测
        outputs = []
        for i in range(N):
            out_i = self.output_heads[i](gated[:, i, :])
            outputs.append(out_i.unsqueeze(1))

        final_output = torch.cat(outputs, dim=1)

        return final_output

