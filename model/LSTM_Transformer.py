import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class LSTM_Transformer(nn.Module):
    def __init__(self, input_dim=7, lstm_hidden_dim=64, transformer_hidden_dim=128,
                 output_dim=12, num_layers=2, num_sites=134, his=60, nhead=2, num_transformer_layers=1,
                 mem_num=20, mem_dim=64):
        super(LSTM_Transformer, self).__init__()

        self.num_sites = num_sites
        self.seq_len = his
        self.lstm_hidden_dim = lstm_hidden_dim
        self.mem_dim = mem_dim

        # LSTM 编码时序特征
        self.lstm_original = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_dim,
                                     num_layers=num_layers, batch_first=True)

        # Memory 记忆模块（全局记忆库）
        self.mem_num = mem_num
        self.memory = self.construct_memory()

        # 拼接后的维度：LSTM输出 + Memory输出
        combined_dim = lstm_hidden_dim + mem_dim
        # Transformer最终输出维度（仅保留feature Transformer这一路）
        final_dim = combined_dim

        # Transformer Encoder（仅保留特征间关系学习）
        encoder_layers = TransformerEncoderLayer(d_model=num_sites, nhead=nhead,
                                                          dim_feedforward=transformer_hidden_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                               num_layers=num_transformer_layers)

        # === 门控残差机制 ===
        # 将原始LSTM输出投影到final_dim维度，以便与Transformer输出进行门控融合
        self.lstm_residual_proj = nn.Sequential(
            nn.Linear(lstm_hidden_dim, final_dim),
            nn.LayerNorm(final_dim),
            nn.ReLU()
        )

        # 门控控制器：根据Transformer输出和原始LSTM特征生成门控信号
        self.gate_controller = nn.Sequential(
            nn.Linear(final_dim + lstm_hidden_dim, final_dim),  # 输入为Transformer输出+原始LSTM
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

        # 门控偏置初始化为负值，初始时更信任Transformer（Memory+LSTM融合）
        gate_linear = self.gate_controller[0]
        if isinstance(gate_linear, nn.Linear):
            nn.init.constant_(gate_linear.bias, -1.0)  # 初始gate接近0.27，更偏向Memory路径

    def construct_memory(self):
        """构建全局记忆库"""
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.lstm_hidden_dim, self.mem_dim), requires_grad=True)

        nn.init.xavier_normal_(memory_dict['Memory'])
        nn.init.xavier_normal_(memory_dict['Wq'])
        return memory_dict

    def query_memory(self, h_t):
        """
        查询全局记忆
        h_t: [B, N, lstm_hidden_dim]
        返回: [B, N, mem_dim]
        """
        query = torch.matmul(h_t, self.memory['Wq'])  # [B, N, mem_dim]
        att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)  # [B, N, mem_num]
        value = torch.matmul(att_score, self.memory['Memory'])  # [B, N, mem_dim]
        return value, query

    def extract_features(self, x):
        """
        返回用于 FFT 的时序特征
        shape: [B, N, T, 64]

        这里统一使用:
        每个时间步的 LSTM 输出(32) + Memory 输出(32) = 64 维
        """
        B, N, T, _ = x.shape  # Batch, Num Sites, Time Steps, Features

        # [B, N, T, F] -> [B*N, T, F]
        x_all = x.view(B * N, T, -1)

        # LSTM 输出整个时间序列
        out_ori, _ = self.lstm_original(x_all)  # [B*N, T, 32]

        # 恢复为 [B, N, T, 32]
        h_seq = out_ori.view(B, N, T, -1)

        return h_seq

    def forward(self, x):
        B, N, T, _ = x.shape  # Batch, Num Sites, Time Steps, Features

        # Step 1: LSTM 处理时序特征
        x_all = x.view(B * N, T, -1)
        out_ori, _ = self.lstm_original(x_all)  # [B*N, T, lstm_hidden_dim]
        h_ori = out_ori[:, -1]  # 取最后时刻 [B*N, lstm_hidden_dim]

        # Step 2: 恢复批次和站点维度 - [B, N, lstm_hidden_dim]
        h_t = h_ori.view(B, N, -1)  # **保存原始LSTM特征用于后续门控**

        # Step 3: 查询全局记忆
        h_att, _ = self.query_memory(h_t)  # [B, N, mem_dim]

        # Step 4: 拼接 LSTM 和 Memory 特征 - [B, N, combined_dim]
        combined = torch.cat([h_t, h_att], dim=-1)

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

        # 融合策略：gate -> 1 表示更信任原始LSTM，gate -> 0 表示更信任Transformer（Memory+LSTM融合）
        gated = gate * h_t_residual + (1 - gate) * transformer_out

        # Step 7: 每个站点使用自己的输出头进行预测
        outputs = []
        for i in range(N):
            out_i = self.output_heads[i](gated[:, i, :])  # 使用门控后的特征 [B, 1]
            outputs.append(out_i.unsqueeze(1))  # [B, 1, 1]

        final_output = torch.cat(outputs, dim=1)  # [B, N, 1]

        return final_output
