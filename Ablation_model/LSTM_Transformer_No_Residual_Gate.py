import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class LSTM_Transformer_No_Residual_Gate(nn.Module):
    def __init__(self, input_dim=7, lstm_hidden_dim=32, transformer_hidden_dim=64,
                 output_dim=1, num_layers=2, num_sites=134, his=60, nhead=2, num_transformer_layers=1,
                 mem_num=20, mem_dim=32):
        super(LSTM_Transformer_No_Residual_Gate, self).__init__()

        self.num_sites = num_sites
        self.seq_len = his
        self.lstm_hidden_dim = lstm_hidden_dim
        self.mem_dim = mem_dim

        # 1. LSTM 编码时序特征
        self.lstm_original = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_dim,
                                     num_layers=num_layers, batch_first=True)

        # 2. Memory 记忆模块（全局记忆库）
        self.mem_num = mem_num
        self.memory = self.construct_memory()

        # 3. 计算拼接后的维度：LSTM输出 + Memory输出
        # 移除门控后，这就是最终传递给输出头的特征维度
        final_dim = lstm_hidden_dim + mem_dim

        # 4. Transformer Encoder（学习站点/特征间关系）
        # 注意：这里 d_model 对应的是序列长度维度 (num_sites)，因为输入是 [combined_dim, B, N]
        encoder_layers = TransformerEncoderLayer(d_model=num_sites, nhead=nhead,
                                                 dim_feedforward=transformer_hidden_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                      num_layers=num_transformer_layers)

        # 5. 每个站点一个输出头
        # 输入维度直接使用 final_dim (LSTM + Memory 拼接后的维度)
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(final_dim, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            ) for _ in range(num_sites)
        ])

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() >= 2:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        # 移除了门控相关的特殊初始化

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
        # 计算注意力分数: [B, N, mem_num]
        att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)
        # 加权求和: [B, N, mem_dim]
        value = torch.matmul(att_score, self.memory['Memory'])
        return value

    def forward(self, x):
        B, N, T, _ = x.shape  # Batch, Num Sites, Time Steps, Features

        # Step 1: LSTM 处理时序特征
        x_all = x.view(B * N, T, -1)
        out_ori, _ = self.lstm_original(x_all)  # [B*N, T, lstm_hidden_dim]
        h_ori = out_ori[:, -1]  # 取最后时刻 [B*N, lstm_hidden_dim]

        # Step 2: 恢复批次和站点维度 [B, N, lstm_hidden_dim]
        h_t = h_ori.view(B, N, -1)

        # Step 3: 查询全局记忆 [B, N, mem_dim]
        h_att = self.query_memory(h_t)

        # Step 4: 拼接 LSTM 和 Memory 特征 [B, N, combined_dim]
        combined = torch.cat([h_t, h_att], dim=-1)

        # Step 5: Transformer
        transformer_input = combined.permute(2, 0, 1)

        # 通过 Transformer
        transformer_out = self.transformer_encoder(transformer_input)

        # 恢复维度: [B, N, combined_dim]
        transformer_out = transformer_out.permute(1, 2, 0)

        final_features = transformer_out  # [B, N, final_dim]

        # Step 6: 每个站点使用自己的输出头进行预测
        outputs = []
        for i in range(N):
            # 直接使用该站点的特征向量
            out_i = self.output_heads[i](final_features[:, i, :])
            outputs.append(out_i.unsqueeze(1))  # [B, 1, 1]

        final_output = torch.cat(outputs, dim=1)  # [B, N, 1]

        return final_output

