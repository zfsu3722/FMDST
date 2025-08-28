import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class LSTM_No_feature_Transformer(nn.Module):
    def __init__(self, input_dim=5, lstm_hidden_dim=32, transformer_hidden_dim=64,
                 output_dim=1, num_layers=2, num_sites=134, his=60, nhead=2, num_transformer_layers=1):
        super(LSTM_No_feature_Transformer, self).__init__()

        self.num_sites = num_sites
        self.seq_len = his

        self.lstm_original = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_dim,
                                     num_layers=num_layers, batch_first=True)

        # Transformer Encoder Layer for learning relationships between sites
        encoder_layers_sites = TransformerEncoderLayer(d_model=lstm_hidden_dim, nhead=nhead,
                                                        dim_feedforward=transformer_hidden_dim)
        self.transformer_encoder_sites = TransformerEncoder(encoder_layers_sites, num_layers=num_transformer_layers)

        #

        # 每个站点一个输出头
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_hidden_dim , 64),
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

    def forward(self, x):
        B, N, T, _ = x.shape  # Batch, Num Sites, Time Steps, Features

        # Step 1: 将所有站点的时间序列合并到 batch 维度 → [B*N, T, F]
        x_all = x.view(B * N, T, -1)  # 合并站点和batch维度


        # 原始序列
        out_ori, _ = self.lstm_original(x_all)  # [B*N, T, H]
        h_ori = out_ori[:, -1]  # [B*N, H]




        combined =  h_ori  # [B*N, H]

        # Step 4: 恢复成 (B, N, H)
        lstm_last_outputs = combined.view(B, N, -1)  # [B, N, H]

        # Step 5: Transformer 学习站点间关系
        transformer_input_sites = lstm_last_outputs.transpose(0, 1)  # [N, B, H]
        transformer_out_sites = self.transformer_encoder_sites(transformer_input_sites)  # [N, B, H]
        transformer_out_sites = transformer_out_sites.transpose(0, 1)  # [B, N, H]


        # Step 7: 合并两个Transformer输出
        combined_representation = transformer_out_sites

        # Step 8: 每个站点使用自己的输出头进行预测
        outputs = []
        for i in range(N):
            out_i = self.output_heads[i](combined_representation[:, i, :])  # [B, 1]
            outputs.append(out_i.unsqueeze(1))  # [B, 1, 1]

        final_output = torch.cat(outputs, dim=1)  # [B, N, 1]

        return final_output