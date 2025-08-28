import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer



class LSTM_Transformer_No_multi_head(nn.Module):
    def __init__(self, input_dim=5, lstm_hidden_dim=32, transformer_hidden_dim=64,
                 output_dim=1, num_layers=2, num_sites=134, his=60, nhead=2, num_transformer_layers=1):
        super(LSTM_Transformer_No_multi_head, self).__init__()

        self.num_sites = num_sites
        self.seq_len = his

        self.lstm_original = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_dim,
                                     num_layers=num_layers, batch_first=True)

        # Transformer Encoder Layer for learning relationships between sites
        encoder_layers_sites = TransformerEncoderLayer(d_model=lstm_hidden_dim, nhead=nhead,
                                                        dim_feedforward=transformer_hidden_dim)
        self.transformer_encoder_sites = TransformerEncoder(encoder_layers_sites, num_layers=num_transformer_layers)

        # Transformer Encoder Layer for learning relationships within features
        encoder_layers_features = TransformerEncoderLayer(d_model=num_sites, nhead=nhead,
                                                           dim_feedforward=transformer_hidden_dim)
        self.transformer_encoder_features = TransformerEncoder(encoder_layers_features, num_layers=num_transformer_layers)

        # ✅ 共享输出头（不再是每个站点一个）
        self.shared_output_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

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


        out_ori, _ = self.lstm_original(x_all)  # [B*N, T, H]
        h_ori = out_ori[:, -1]  # [B*N, H]




        combined =  h_ori  # [B*N, H]

        # Step 4: 恢复成 (B, N, H)
        lstm_last_outputs = combined.view(B, N, -1)  # [B, N, H]

        # Step 5: Transformer 学习站点间关系
        transformer_input_sites = lstm_last_outputs.transpose(0, 1)  # [N, B, H]
        transformer_out_sites = self.transformer_encoder_sites(transformer_input_sites)  # [N, B, H]
        transformer_out_sites = transformer_out_sites.transpose(0, 1)  # [B, N, H]

        # Step 6: Transformer 学习特征间关系（转置后输入）
        transformer_input_features = lstm_last_outputs.permute(2, 0, 1)  # [H, B, N]
        transformer_out_features = self.transformer_encoder_features(transformer_input_features)  # [H, B, N]
        transformer_out_features = transformer_out_features.permute(1, 2, 0)  # [B, N, H]

        # Step 7: 合并两个Transformer输出
        combined_representation = torch.cat([transformer_out_sites, transformer_out_features], dim=-1)  # [B, N, 2*H]

        B, N, _ = combined_representation.shape
        combined_representation = combined_representation.view(B * N, -1)  # -> (B*N, 2*C)
        outputs = self.shared_output_head(combined_representation)  # -> (B*N, output_dim)
        outputs = outputs.view(B, N, -1)  # -> (B, N, output_dim)

        return outputs