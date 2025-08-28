import torch
import torch.nn as nn
import pywt
import numpy as np


class ShareLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1, site=134, his=60):
        super(ShareLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.site = site
        self.his = his

        # 共享的 LSTM 层
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)

        # 全连接层（如果希望每个站点有自己的头，也可以做成 ModuleList）
        self.fc = nn.Linear(hidden_dim, output_dim)

        # 初始化权重（可选）
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        """
        x: shape (batch_size, num_stations, seq_len, input_dim)
        """
        batch_size, num_stations, seq_len, input_dim = x.size()

        # reshape to (batch_size * num_stations, seq_len, input_dim)
        x = x.view(-1, seq_len, input_dim)

        # LSTM 处理
        lstm_out, _ = self.lstm(x)  # shape: (batch_size * num_stations, seq_len, hidden_dim)

        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]  # shape: (batch_size * num_stations, hidden_dim)

        # 接全连接层
        out = self.fc(lstm_out)  # shape: (batch_size * num_stations, output_dim)

        # reshape 回 (batch_size, num_stations, output_dim)
        out = out.view(batch_size, num_stations, -1)

        return out