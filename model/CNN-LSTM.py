import torch
import torch.nn as nn


class SharedCNN_LSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1, site=134, his=60):
        super(SharedCNN_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.site = site
        self.his = his

        # CNN 部分：可以多层 Conv + ReLU + Pooling
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3, padding=1),  # (batch, 16, L)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 下采样一半长度

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),  # (batch, 32, L/2)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # (batch, 32, L/4)
        )

        # 计算经过 CNN 后的 seq_len
        cnn_output_len = his // 4  # 因为两次 MaxPool kernel_size=2

        # LSTM 层
        self.lstm = nn.LSTM(input_size=32,  # CNN 输出通道数
                            hidden_size=hidden_dim,
                            num_layers=2,
                            batch_first=True)

        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

        # 初始化权重（可选）
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        """
        x: shape (batch_size, num_stations, seq_len, input_dim)
        """
        batch_size, num_stations, seq_len, input_dim = x.size()

        # reshape to (batch_size * num_stations, seq_len, input_dim)
        x = x.view(-1, seq_len, input_dim)

        # Transpose to (batch_size * num_stations, input_dim, seq_len) for Conv1d
        x = x.transpose(1, 2)

        # CNN 处理
        cnn_out = self.cnn(x)


        cnn_out = cnn_out.transpose(1, 2)

        # LSTM 处理
        lstm_out, _ = self.lstm(cnn_out)

        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]

        # 全连接层
        out = self.fc(lstm_out)

        # reshape 回 (batch_size, num_stations, output_dim)
        out = out.view(batch_size, num_stations, -1)

        return out