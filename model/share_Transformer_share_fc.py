import torch
import torch.nn as nn
import math


class SharedTransformer(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1,
                 site=134, his=60, nhead=2, dropout=0.1):
        super(SharedTransformer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.site = site
        self.his = his
        self.nhead = nhead

        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, hidden_dim)

        # 位置编码
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout, his)

        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 全连接输出层
        self.fc = nn.Linear(hidden_dim, output_dim)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        """
        x: shape (batch_size, num_stations, seq_len, input_dim)
        """
        batch_size, num_stations, seq_len, input_dim = x.size()

        # reshape to (batch_size * num_stations, seq_len, input_dim)
        x = x.view(-1, seq_len, input_dim)

        # 输入嵌入
        x = self.input_embedding(x)  # (batch_size*num_stations, seq_len, hidden_dim)

        # 添加位置编码
        x = self.positional_encoding(x)

        # Transformer处理
        transformer_out = self.transformer_encoder(x)  # (batch_size*num_stations, seq_len, hidden_dim)

        # 取最后一个时间步的输出
        transformer_out = transformer_out[:, -1, :]  # (batch_size*num_stations, hidden_dim)

        # 全连接层
        out = self.fc(transformer_out)  # (batch_size*num_stations, output_dim)

        # reshape回 (batch_size, num_stations, output_dim)
        out = out.view(batch_size, num_stations, -1)

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] or [batch_size, seq_len, embedding_dim]
        """
        if x.dim() == 3 and x.size(1) == self.pe.size(0):  # batch_first=True
            x = x + self.pe.squeeze(1)
        else:
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)