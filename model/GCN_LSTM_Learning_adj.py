import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse


class GCN_LSTM_Learning_adj_Model(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32, output_dim=1, dropout=0.1, num_nodes=10):
        super().__init__()
        # 两层 GCN
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        # Dropout 层
        self.dropout = nn.Dropout(dropout)

        # LSTM 时间建模
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # 输出层
        self.fc = nn.Linear(hidden_dim, output_dim)

        # 可学习的邻接矩阵参数
        self.N1 = nn.Parameter(torch.randn(num_nodes, 24))  # [N, 24]
        self.N2 = nn.Parameter(torch.randn(24, num_nodes))  # [24, N]

        # 缓存边索引和邻接矩阵
        self.register_buffer('cached_edge_index', None)
        self.register_buffer('cached_adj_matrix', None)

    def forward(self, x):
        """
        Args:
            x: [batch_size, num_nodes, seq_len, input_dim]
        Returns:
            output: [batch_size, num_nodes, output_dim]
        """
        batch_size, num_nodes, seq_len, feat_dim = x.shape

        # 1. 动态生成邻接矩阵 A = sigmoid(N1 @ N2)
        if self.cached_adj_matrix is None or self.cached_adj_matrix.shape[0] != num_nodes:
            adj_matrix = torch.sigmoid(self.N1 @ self.N2)  # [N, N]
            self.cached_adj_matrix = adj_matrix

        # 2. 构造 edge_index（仅第一次）
        if self.cached_edge_index is None:
            edge_index, _ = dense_to_sparse(self.cached_adj_matrix)
            self.cached_edge_index = edge_index.to(x.device)

        # 3. 批量展平输入
        x_flat = x.permute(0, 2, 1, 3).reshape(-1, num_nodes, feat_dim)  # [B*T, N, F]
        x_gcn = x_flat.reshape(-1, feat_dim)  # [B*T*N, F]

        # 4. 扩展边索引以匹配批量数据
        num_graphs = x_flat.size(0)
        batch_edge_index = self._expand_edge_index(
            self.cached_edge_index, num_graphs, num_nodes
        )

        # 5. 空间特征提取 - 两层 GCN + ReLU + Dropout
        spatial_features = F.relu(self.gcn1(x_gcn, batch_edge_index))
        spatial_features = self.dropout(spatial_features)
        spatial_features = F.relu(self.gcn2(spatial_features, batch_edge_index))

        # 6. 时间特征提取
        spatial_features = spatial_features.view(batch_size, seq_len, num_nodes, -1)  # [B, T, N, H]
        lstm_input = spatial_features.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, -1)  # [B*N, T, H]
        lstm_out, _ = self.lstm(lstm_input)  # [B*N, T, H]

        # 7. 输出层
        out = self.fc(lstm_out[:, -1])  # [B*N, O]
        return out.view(batch_size, num_nodes, -1)  # [B, N, O]

    def _expand_edge_index(self, edge_index, num_graphs, num_nodes):
        """
        将原始 edge_index 扩展为支持批量处理的版本
        Args:
            edge_index: [2, E]
            num_graphs: int
            num_nodes: int
        Returns:
            expanded_edge_index: [2, E * num_graphs]
        """
        device = edge_index.device
        offset = torch.arange(0, num_graphs * num_nodes, num_nodes, device=device)
        offset = offset.view(-1, 1)  # [num_graphs, 1]
        edge_index = edge_index.unsqueeze(0)  # [1, 2, E]
        expanded = edge_index + offset.unsqueeze(-1)  # [num_graphs, 2, E]
        return expanded.view(2, -1)  # [2, E * num_graphs]