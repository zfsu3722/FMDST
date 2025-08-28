import torch
import torch.nn as nn

# 假设你已经定义了这两个模型类
from model.TCN_Transformer import TCN_Transformer
from model.LSTM_Transformer import LSTM_Transformer

class FMDST(nn.Module):
    def __init__(self, input_dim=7, output_dim=1, site=134, his=60):
        super(FMDST, self).__init__()

        # 定义第一个子模型：TCN + 双 Transformer
        self.tcn_model = TCN_Transformer(
            input_dim=input_dim,
            output_dim=output_dim
        )

        # 定义第二个子模型：LSTM + 双 Transformer
        self.lstm_model = LSTM_Transformer(
            input_dim=input_dim,
            output_dim=output_dim
        )

        # 加载预训练权重（路径根据你的实际情况调整）
        tcn_weight_path = "model_save/Power_Pred_TCN_Transformer_his60_Pred1-6_134site_7feature_best_model.pth"
        lstm_weight_path = "model_save/Power_Pred_LSTM_Transformer_his60_Pred1-6_134site_7feature_best_model.pth"

        # 加载权重（注意 map_location 的使用，以防 GPU/CPU 不一致）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tcn_model.load_state_dict(torch.load(tcn_weight_path, map_location=device))
        self.lstm_model.load_state_dict(torch.load(lstm_weight_path, map_location=device))

        # 冻结参数（可选）
        for param in self.tcn_model.parameters():
            param.requires_grad = False
        for param in self.lstm_model.parameters():
            param.requires_grad = False

        # 可学习融合权重矩阵：每个站点一个权重（例如 sigmoid 归一化）
        # self.weight_matrix = nn.Parameter(torch.rand(site, output_dim))
        # 每个站点一个可学习融合权重，初始化为均匀分布或随机值
        self.site_weights = nn.Parameter(torch.rand(site))  # shape: (134,)

    def forward(self, x):
        """
        :param x: shape (batch_size, num_stations, seq_len, input_dim)
        :return: shape (batch_size, num_stations, output_dim)
        """

        # 获取两个模型的输出
        tcn_out = self.tcn_model(x)   # (B, N, F) (batch_size, num_stations, forecast_)
        lstm_out = self.lstm_model(x) # (B, N, F)

        # 使用 sigmoid 将原始权重映射到 [0, 1]
        weights = torch.sigmoid(self.site_weights)  # shape: (134, )

        # 扩展维度以便广播相乘
        weights = weights[None, :, None]  # shape: (1, 134, 1) -> 广播到 (B, 134, F)

        # 融合输出
        fused_output = weights * tcn_out + (1 - weights) * lstm_out  # shape: (B, 134, F)

        return fused_output