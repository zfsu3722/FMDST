import torch
import torch.nn as nn


class STAEformerForPowerPred(nn.Module):
    """适配电力预测任务的STAEformer，输入格式 [batch, sites, timesteps, features]，输出 [batch, sites, timesteps]"""

    def __init__(
            self,
            num_nodes=134,  # 站点数
            in_steps=60,  # 历史步长
            out_steps=6,  # 预测步长
            steps_per_day=288,
            input_dim=7,  # 输入特征数：风速、风向、内温、外温、机舱方向、叶片俯仰角、历史功率
            output_dim=1,  # 输出特征数（每个时间步预测一个值，这里是功率）
            input_embedding_dim=24,
            tod_embedding_dim=0,  # 设置为0，因为没有tod特征
            dow_embedding_dim=0,  # 设置为0，因为没有dow特征
            spatial_embedding_dim=0,
            adaptive_embedding_dim=64,
            feed_forward_dim=128,
            num_heads=2,
            num_layers=2,
            dropout=0.1,
            use_mixed_proj=True,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
                input_embedding_dim
                + tod_embedding_dim
                + dow_embedding_dim
                + spatial_embedding_dim
                + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        # 输入投影：对所有特征进行投影
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)

        # 空间嵌入（可选）
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)

        # 自适应嵌入
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.Parameter(
                torch.empty(in_steps, num_nodes, adaptive_embedding_dim)
            )
            nn.init.xavier_uniform_(self.adaptive_embedding)

        # 输出投影
        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, output_dim)

        # 注意力层
        self.attn_layers_t = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.attn_layers_s = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        输入 x: [batch_size, num_sites, in_steps, input_dim]
        其中 input_dim=7，包含：
        - 第0维：风速
        - 第1维：风向
        - 第2维：内温
        - 第3维：外温
        - 第4维：机舱方向
        - 第5维：叶片俯仰角
        - 第6维：历史功率（需要预测的是未来的功率）

        输出: [batch_size, num_sites, out_steps]
        """
        # 原始输入形状: [batch_size, num_sites, in_steps, input_dim]
        batch_size, num_sites, seq_len, input_dim = x.shape

        # 转换为STAEformer期望的格式: [batch_size, in_steps, num_sites, input_dim]
        x = x.transpose(1, 2)  # [batch_size, in_steps, num_sites, input_dim]

        # 输入投影（对所有特征进行投影）
        x_embed = self.input_proj(x)  # [batch_size, in_steps, num_sites, input_embedding_dim]

        # 特征拼接
        features = [x_embed]

        # 空间嵌入
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )  # [batch_size, in_steps, num_sites, spatial_embedding_dim]
            features.append(spatial_emb)

        # 自适应嵌入
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )  # [batch_size, in_steps, num_sites, adaptive_embedding_dim]
            features.append(adp_emb)

        # 拼接所有特征
        x = torch.cat(features, dim=-1)  # [batch_size, in_steps, num_sites, model_dim]

        # 时间注意力（在时间维度上）
        for attn in self.attn_layers_t:
            x = attn(x, dim=1)  # 在时间维度上注意力

        # 空间注意力（在站点维度上）
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)  # 在站点维度上注意力

        # 输出投影
        if self.use_mixed_proj:
            # 将时间维度展平
            out = x.transpose(1, 2)  # [batch_size, num_sites, in_steps, model_dim]
            out = out.reshape(batch_size, num_sites, -1)  # [batch_size, num_sites, in_steps * model_dim]
            out = self.output_proj(out)  # [batch_size, num_sites, out_steps * output_dim]
            # 调整形状: output_dim=1，所以直接得到 [batch_size, num_sites, out_steps]
            out = out.view(batch_size, num_sites, self.out_steps)
        else:
            # 先进行时间维度投影
            out = x.transpose(1, 3)  # [batch_size, model_dim, num_sites, in_steps]
            out = self.temporal_proj(out)  # [batch_size, model_dim, num_sites, out_steps]
            # 再进行特征维度投影
            out = out.transpose(1, 3)  # [batch_size, out_steps, num_sites, model_dim]
            out = self.output_proj(out)  # [batch_size, out_steps, num_sites, output_dim]
            out = out.transpose(1, 2)  # [batch_size, num_sites, out_steps, output_dim]
            out = out.squeeze(-1)  # [batch_size, num_sites, out_steps]

        return out


class SelfAttentionLayer(nn.Module):
    """注意力层"""

    def __init__(self, model_dim, feed_forward_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = AttentionLayer(model_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        residual = x
        out = self.attn(x, x, x)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class AttentionLayer(nn.Module):
    """注意力层"""

    def __init__(self, model_dim, num_heads=8):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        batch_size = query.shape[0]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # 检查维度是否能被num_heads整除
        if query.shape[-1] % self.num_heads != 0:
            raise ValueError(f"model_dim ({query.shape[-1]}) must be divisible by num_heads ({self.num_heads})")

        # 多头分割
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(-1, -2)
        attn_score = (query @ key) / self.head_dim ** 0.5
        attn_score = torch.softmax(attn_score, dim=-1)

        out = attn_score @ value
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        out = self.out_proj(out)

        return out


# 简化包装类
class STAEformerWrapper(nn.Module):
    """简化包装类，完全兼容你的训练接口"""

    def __init__(self, input_dim=7, output_dim=6, site=134, his=60):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.site = site
        self.his = his

        # 创建STAEformer
        self.model = STAEformerForPowerPred(
            num_nodes=site,
            in_steps=his,
            out_steps=output_dim,  # 预测步数
            input_dim=input_dim,  # 输入特征数
            output_dim=1,  # 每个时间步输出1个值（功率）
        )

    def forward(self, x):
        """
        x: shape (batch_size, num_stations, seq_len, input_dim)
        输出: shape (batch_size, num_stations, output_dim)
        """
        return self.model(x)


# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = STAEformerWrapper(input_dim=7, output_dim=6, site=134, his=60)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 假设输入数据
    batch_size = 32
    x = torch.randn(batch_size, 134, 60, 7)  # [batch, sites, timesteps, features]

    # 前向传播
    output = model(x)  # 输出 [32, 134, 6]
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")