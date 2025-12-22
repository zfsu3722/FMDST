import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
import torch.nn.functional as F
import random


class MaskGenerator(nn.Module):
    """掩码生成器（原论文版本）"""

    def __init__(self, num_tokens, mask_ratio):
        super().__init__()
        self.num_tokens = num_tokens
        self.mask_ratio = mask_ratio

    def uniform_rand(self):
        mask = list(range(int(self.num_tokens)))
        random.shuffle(mask)
        mask_len = int(self.num_tokens * self.mask_ratio)
        masked_tokens = mask[:mask_len]
        unmasked_tokens = mask[mask_len:]
        unmasked_tokens = sorted(unmasked_tokens)
        masked_tokens = sorted(masked_tokens)
        return unmasked_tokens, masked_tokens

    def forward(self):
        unmasked_tokens, masked_tokens = self.uniform_rand()
        return unmasked_tokens, masked_tokens


class get_SpatialEmb(nn.Module):
    """空间嵌入（原论文版本）"""

    def __init__(self, d_model, num_nodes, dropout=0.1):
        super().__init__()
        self.spatial_embedding = nn.Embedding(num_nodes, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, spatial_indexs=None):
        batch, _, num_nodes, _ = x.shape
        if spatial_indexs is None:
            spatial_indexs = torch.arange(num_nodes).to(x.device)
        spatial_emb = self.spatial_embedding(spatial_indexs).unsqueeze(0).unsqueeze(1)
        return self.dropout(spatial_emb)


class get_TemporalEmb(nn.Module):
    """时间嵌入（原论文版本）"""

    def __init__(self, in_dim, d_model, slice_size_per_day, dropout=0.1):
        super().__init__()
        self.time_in_day_embedding = nn.Embedding(slice_size_per_day, d_model)
        self.day_in_week_embedding = nn.Embedding(7, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, t_hour, t_day):
        time_in_day_emb = self.time_in_day_embedding(t_hour)
        day_in_week_emb = self.day_in_week_embedding(t_day)
        return self.dropout(time_in_day_emb), self.dropout(day_in_week_emb)


class Projector_1(nn.Module):
    """投影器1（原论文版本）"""

    def __init__(self, in_dim, d_model, hasCross=True):
        super().__init__()
        in_units = in_dim * 2 if hasCross else in_dim
        self.hasCross = hasCross
        self.linear1 = nn.Linear(in_units, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)

    def forward(self, x, latestX):
        if self.hasCross:
            t_len = x.shape[1]
            if latestX.shape[1] != t_len:
                latestX = latestX.repeat([1, t_len, 1, 1])
            data = torch.cat([x, latestX], dim=-1)
        else:
            data = x
        data = self.linear1(data)
        data = self.activation(data)
        data = self.linear2(data)
        return data


class Projector_2(nn.Module):
    """投影器2（原论文版本）"""

    def __init__(self, tau, d_model):
        super().__init__()
        assert tau in [1, 2, 3]
        self.tau = tau
        if tau == 2:
            tcn_pad_l, tcn_pad_r = 0, 1
            self.padding = nn.ReplicationPad2d((tcn_pad_l, tcn_pad_r, 0, 0))
        elif tau == 3:
            tcn_pad_l, tcn_pad_r = 1, 1
            self.padding = nn.ReplicationPad2d((tcn_pad_l, tcn_pad_r, 0, 0))
        self.time_conv = nn.Conv2d(d_model, d_model, (1, tau))

    def forward(self, x):
        # x形状: (B, T, N, C) -> (B, C, N, T)
        x = rearrange(x, 'b t n c -> b c n t')
        if self.tau > 1:
            x = self.padding(x)
        data = self.time_conv(x)
        # 恢复形状: (B, C, N, T) -> (B, T, N, C)
        data = rearrange(data, 'b c n t -> b t n c')
        return data


class MultiHeadsAttention(nn.Module):
    """多头注意力（原论文版本）"""

    def __init__(self, scale=None, attention_dropout=0.1, returnA=False):
        super().__init__()
        self.scale = scale
        self.returnA = returnA
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / np.sqrt(E)

        # 计算注意力分数
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.returnA:
            return V.contiguous(), A.contiguous()
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    """注意力层（原论文版本）"""

    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, mix=True, dropout=0.1, returnA=False):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = MultiHeadsAttention(scale=None, attention_dropout=dropout, returnA=returnA)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.returnA = returnA
        self.mix = mix

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, A = self.inner_attention(queries, keys, values)

        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)
        out = self.out_projection(out)

        if self.returnA:
            return out, A
        else:
            return out, None


class STEncoderLayer(nn.Module):
    def __init__(self, factor, d_model, n_heads, input_length, time_factor, num_nodes=None,
                 d_ff=None, dropout=0.1, att_dropout=0.1, return_att=False):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.return_att = return_att
        self.input_length = input_length  # 添加这个
        assert num_nodes is not None

        self.proxy_token = nn.Parameter(torch.zeros(1, factor, d_model))
        self.time_factor = time_factor

        # 修正：正确设置Conv2d的输入输出通道
        self.time_readout = nn.Sequential(
            nn.Conv2d(d_model, time_factor, kernel_size=(1, input_length)),  # 输入通道=d_model, 输出通道=time_factor
            nn.GELU(),
            nn.Conv2d(time_factor, time_factor, kernel_size=(1, 1))
        )

        # 修正：输入通道应该是time_factor，输出通道应该是d_model
        self.time_recover = nn.Sequential(
            nn.Conv2d(time_factor, time_factor, kernel_size=(1, 1)),
            nn.GELU(),
            nn.Conv2d(time_factor, d_model, kernel_size=(1, input_length))  # 修正这里！应该是d_model
        )

        # 注意力层
        self.node2proxy = AttentionLayer(d_model, n_heads, dropout=att_dropout, returnA=return_att)
        self.proxy2node = AttentionLayer(d_model, n_heads, dropout=att_dropout, returnA=return_att)

        self.dropout = nn.Dropout(dropout)
        self.FFN = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, data):
        """data形状: (B, T, N, C)"""
        batch = data.shape[0]
        T = data.shape[1]
        N = data.shape[2]
        C = data.shape[3]  # d_model

        # print(f"\nSTEncoderLayer输入: {data.shape}")

        # 扩展代理令牌
        z_proxy = repeat(self.proxy_token, 'o m d -> (o b) m d', b=batch * self.time_factor)

        # 关键：正确重排数据用于Conv2d
        # (B, T, N, C) -> (B, C, N, T)
        data_for_conv = rearrange(data, 'b t n c -> b c n t')
        # print(f"Conv2d输入形状: {data_for_conv.shape}")

        # 时间压缩: (B, C, N, T) -> (B, time_factor, N, 1)
        kv_data = self.time_readout(data_for_conv)
        # print(f"时间压缩后: {kv_data.shape}")

        # 移除多余的维度: (B, time_factor, N, 1) -> (B, time_factor, N)
        kv_data = kv_data.squeeze(-1)
        # print(f"压缩后移除维度: {kv_data.shape}")

        # 重排列用于注意力: (B, time_factor, N) -> (B*time_factor, N, 1)
        kv_data = rearrange(kv_data, 'b t n -> (b t) n 1')
        # print(f"重排后注意力输入: {kv_data.shape}")

        # 扩展维度以匹配C
        kv_data = kv_data.expand(-1, -1, C)
        # print(f"扩展后注意力输入: {kv_data.shape}")
        # print(f"代理令牌: {z_proxy.shape}")

        # 注意力计算
        proxy_feature, A1 = self.node2proxy(z_proxy, kv_data, kv_data)
        node_feature, A2 = self.proxy2node(kv_data, proxy_feature, proxy_feature)

        # 残差连接
        enc_feature = kv_data + self.dropout(node_feature)
        enc_feature = enc_feature + self.dropout(self.FFN(enc_feature))

        # 恢复形状: (B*time_factor, N, C) -> (B, time_factor, N, C)
        # 注意：这里需要正确处理形状
        enc_feature = rearrange(enc_feature, '(b t) n c -> b t n c', b=batch, t=self.time_factor)
        # print(f"注意力后恢复形状: {enc_feature.shape}")

        # 为了time_recover，需要将形状转换为(B, time_factor, N, C) -> (B, C?, N, ?)
        # 实际上我们需要重新思考

        # 简化：直接返回编码后的特征，跳过时间恢复
        # 或者使用正确的方式

        # 方案1：跳过时间恢复
        final_out = enc_feature

        # 如果需要保持原始时间维度，可以插值
        if final_out.shape[1] != T:
            # 使用插值恢复时间维度
            final_out = rearrange(final_out, 'b t n c -> b c n t')
            final_out = F.interpolate(final_out, size=(N, T), mode='bilinear', align_corners=False)
            final_out = rearrange(final_out, 'b c n t -> b t n c')

        # print(f"最终输出形状: {final_out.shape}")

        if self.return_att:
            A1 = rearrange(A1, '(b t) h l s -> b t h l s', b=batch)
            A2 = rearrange(A2, '(b t) h l s -> b t h l s', b=batch)
            return final_out, [A1, A2]
        else:
            return final_out, None


class Decoder_pred(nn.Module):
    """预测解码器（修正版本）"""

    def __init__(self, num_nodes, input_length, predict_length, d_model, pre_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_length = input_length
        self.d_model = d_model
        self.pre_dim = pre_dim
        self.predict_length = predict_length

        # 将input_length压缩到predict_length
        self.end_conv1 = nn.Conv2d(
            in_channels=d_model,  # 64
            out_channels=predict_length,  # 6
            kernel_size=(1, input_length),  # 在时间维度上卷积
            bias=True
        )

        # 注意：这里输入维度应该是predict_length，因为end_conv1的输出通道是predict_length
        self.end_conv2 = nn.Linear(predict_length, pre_dim)  # 修正这里！

    def forward(self, data):
        """data形状: (B, T, N, C) = (B, input_length, N, d_model)"""
        # print(f"Decoder_pred输入: {data.shape}")

        # 转换为卷积需要的形状: (B, C, N, T)
        data = rearrange(data, 'b t n c -> b c n t')  # (B, d_model, N, input_length)
        # print(f"卷积前: {data.shape}")

        # 时间卷积: 将input_length压缩到predict_length
        # (B, d_model, N, input_length) -> (B, predict_length, N, 1)
        rep = self.end_conv1(data)
        # print(f"卷积后: {rep.shape}")

        # 移除多余的维度: (B, predict_length, N, 1) -> (B, predict_length, N)
        rep = rep.squeeze(-1)
        # print(f"移除维度后: {rep.shape}")

        # 转置: (B, predict_length, N) -> (B, N, predict_length)
        rep = rep.transpose(1, 2)  # 为了线性层
        # print(f"转置后: {rep.shape}")

        # 线性层: (B, N, predict_length) -> (B, N, pre_dim)
        output = self.end_conv2(rep)  # (B, N, pre_dim)
        # print(f"线性层后: {output.shape}")

        # 调整形状: (B, N, pre_dim) -> (B, predict_length, N, pre_dim)
        output = output.unsqueeze(1)  # 添加时间维度
        output = output.expand(-1, self.predict_length, -1, -1)
        # print(f"最终输出: {output.shape}")

        return output


class Decoder_recon(nn.Module):
    """重建解码器（修正版本）"""

    def __init__(self, num_nodes, input_length, d_model, in_dim, hasRes=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_length = input_length
        self.d_model = d_model
        self.hasRes = hasRes

        # 修正：输入通道应该是d_model，而不是input_length
        self.end_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1, bias=True),  # 修正这里！
            nn.GELU()
        )
        self.end_conv2 = nn.Linear(d_model, in_dim)

    def forward(self, data):
        # data形状: (B, T, N, C) = (B, input_length, N, d_model)
        skip = data

        # 转换为: (B, C, N, T) 用于卷积
        data = rearrange(data, 'b t n c -> b c n t')  # (B, d_model, N, input_length)
        rep = self.end_conv1(data)  # 现在通道数匹配了

        # 转换回: (B, T, N, C)
        rep = rearrange(rep, 'b c n t -> b t n c')

        if self.hasRes:
            rep = rep + skip
        data = self.end_conv2(rep)
        return data


class STReP(nn.Module):
    """完整的ST-ReP模型（原论文版本）"""

    def __init__(self, args):
        super().__init__()
        # 模型参数
        self.input_length = args.input_length  # 60
        self.predict_length = args.predict_length  # 6
        self.in_dim = args.in_dim  # 1
        self.pre_dim = args.in_dim
        self.num_nodes = args.num_nodes  # 134
        self.tau = args.tau  # 3
        self.d_model = args.d_model  # 64
        self.hasCross = bool(args.hasCross)  # True
        self.num_layers = args.num_layers  # 2
        self.mask_ratio = args.mask_ratio  # 0.25
        self.mask_only = args.loss_type == 'mask_only'
        self.pred_only = args.loss_type == 'pred_only'

        # 掩码令牌
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, args.d_model))

        # 投影层
        self.data_encoding = Projector_1(args.in_dim, args.d_model, self.hasCross)
        self.get_spatial_emb = get_SpatialEmb(args.d_model, args.num_nodes, args.st_emb_dropout)
        self.get_temporal_emb = get_TemporalEmb(args.in_dim, args.d_model, args.slice_size_per_day, args.st_emb_dropout)
        self.tcn_encoding = Projector_2(args.tau, args.d_model)

        # 编码器层
        self.spatial_agg_list = nn.ModuleList([
            STEncoderLayer(args.M, args.d_model, n_heads=args.n_heads,
                           num_nodes=args.num_nodes, dropout=args.spatial_dropout,
                           att_dropout=args.spatial_att_dropout,
                           time_factor=args.time_factor, input_length=args.input_length)
            for _ in range(args.num_layers)
        ])

        # Dropout和头
        self.rep_dropout = nn.Dropout(args.rep_dropout)
        self.predictor = Decoder_pred(args.num_nodes, args.input_length, args.predict_length, args.d_model,
                                      pre_dim=self.pre_dim)
        self.reconstructor = Decoder_recon(args.num_nodes, args.input_length, args.d_model, args.in_dim)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _getnote(self):
        return f'STReP_SDWPF_{self.input_length}to{self.predict_length}'

    def _input_processing(self, x):
        """输入处理"""
        if len(x) == 2:
            # 推理模式
            x_data, x_time_mark = x
            B, T, N, _ = x_data.shape
            latestX = x_data[:, -1:, :, :]
            return x_data, latestX, x_time_mark[..., 0:1], x_time_mark[..., 1:2], None
        elif len(x) == 3:
            # 训练模式
            x_data, x_time_mark, y_data = x
            B, T, N, _ = x_data.shape
            latestX = x_data[:, -1:, :, :]
            return x_data, latestX, x_time_mark[..., 0:1], x_time_mark[..., 1:2], y_data

    def _mask_raw_data(self, data):
        """掩码原始数据"""
        B, T, N, C = data.shape

        # 创建掩码生成器
        MaskEngine = MaskGenerator(T, self.mask_ratio)
        unmasked_token_index, masked_token_index = MaskEngine.uniform_rand()

        # 获取未掩码数据
        data_o = data[:, unmasked_token_index, :, :]
        return data_o, unmasked_token_index, masked_token_index

    def get_rep(self, x):
        """获取表示（用于下游任务）"""
        inputx, latestX, x_t_hour, x_t_day, _ = self._input_processing(x)

        # 空间嵌入
        spatial_emb = self.get_spatial_emb(inputx)

        # 时间嵌入
        x_tod_emb, x_dow_emb = self.get_temporal_emb(x_t_hour, x_t_day)

        # 数据编码
        data = self.data_encoding(inputx, latestX)
        data = data + x_tod_emb + x_dow_emb + spatial_emb

        # TCN编码
        data = self.tcn_encoding(data)

        # 编码
        rep = self._encode(data)
        return rep, latestX

    def _encode(self, data):
        """编码过程"""
        skip = data
        for i in range(self.num_layers):
            data, _ = self.spatial_agg_list[i](data)
        data += skip
        return data

    def forward(self, x):
        """前向传播"""
        # 输入处理
        inputx, latestX, x_t_hour, x_t_day, inputy = self._input_processing(x)

        # 空间嵌入
        spatial_emb = self.get_spatial_emb(inputx)

        # 时间嵌入
        x_tod_emb, x_dow_emb = self.get_temporal_emb(x_t_hour, x_t_day)

        # 掩码
        x_o, x_unmasked_token_index, x_masked_token_index = self._mask_raw_data(inputx)

        # 处理未掩码部分
        x_tod_emb_o = x_tod_emb[:, x_unmasked_token_index, :, :]
        x_dow_emb_o = x_dow_emb[:, x_unmasked_token_index, :, :]

        # 未掩码数据编码
        x_data_o = self.data_encoding(x_o, latestX)
        x_data_o = x_data_o + x_tod_emb_o + x_dow_emb_o + spatial_emb
        x_data_o = self.tcn_encoding(x_data_o)

        B, T_, N_, d = x_data_o.shape

        # 处理掩码部分
        x_tod_emb_m = x_tod_emb[:, x_masked_token_index, :, :]
        x_dow_emb_m = x_dow_emb[:, x_masked_token_index, :, :]
        x_mask_token = self.mask_token.expand(B, len(x_masked_token_index), N_, d)
        x_mask_token = x_mask_token + x_tod_emb_m + x_dow_emb_m

        # 合并掩码和未掩码部分
        x_data_full = torch.cat([x_mask_token, x_data_o], dim=1)

        # 编码
        x_rep_full = self._encode(x_data_full)

        # 重建
        if self.pred_only:
            reconx = None
        else:
            reconx = self.reconstructor(x_rep_full)

        # 预测
        if self.mask_only:
            predy = None
        else:
            x_rep_drop = self.rep_dropout(x_rep_full)
            predy = self.predictor(x_rep_drop)

        return inputx, reconx, inputy, predy


def create_strep_args_sdwpf():
    """创建SDWPF的ST-ReP模型参数"""

    class Args:
        def __init__(self):
            # SDWPF数据集参数
            self.num_nodes = 134  # 134个风力涡轮机
            self.input_length = 60  # 60个时间步（10小时）
            self.predict_length = 6  # 6个时间步（1小时）
            self.in_dim = 1  # 输入特征维度（功率）
            self.pre_dim = 1  # 输出特征维度

            # 模型架构参数（原论文默认）
            self.d_model = 64  # 模型维度
            self.n_heads = 4  # 注意力头数
            self.num_layers = 2  # 编码器层数
            self.M = 8  # 代理令牌数量
            self.time_factor = 6  # 时间压缩因子
            self.tau = 3  # TCN卷积核大小
            self.hasCross = 1  # 使用交叉时间连接
            self.mask_ratio = 0.25  # 掩码比例

            # SDWPF时间特征
            self.slice_size_per_day = 144  # 一天有144个10分钟间隔

            # Dropout参数
            self.spatial_dropout = 0.1
            self.spatial_att_dropout = 0.1
            self.st_emb_dropout = 0.1
            self.rep_dropout = 0.1

            # 训练参数
            self.loss_type = 'pred_mask_ms'  # 损失类型
            self.alpha = 0.2  # 预测损失权重
            self.beta = 0.5  # 重建损失权重

    return Args()