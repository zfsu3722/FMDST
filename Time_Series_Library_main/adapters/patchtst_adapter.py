import torch
import torch.nn as nn
from types import SimpleNamespace
from Time_Series_Library_main.models.PatchTST import Model as PatchTSTModel


class PatchTSTAdapter(nn.Module):
    """
    输入:  x [B, N, T, F]
    输出:  y [B, N, pred_len]
    说明:
    - 使用 TSLib 的 PatchTST 主体
    - 将多站点维 N 展平到 batch
    - 只预测单目标（功率）
    """

    def __init__(
        self,
        input_dim=7,
        seq_len=60,
        pred_len=12,
        label_len=30,
        target_idx=-1,   # 保留接口，当前版本不直接用
        d_model=128,
        n_heads=8,
        e_layers=3,
        d_ff=256,
        dropout=0.1,
        fc_dropout=0.1,
        head_dropout=0.0,
        patch_len=16,
        stride=8,
        factor=3,
        activation='gelu',
        embed='timeF',
        freq='h',
        individual=False
    ):
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.target_idx = target_idx

        configs = SimpleNamespace(
            task_name='long_term_forecast',
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,

            enc_in=input_dim,
            dec_in=input_dim,
            c_out=1,
            target_idx=target_idx,

            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=1,
            d_ff=d_ff,
            factor=factor,
            dropout=dropout,
            fc_dropout=fc_dropout,
            head_dropout=head_dropout,
            activation=activation,
            embed=embed,
            freq=freq,
            output_attention=False,
            distil=True,
            patch_len=patch_len,
            stride=stride,
            individual=individual,
        )

        self.model = PatchTSTModel(configs)

    def forward(self, x):
        # x: [B, N, T, F]
        B, N, T, F = x.shape
        assert T == self.seq_len, f"seq_len mismatch: got {T}, expected {self.seq_len}"
        assert F == self.input_dim, f"input_dim mismatch: got {F}, expected {self.input_dim}"

        x_enc = x.reshape(B * N, T, F)

        x_mark_enc = torch.zeros(B * N, T, 4, device=x.device, dtype=x.dtype)

        dec_zeros = torch.zeros(B * N, self.pred_len, F, device=x.device, dtype=x.dtype)
        x_dec = torch.cat([x_enc[:, -self.label_len:, :], dec_zeros], dim=1)
        x_mark_dec = torch.zeros(B * N, self.label_len + self.pred_len, 4, device=x.device, dtype=x.dtype)

        y = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        if isinstance(y, tuple):
            y = y[0]

        # 期望输出 [B*N, pred_len, 1]
        if y.dim() == 3:
            y = y[..., -1]

        y = y.reshape(B, N, self.pred_len)
        return y