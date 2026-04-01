import torch
import torch.nn as nn
from types import SimpleNamespace
from models.Reformer import Model as ReformerModel


class ReformerAdapter(nn.Module):
    """
    输入:  x [B, N, T, F]
    输出:  y [B, N, pred_len]
    """

    def __init__(
        self,
        input_dim=7,
        seq_len=60,
        pred_len=12,
        label_len=30,
        target_idx=0,
        d_model=128,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=256,
        factor=5,
        dropout=0.1,
        activation='gelu',
        embed='timeF',
        freq='h',
        bucket_size=4,
        n_hashes=4
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

            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            factor=factor,
            dropout=dropout,
            activation=activation,
            embed=embed,
            freq=freq,
            output_attention=False,
            bucket_size=bucket_size,
            n_hashes=n_hashes
        )

        self.model = ReformerModel(configs)

    def forward(self, x):
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

        if y.dim() == 3:
            y = y[..., 0]

        y = y.reshape(B, N, self.pred_len)
        return y