import torch
import torch.nn as nn
from ST_ReP.strep_model_full import STReP, create_strep_args_sdwpf


class STRePWrapper(nn.Module):
    def __init__(self, input_dim=7, output_dim=6, site=134, his=60, **kwargs):
        super(STRePWrapper, self).__init__()

        # 注意：我们只使用 input_dim 的最后一个特征（功率）
        # 创建 ST-ReP 参数（固定为单变量功率预测）
        args = create_strep_args_sdwpf()
        args.input_length = his  # 60
        args.predict_length = output_dim  # 6
        args.num_nodes = site  # 134
        args.in_dim = 1  # ⚠️ 只用功率，所以 in_dim=1

        self.model = STReP(args)

    def forward(self, x):
        """
        Input:
            x: [B, S, T, F]  -> e.g., [32, 134, 60, 7]
                where x[..., -1] is the power (Patv)
        Output:
            out: [B, S, pred_len] -> e.g., [32, 134, 6]
        """
        B, S, T, F = x.shape

        # ✅ 提取最后一个特征（功率），并保留维度
        power = x[..., -1:]  # [B, S, T, 1]

        # 转置为 ST-ReP 所需格式: [B, T, S, 1]
        x_strep = power.permute(0, 2, 1, 3)  # [B, T, S, 1]

        # 自动生成时间戳（假设连续）
        device = x.device
        t_of_d = (torch.arange(T, device=device) % 144).long()  # [T]
        d_of_w = ((torch.arange(T, device=device) // 144) % 7).long()  # [T]
        time_mark = torch.stack([t_of_d, d_of_w], dim=-1)  # [T, 2]
        time_mark = time_mark.unsqueeze(0).expand(B, -1, -1)  # [B, T, 2]

        # 构造 dummy y（满足 ST-ReP 接口要求）
        y_dummy = torch.zeros(B, 6, S, 1, device=device)

        # 调用原始 ST-ReP 模型
        _, _, _, pred = self.model((x_strep, time_mark, y_dummy))  # pred: [B, 6, S, 1]

        # 调整输出形状：[B, 6, S, 1] → [B, S, 6]
        pred = pred.squeeze(-1).permute(0, 2, 1)  # [B, S, 6]

        return pred





if __name__ == "__main__":
    print("🔍 简单测试 STRePWrapper 输入输出形状...")

    model = STRePWrapper(input_dim=7, output_dim=6, site=134, his=60)
    model.eval()

    # 任意 batch，任意 input_dim（比如 9），只要最后一个是功率就行
    x = torch.randn(32, 134, 60, 7)  # [B, S, T, F]

    with torch.no_grad():
        out = model(x)

    print(f"✅ 输入 shape: {x.shape}")
    print(f"✅ 输出 shape: {out.shape}")

    assert out.shape == (32, 134, 6), f"输出 shape 不对！期望 (32, 134, 6)，得到 {out.shape}"
    print("🎉 Shape 测试通过！")