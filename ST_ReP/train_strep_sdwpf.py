import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from ST_ReP.sdwpf_data_loader import get_sdwpf_dataloaders
from ST_ReP.strep_model_full import STReP


def train_model():
    # 1. 首先预处理数据（如果需要）
    from sdwpf_preprocess_strep import create_sdwpf_dataset
    data_path = './SDWPF/SDWPF.npz'
    if not os.path.exists(data_path):
        print("创建数据集...")
        create_sdwpf_dataset()

    # 2. 设置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_len = 60
    pred_len = 6
    batch_size = 32

    # 3. 加载数据
    print("加载数据...")
    train_loader, val_loader, test_loader, scaler = get_sdwpf_dataloaders(
        seq_len=seq_len, pred_len=pred_len, batch_size=batch_size
    )

    # 验证数据形状
    for x, t, y in train_loader:
        print(f"输入: {x.shape}, 时间: {t.shape}, 目标: {y.shape}")
        break

    # 4. 创建模型参数（直接在这里定义）
    class ModelArgs:
        def __init__(self):
            # SDWPF数据集参数
            self.num_nodes = 134  # 134个风力涡轮机
            self.input_length = 60  # 将在这里设置
            self.predict_length = 6  # 将在这里设置
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

    # 创建参数对象
    args = ModelArgs()
    args.input_length = seq_len
    args.predict_length = pred_len

    # 5. 创建模型（正确方式）
    model = STReP(args).to(device)

    # 6. 修改模型调用方式以适应您的训练循环
    # 原论文模型返回4个值，我们需要的是第4个（predy）
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 7. 训练
    print("\n开始训练...")
    best_loss = float('inf')

    for epoch in range(100):
        model.train()
        train_loss = 0

        for batch_x, batch_t, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_t = batch_t.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # STReP模型期望的输入格式是 (x, x_mark, y)
            # 它返回 (inputx, reconx, inputy, predy)
            _, _, _, pred = model((batch_x, batch_t, batch_y))

            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_t, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_t = batch_t.to(device)
                batch_y = batch_y.to(device)

                _, _, _, pred = model((batch_x, batch_t, batch_y))
                loss = criterion(pred, batch_y)
                val_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Epoch {epoch}: 保存最佳模型 (val_loss: {avg_val:.4f})")

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}")

    # 8. 测试
    print("\n测试...")
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for batch_x, batch_t, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_t = batch_t.to(device)
            batch_y = batch_y.to(device)

            _, _, _, pred = model((batch_x, batch_t, batch_y))

            # 反标准化
            pred_np = pred.cpu().numpy()
            target_np = batch_y.cpu().numpy()

            for i in range(pred_np.shape[0]):
                pred_denorm = scaler.inverse_transform(pred_np[i])
                target_denorm = scaler.inverse_transform(target_np[i])
                predictions.append(pred_denorm)
                targets.append(target_denorm)

    predictions = np.array(predictions)
    targets = np.array(targets)

    # 计算指标
    mae = np.abs(predictions - targets).mean()
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    print(f"\n测试结果:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"预测形状: {predictions.shape}")

    # 每个预测步长的指标
    print("\n各预测步长:")
    for i in range(pred_len):
        step_mae = np.abs(predictions[:, i] - targets[:, i]).mean()
        step_rmse = np.sqrt(np.mean((predictions[:, i] - targets[:, i]) ** 2))
        print(f"步 {i + 1}: MAE={step_mae:.4f}, RMSE={step_rmse:.4f}")

    return model, mae, rmse


if __name__ == "__main__":
    train_model()