import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import random
from model.LSTM_Transformer import LSTM_Transformer
from model.TCN_Transformer import TCN_Transformer
from model.FMDST import FMDST
from model.LSTM import ShareLSTM
from model.share_Transformer_share_fc import SharedTransformer


def evaluate_step_metrics(predictions, targets):
    """
    输入:
        predictions: np.array, shape = [batch_size, num_sites, output_dim]
        targets: np.array, shape = [batch_size, num_sites, output_dim]

    输出:
        {
            'mae_per_step': [step_0_mae, step_1_mae, ..., step_T_mae],
            'rmse_per_step': 同上,
            'r2_per_step': 同上
        }
    其中每个 step 的指标是基于所有 batch × 所有站点 在该预测时间步上的误差。
    """

    # 确保输入是 numpy 数组
    predictions = np.array(predictions)
    targets = np.array(targets)

    # 检查维度是否正确
    assert predictions.ndim == 3 and targets.ndim == 3, "输入必须是三维数组 [batch, site, output_dim]"
    assert predictions.shape == targets.shape, "预测值与真实值的形状必须一致"

    batch_size, num_sites, output_dim = predictions.shape

    mae_per_step = []
    rmse_per_step = []
    r2_per_step = []

    for step in range(output_dim):
        # 取出当前时间步的所有预测值和真实值
        pred_step = predictions[:, :, step]  # shape: [batch_size, num_sites]
        target_step = targets[:, :, step]    # shape: [batch_size, num_sites]

        # 展平成一维向量，合并 batch 和 site 维度
        pred_flat = pred_step.flatten()     # shape: [batch_size * num_sites]
        target_flat = target_step.flatten()

        # 计算 MAE
        mae = np.mean(np.abs(pred_flat - target_flat))

        # 计算 RMSE
        rmse = np.sqrt(np.mean((pred_flat - target_flat) ** 2))

        # 计算 R²
        ss_res = np.sum((target_flat - pred_flat) ** 2)
        ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')

        mae_per_step.append(round(float(mae), 4))
        rmse_per_step.append(round(float(rmse), 4))
        r2_per_step.append(round(float(r2), 4))

    return {
        'mae_per_step': mae_per_step,
        'rmse_per_step': rmse_per_step,
        'r2_per_step': r2_per_step
    }



def train_and_evaluate_model(
    model_class,
    save_directory_data='clean_power_his60_future60_7features',
    input_dim=7,
    output_dim=60,
    batch_size=32,
    num_epochs=1000,
    patience=20,
    lr=0.002,
    model_save_dir='model_save',

):
    # 动态生成 current_file_name
    model_name = model_class.__name__
    current_file_name = f'Power_Pred_{model_name}_his60_Pred60_134site_7feature'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f'Selected device: {device}')

    # 检查数据目录是否存在
    if not os.path.exists(save_directory_data):
        raise FileNotFoundError(f"The directory {save_directory_data} does not exist.")

    # 加载数据集
    train_data = torch.load(os.path.join(save_directory_data, 'train_dataset.pt'))
    valid_data = torch.load(os.path.join(save_directory_data, 'valid_dataset.pt'))
    test_data = torch.load(os.path.join(save_directory_data, 'test_dataset.pt'))

    train_dataset = TensorDataset(train_data[0], train_data[1])
    valid_dataset = TensorDataset(valid_data[0], valid_data[1])
    test_dataset = TensorDataset(test_data[0], test_data[1])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 归一化向量
    max_values = torch.load(os.path.join(save_directory_data, 'max_values.pt'))
    min_values = torch.load(os.path.join(save_directory_data, 'min_values.pt'))
    # site_adj = torch.load(os.path.join(save_directory_data, 'adj_matrix.pt')).to(device)

    # 创建模型
    model = model_class(input_dim=input_dim, output_dim=output_dim).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 模型保存路径
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_file = os.path.join(model_save_dir, f'{current_file_name}_best_model.pth')

    # ⏱️ 开始计时
    start_time = time.time()

    # 训练循环
    best_val_loss = float('inf')
    best_model_wts = None
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)  # 默认输入 shape: (B, S, T, F)

            loss = criterion(outputs, targets[:, :, :, -1])
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, targets[:, :, :, -1])
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(valid_loader)


        # 早停机制
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = model.state_dict().copy()
            torch.save(best_model_wts, best_model_file)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    # ⏱️ 结束训练计时
    training_time = time.time() - start_time
    # 🕒 打印训练时间
    print(f"Training time: {training_time:.2f} seconds ({training_time / 60:.2f} minutes)")
    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_file))

    # ⏱️ 【2】开始测试计时
    test_start = time.time()
    # 测试
    model.eval()
    total_test_loss = 0
    predictions, targets_list = [], []
    with torch.no_grad():
        for inputs, targets_batch in test_loader:
            inputs, targets_batch = inputs.to(device), targets_batch.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets_batch[:, :, :, -1])
            total_test_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            targets_list.extend(targets_batch[:, :, :, -1].cpu().numpy())

    test_time = time.time() - test_start  # ✅ 测试总耗时
    print(f"Test time:       {test_time:.2f} s ({test_time / 60:.2f} min)")


    average_test_loss = total_test_loss / len(test_loader)
    predictions = np.array(predictions)
    targets_array = np.array(targets_list)

    # 反归一化
    Max_s = max_values[-1]
    Min_s = min_values[-1]
    predictions = predictions * (Max_s - Min_s) + Min_s
    targets_array = targets_array * (Max_s - Min_s) + Min_s
    print("---------------------------")
    print(predictions.shape)
    print(targets_array.shape)
    print("---------------------------")

    # 指标计算（总）
    mae_total = np.mean(np.abs(predictions - targets_array))
    rmse_total = np.sqrt(np.mean((predictions - targets_array) ** 2))
    r2_total = 1 - ((np.sum((predictions - targets_array) ** 2)) / np.sum((targets_array - np.mean(targets_array)) ** 2))

    # 每一步的指标
    step_metrics = evaluate_step_metrics(predictions, targets_array)

    # 打印
    print(f'Test Loss: {average_test_loss:.4f}, MAE: {mae_total:.4f}, RMSE: {rmse_total:.4f}, R²: {r2_total:.4f}')

    return {
        'mae': mae_total,
        'rmse': rmse_total,
        'r2': r2_total,
        'mae_per_step': step_metrics['mae_per_step'],
        'rmse_per_step': step_metrics['rmse_per_step'],
        'r2_per_step': step_metrics['r2_per_step']
    }


# finetune function
def fine_tune_and_evaluate_model(
        model_class,
        save_directory_data='clean_power_his60_future6_7features_season',
        input_dim=7,
        output_dim=6,
        batch_size=32,
        num_epochs=100,
        patience=15,
        lr=0.001,
        model_save_dir='model_save',  # 预训练模型所在目录
):
    """
    在 season_train (60%-70%) 上微调预训练模型，并在统一的 valid/test (70%-100%) 上评估。
    """
    model_name = model_class.__name__
    pretrained_file_name = f'Power_Pred_{model_name}_his60_Pred6_134site_7feature_Season_best_model.pth'
    pretrained_model_path = os.path.join(model_save_dir, pretrained_file_name)

    current_file_name = f'Power_Pred_{model_name}_his60_Pred6_134site_7feature_Season'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ 加载微调训练集（60%-70%）
    train_data = torch.load(os.path.join(save_directory_data, 'season_train_dataset.pt'))
    train_dataset = TensorDataset(train_data[0], train_data[1])

    # ✅ 加载统一的验证集和测试集（来自 70%-100%，已打乱并保存为 valid_dataset.pt / test_dataset.pt）
    valid_data = torch.load(os.path.join(save_directory_data, 'valid_dataset.pt'))
    test_data = torch.load(os.path.join(save_directory_data, 'test_dataset.pt'))
    valid_dataset = TensorDataset(valid_data[0], valid_data[1])
    test_dataset = TensorDataset(test_data[0], test_data[1])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 归一化参数
    max_values = torch.load(os.path.join(save_directory_data, 'max_values.pt'))
    min_values = torch.load(os.path.join(save_directory_data, 'min_values.pt'))

    # 创建模型并加载预训练权重
    model = model_class(input_dim=input_dim, output_dim=output_dim).to(device)
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    print(f"✅ Loaded pretrained weights from: {pretrained_model_path}")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 微调后模型保存路径
    finetune_save_dir = 'model_save'
    os.makedirs(finetune_save_dir, exist_ok=True)
    best_model_file = os.path.join(finetune_save_dir, f'{current_file_name}_best_finetuned.pth')

    # 微调训练循环
    best_val_loss = float('inf')
    counter = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # 注意：确保 targets[:, :, :6, -1] 与你的模型输出 shape 一致
            loss = criterion(outputs, targets[:, :, :6, -1])
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # 验证
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets[:, :, :6, -1])
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(valid_loader)

        # 早停 & 保存最佳
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_file)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    training_time = time.time() - start_time
    print(f"Fine-tuning time: {training_time:.2f}s ({training_time / 60:.2f} min)")

    # 加载最佳微调模型用于测试
    model.load_state_dict(torch.load(best_model_file, map_location=device))

    # 测试
    test_start = time.time()
    model.eval()
    predictions, targets_list = [], []
    with torch.no_grad():
        for inputs, targets_batch in test_loader:
            inputs, targets_batch = inputs.to(device), targets_batch.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            targets_list.extend(targets_batch[:, :, :6, -1].cpu().numpy())
    test_time = time.time() - test_start
    print(f"Test time: {test_time:.2f}s")

    # 反归一化
    Max_s = max_values[-1]
    Min_s = min_values[-1]
    predictions = np.array(predictions) * (Max_s - Min_s) + Min_s
    targets_array = np.array(targets_list) * (Max_s - Min_s) + Min_s

    # 总体指标
    mae_total = np.mean(np.abs(predictions - targets_array))
    rmse_total = np.sqrt(np.mean((predictions - targets_array) ** 2))
    r2_total = 1 - (np.sum((predictions - targets_array) ** 2) / np.sum((targets_array - np.mean(targets_array)) ** 2))

    # 分步指标（假设 evaluate_step_metrics 已定义）
    step_metrics = evaluate_step_metrics(predictions, targets_array)

    print(f'[Fine-tuned] MAE: {mae_total:.4f}, RMSE: {rmse_total:.4f}, R²: {r2_total:.4f}')

    return {
        'mae': mae_total,
        'rmse': rmse_total,
        'r2': r2_total,
        'mae_per_step': step_metrics['mae_per_step'],
        'rmse_per_step': step_metrics['rmse_per_step'],
        'r2_per_step': step_metrics['r2_per_step']
    }


# 用于评估定制缺失补全后的数据集函数
def evaluate_data_model(
    model_class,
    save_directory_data='clean_power_original_data_with_missing_his60_future6_7features',
    input_dim=7,
    output_dim=6,
    batch_size=32,
    model_save_dir='model_save',
):
    """
    加载已保存的模型并在 test_dataset 上进行评估（无训练）。
    模型权重文件名格式：Power_Pred_{ModelName}_his60_Pred6_134site_7feature_Season_best_model.pth
    """
    model_name = model_class.__name__
    eval_model_file_name = f'Power_Pred_{model_name}_his60_Pred6_134site_7feature_best_model.pth'
    eval_model_path = os.path.join(model_save_dir, eval_model_file_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 检查数据目录是否存在
    if not os.path.exists(save_directory_data):
        raise FileNotFoundError(f"The directory {save_directory_data} does not exist.")

    # ✅ 只加载测试集
    test_data = torch.load(os.path.join(save_directory_data, 'test_dataset.pt'))
    test_dataset = TensorDataset(test_data[0], test_data[1])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 归一化参数
    max_values = torch.load(os.path.join(save_directory_data, 'max_values.pt'))
    min_values = torch.load(os.path.join(save_directory_data, 'min_values.pt'))

    # 创建模型并加载待评估的权重
    model = model_class(input_dim=input_dim, output_dim=output_dim).to(device)
    if not os.path.exists(eval_model_path):
        raise FileNotFoundError(f"Evaluation model not found at: {eval_model_path}")
    model.load_state_dict(torch.load(eval_model_path, map_location=device))
    model.eval()
    print(f"✅ Loaded model for evaluation from: {eval_model_path}")

    # 测试推理
    test_start = time.time()
    predictions, targets_list = [], []
    with torch.no_grad():
        for inputs, targets_batch in test_loader:
            inputs, targets_batch = inputs.to(device), targets_batch.to(device)
            outputs = model(inputs)
            # 提取目标：targets shape [B, N, 6, T] → 取最后时间步的前6个预测变量
            targets_selected = targets_batch[:, :, :output_dim, -1]
            predictions.extend(outputs.cpu().numpy())
            targets_list.extend(targets_selected.cpu().numpy())
    test_time = time.time() - test_start
    print(f"Test time: {test_time:.2f}s ({test_time / 60:.2f} min)")

    # 转为 numpy 并反归一化（使用最后一个特征的 min/max，即功率）
    predictions = np.array(predictions)
    targets_array = np.array(targets_list)
    Max_s = max_values[-1]
    Min_s = min_values[-1]
    predictions = predictions * (Max_s - Min_s) + Min_s
    targets_array = targets_array * (Max_s - Min_s) + Min_s

    print("---------------------------")
    print("Predictions shape:", predictions.shape)
    print("Targets shape:    ", targets_array.shape)
    print("---------------------------")

    # 总体指标
    mae_total = np.mean(np.abs(predictions - targets_array))
    rmse_total = np.sqrt(np.mean((predictions - targets_array) ** 2))
    r2_total = 1 - (np.sum((predictions - targets_array) ** 2) /
                    np.sum((targets_array - np.mean(targets_array)) ** 2))

    # 分步指标
    step_metrics = evaluate_step_metrics(predictions, targets_array)

    print(f'[Evaluation] MAE: {mae_total:.4f}, RMSE: {rmse_total:.4f}, R²: {r2_total:.4f}')

    return {
        'mae': mae_total,
        'rmse': rmse_total,
        'r2': r2_total,
        'mae_per_step': step_metrics['mae_per_step'],
        'rmse_per_step': step_metrics['rmse_per_step'],
        'r2_per_step': step_metrics['r2_per_step']
    }


def visualize_predictions(
        models,
        model_names,
        model_weight_paths,
        data_path='clean_power_his60_future60_7features',
        batch_size=32,
        batch_idx=0,  # 第几个 batch（从 test_loader 中取）
        sample_in_batch=0,  # 在该 batch 中选第几个样本（0 ~ batch_size-1）
        site_idx=0,  # 选哪个风机（0 ~ N-1）
        device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 1. 按训练时的方式加载测试数据 ===
    test_data = torch.load(os.path.join(data_path, 'test_dataset.pt'))
    test_dataset = TensorDataset(test_data[0], test_data[1])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)  # 必须 shuffle=False！

    # 获取第 batch_idx 个 batch
    inputs, targets = next(iter(test_loader))
    for i, (inp, tar) in enumerate(test_loader):
        if i == batch_idx:
            inputs, targets = inp, tar
            break
    B, N, H_in, F = inputs.shape
    _, _, H_out, _ = targets.shape


    print(f"Using batch {batch_idx}, sample_in_batch={sample_in_batch}, site={site_idx}")

    # 提取真实值（用于绘图）
    x_hist = inputs[sample_in_batch, site_idx, :, -1].cpu().numpy()  # (H_in,)
    y_true = targets[sample_in_batch, site_idx, :, -1].cpu().numpy()  # (H_out,)

    # 反归一化参数（这里假设已知如何获取max_val和min_val）
    max_val = torch.load(os.path.join(data_path, 'max_values.pt'))[-1]
    min_val = torch.load(os.path.join(data_path, 'min_values.pt'))[-1]
    x_hist = x_hist * (max_val - min_val) + min_val
    y_true = y_true * (max_val - min_val) + min_val

    # 时间轴
    history_len = len(x_hist)
    horizon_len = len(y_true)
    x_time = np.arange(history_len)
    y_time = np.arange(history_len, history_len + horizon_len)

    # === 2. 所有模型对整个 batch 推理（保持 batch_size=32）===
    predictions = {}
    for name, model_class, weight_path in zip(model_names, models, model_weight_paths):
        model = model_class(input_dim=7, output_dim=60).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()

        with torch.no_grad():
            inputs_batch = inputs.to(device)  # (B, N, H_in, F)
            pred = model(inputs_batch)  # 应输出 (B, N, H_out, 1)

            # 提取指定样本和风机
            pred_site = pred[sample_in_batch, site_idx, :].cpu().numpy()  # (H_out,)
            pred_site = pred_site * (max_val - min_val) + min_val
            predictions[name] = pred_site

        # === 3. 绘图 ===
        plt.figure(figsize=(14, 6))

        # 历史输入（黑色实线+圆点）
        plt.plot(x_time, x_hist, 'k-o', label='Historical Input (Patv)', markersize=3)

        # 真实未来（绿色实线+方块）
        plt.plot(y_time, y_true, 'g-s', label='Ground Truth Future', markersize=4)

        # 各模型预测（自动颜色，虚线+三角）
        for name, pred in predictions.items():
            plt.plot(y_time, pred, marker='^', linestyle='--', label=f'{name} Prediction')

        # 分隔线：“Now”
        plt.axvline(x=history_len - 0.5, color='gray', linestyle=':', linewidth=1)
        plt.text(history_len - 2, plt.ylim()[1] * 0.9, 'Now', rotation=90, color='gray', fontsize=10)

        # 标签与标题
        plt.xlabel('Time Step (10-min intervals)')
        plt.ylabel('Active Power (kW)')
        plt.title(
            f'Wind Power Forecasting Comparison\n(Batch={batch_idx}, SampleInBatch={sample_in_batch}, Site={site_idx})')

        # 图例、网格、布局
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        # 保存并显示
        plt.savefig(f'forecast_comparison_b{batch_idx}_s{sample_in_batch}_site{site_idx}.png', dpi=300)
        plt.show()


if __name__ == "__main__":
    """
    Comparative model drawing
    """
    models = [ShareLSTM,SharedTransformer, FMDST]
    model_names = ["ShareLSTM", "SharedTransformer", "FMDST"]
    weight_paths = [
        "model_save/Power_Pred_ShareLSTM_his60_Pred60_134site_7feature_best_model.pth",
        "model_save/Power_Pred_SharedTransformer_his60_Pred60_134site_7feature_best_model.pth",
        "model_save/Power_Pred_FMDST_his60_Pred60_134site_7feature_best_model.pth",
    ]


    visualize_predictions(
        models=models,
        model_names=model_names,
        model_weight_paths=weight_paths,
        batch_size=32,
        batch_idx=27,           # 第 5 个 batch（对应样本 160~191）
        sample_in_batch=27,     # 该 batch 中第 3 个样本
        site_idx=27,           # 第 10 号风机
        data_path='clean_power_his60_future60_7features'
    )




