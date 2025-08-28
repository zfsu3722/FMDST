import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset



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
    output_dim=6,
    batch_size=32,
    num_epochs=1000,
    patience=20,
    lr=0.002,
    model_save_dir='model_save',

):
    # 动态生成 current_file_name
    model_name = model_class.__name__
    current_file_name = f'Power_Pred_{model_name}_his60_Pred1-6_134site_7feature'
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

            loss = criterion(outputs, targets[:, :, 0:6, -1])
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

                loss = criterion(outputs, targets[:, :, 0:6, -1])
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

    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_file))



    # 测试
    model.eval()
    total_test_loss = 0
    predictions, targets_list = [], []
    with torch.no_grad():
        for inputs, targets_batch in test_loader:
            inputs, targets_batch = inputs.to(device), targets_batch.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets_batch[:, :, 0:6, -1])
            total_test_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            targets_list.extend(targets_batch[:, :, 0:6, -1].cpu().numpy())

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