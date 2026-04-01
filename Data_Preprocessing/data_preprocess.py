import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.spatial.distance import pdist, squareform
import os
import random
import math
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def creat_dict(dir_data, dir_site, site):  # 文件路径、读取站点数
    data = pd.read_csv(dir_data, header=0)
    data = data[data['TurbID'] <= site]

    data.loc[data['Prtv'] < 0, 'Prtv'] = 0
    data.loc[data['Patv'] < 0, 'Patv'] = 0



    column_names = data.columns[3:]
    # 存储最大最小值
    Max = np.full(len(column_names), -np.inf)
    Min = np.full(len(column_names), np.inf)

    for j, column_name in enumerate(column_names):

        if Max[j] < data[column_name].max():
            Max[j] = data[column_name].max()
        if Min[j] > data[column_name].min():
            Min[j] = data[column_name].min()
    normalized_df = data.copy()
    # 检查并修正无效样本
    normalized_df = validate_and_correct_samples(normalized_df)
    # """
    j = 0
    for column_name in column_names:
        # 应用归一化公式
        normalized_df[column_name] = (data[column_name] - Min[j]) / (Max[j] - Min[j])
        j = j + 1
    # """
    data = normalized_df
    del normalized_df

    current_id = data.iloc[0, 0]
    dataset = {}
    i = 0
    min = i
    max = 0

    while i <= len(data) - 2:
        if (len(dataset) == site):
            break

        if data.iloc[i, 0] == current_id:
            # current_data.append(data.iloc[i])
            i = i + 1
        if data.iloc[i, 0] != current_id or i == len(data) - 2:
            # print("当前站点是 %d" % (current_id))
            max = i
            # start = data.iloc[min]
            # last_1 = data.iloc[max - 1]
            # print(max - 2 - min)
            # dataset[current_id] = (min + 1, max)
            dataset[current_id] = np.array(data.iloc[min + 1:max - 1])
            current_id = data.iloc[i, 0]
            min = i
    del data
    data_site = pd.read_csv(dir_site, header=0)
    # site_adj = torch.zeros(134, 134)
    site_adj = torch.zeros(site, site)
    site_x_y = data_site.iloc[:, 1:]
    # 计算所有风电场之间的两两欧几里得距离
    distances = pdist(site_x_y, 'euclidean')
    # 如果你需要查看完整的距离矩阵
    distance_matrix = squareform(distances)
    # 计算所有站点之间距离的平均值
    mean_distance = np.mean(distances)
    # 计算平均距离的一半为阈值
    threshold = mean_distance / 2
    # 计算距离的标准差
    std_distance = np.std(distances)
    i = 0
    # while i < 134:
    while i < site:
        j = 0
        while j < site:
            if distance_matrix[i][j] <= threshold:
                site_adj[i][j] = np.exp(-(distance_matrix[i][j] ** 2) / (std_distance ** 2))
            j = j + 1
        i = i + 1
    # site_adj = np.array(site_adj)
    sub_site = site_adj[:site, :site]
    return dataset, sub_site, Max, Min


def is_valid_sample(row):
    # 检查是否包含空值
    if any(math.isnan(x) if isinstance(x, float) else x is None for x in row):
        return False
    # 检查 Ndir 和 Wdir 是否在合理范围内
    if not (-720 <= row[7] <= 720) or not (-180 <= row[4] <= 180):
        return False
    # 检查风速和有功功率的关系
    if row[12] <= 0 and row[3] > 2.5:
        return False

    # 检查叶片桨距角
    if row[8] > 89 or row[9] > 89 or row[10] > 89:
        return False

    return True


def validate_and_correct_samples(data):
    """函数遍历整个DataFrame，对于每个不符合is_valid_sample条件的行，它会将前10个参数设置为NaN."""
    # 使用列名来提高代码可读性和健壮性
    Wspd = data.columns[3]
    Wdir = data.columns[4]
    Etmp = data.columns[5]
    Itmp = data.columns[6]
    Ndir = data.columns[7]
    Pab1 = data.columns[8]
    Pab2 = data.columns[9]
    Pab3 = data.columns[10]

    Patv = data.columns[12]

    k = 0
    j = 0
    z = 0
    m = 0
    n = 0
    a = 0
    b = 0
    for index, row in data.iterrows():
        # """
        if not (-720 <= row[Ndir] <= 720):
            j = j + 1
            a = a + 1
            
            if row[Ndir] < -720:
                data.at[index, Ndir] = np.nan
            if row[Ndir] > 720:
                data.at[index, Ndir] = np.nan
            # return False
        # """
        if not (-180 <= row[Wdir] <= 180):
            j = j + 1
            z = z + 1
            if row[Wdir] < -180:
                data.at[index, Wdir] = np.nan
            if row[Wdir] > 180:
                data.at[index, Wdir] = np.nan
            # return False
        # """# 检查风速和有功功率的关系
        if row[Patv] <= 0 and row[Wspd] > 2.5:
            j = j + 1
            k = k + 1
            data.at[index, Patv] = np.nan
            # return False

        # 检查叶片桨距角
        if row[Pab1] > 89 or row[Pab2] > 89 or row[Pab3] > 89:
            j = j + 1
            b = b + 1
            data.at[index, Pab1] = np.nan
            data.at[index, Pab2] = np.nan
            data.at[index, Pab3] = np.nan
            # return False
        # """
        if row[Etmp] > 60 or row[Etmp] < -21:
            j = j + 1
            m = m + 1
            data.at[index, Etmp] = np.nan
            # return False
        if row[Itmp] > 70 or row[Itmp] < -21:
            n = n + 1
            j = j + 1
            data.at[index, Itmp] = np.nan
            # return False

    return data


def is_null_sample(row):
    # 检查是否包含空值，返回 True 表示没有空值
    return not any(math.isnan(x) if isinstance(x, float) else x is None for x in row)


def build_sample(data, his=60, future=60, batch_size=32):
    lenght = len(data[1])
    site = len(data)
    sample = []
    Y = []
    i = 0

    while i < lenght - future - his - his:
        sub_sample = []
        sub_y = []
        valid = True  # 标记当前窗口有用
        j = 1
        while j <= site:
            # 检查窗口内所有数据点是否有效
            for k in range(i, i + his + future):
                if not is_null_sample(data[j][k]):
                    valid = False
                    break
            if not valid:
                break
            columns_to_select = [3, 4, 5, 6, 7, 8, 12]  # 选择7个特征
            # columns_to_select = [3, 4, 5,12]
            sub_sample_x = np.array(data[j][i:i + his, columns_to_select].astype(np.float32))
            sub_sample.append(sub_sample_x)
            future_data = np.array(data[j][i + his:i + his + future, [columns_to_select[-1]]].astype(np.float32))
            sub_y.append(future_data)
            j = j + 1
        if valid:
            sub_sample = np.stack(sub_sample, axis=0)
            sub_y = np.stack(sub_y, axis=0)
            sample.append(sub_sample)
            Y.append(sub_y)
        i = i + 1

    sample = np.stack(sample, axis=0)
    Y = np.stack(Y, axis=0)
    del data

    # 将数据转换为 PyTorch Tensors
    samples_x = torch.tensor(sample, dtype=torch.float32)
    samples_y = torch.tensor(Y, dtype=torch.float32)
    print(f"总样本形状: {samples_x.shape}")
    del sample, Y

    # 按时间顺序划分数据集（不打乱）
    total_samples = len(samples_x)
    train_end = int(total_samples * 0.7)  # 前70%作为训练集
    val_end = int(total_samples * 0.85)  # 接下来15%作为验证集

    # 训练集（时间最早的部分）
    train_x = samples_x[:train_end]
    train_y = samples_y[:train_end]

    # 验证集（中间部分）
    val_x = samples_x[train_end:val_end]
    val_y = samples_y[train_end:val_end]

    # 测试集（时间最晚的部分）
    test_x = samples_x[val_end:]
    test_y = samples_y[val_end:]

    print(f"训练集样本数: {len(train_x)}")
    print(f"验证集样本数: {len(val_x)}")
    print(f"测试集样本数: {len(test_x)}")

    # 删除不再需要的数据以释放内存
    del samples_x, samples_y

    # 创建数据集
    train_dataset = TensorDataset(train_x, train_y)
    valid_dataset = TensorDataset(val_x, val_y)
    test_dataset = TensorDataset(test_x, test_y)

    del train_x, val_x, test_x, train_y, val_y, test_y

    # 确保目录存在
    save_directory = '../clean_power_his60_future6_7features'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # 保存数据集
    torch.save(train_dataset.tensors, os.path.join(save_directory, 'train_dataset.pt'))
    torch.save(valid_dataset.tensors, os.path.join(save_directory, 'valid_dataset.pt'))
    torch.save(test_dataset.tensors, os.path.join(save_directory, 'test_dataset.pt'))

    # 保存归一化向量 (Max 和 Min)
    torch.save(Max, os.path.join(save_directory, 'max_values.pt'))
    torch.save(Min, os.path.join(save_directory, 'min_values.pt'))
    torch.save(site_adj, os.path.join(save_directory, 'adj_matrix.pt'))

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # 训练集可以shuffle
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)  # 验证集不shuffle
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 测试集不shuffle

    return train_loader, valid_loader, test_loader

# 微调用于季节性的参数划分
def build_sample_season(data, his=60, future=60, batch_size=32):
    lenght = len(data[1])
    site = len(data)
    sample = []
    Y = []
    i = 0

    while i < lenght - future - his - his + 10:
        sub_sample = []
        sub_y = []
        valid = True  # 标记当前窗口有用
        j = 1
        while j <= site:
            for k in range(i, i + his + future):
                if not is_null_sample(data[j][k]):
                    valid = False
                    break
            if not valid:
                break
            columns_to_select = [3, 4, 5, 6, 7, 8, 12]
            sub_sample_x = np.array(data[j][i:i + his, columns_to_select].astype(np.float32))
            sub_sample.append(sub_sample_x)
            future_data = np.array(data[j][i + his:i + his + future, columns_to_select].astype(np.float32))
            sub_y.append(future_data)
            j = j + 1
        if valid:
            sub_sample = np.stack(sub_sample, axis=0)
            sub_y = np.stack(sub_y, axis=0)
            sample.append(sub_sample)
            Y.append(sub_y)
        i = i + 1

    sample = np.stack(sample, axis=0)
    Y = np.stack(Y, axis=0)
    del data

    # 将数据转换为 PyTorch Tensors
    samples_x = torch.tensor(sample, dtype=torch.float32)
    samples_y = torch.tensor(Y, dtype=torch.float32)
    print(f"总样本形状: {samples_x.shape}")
    del sample, Y

    total_len = len(samples_x)

    # 按时间顺序划分数据集
    # 1. 主训练集：前 60%（时间最早的部分）
    train_end = int(total_len * 0.6)
    train_x = samples_x[:train_end]
    train_y = samples_y[:train_end]

    # 2. 微调训练集：60% ~ 70%（接下来的10%）
    finetune_end = int(total_len * 0.7)
    season_train_x = samples_x[train_end:finetune_end]
    season_train_y = samples_y[train_end:finetune_end]

    # 3. 验证集：70% ~ 85%（接下来15%）
    val_end = int(total_len * 0.85)
    val_x = samples_x[finetune_end:val_end]
    val_y = samples_y[finetune_end:val_end]

    # 4. 测试集：85% ~ 100%（最后15%，时间最晚的部分）
    test_x = samples_x[val_end:]
    test_y = samples_y[val_end:]

    print(f"主训练集样本数: {len(train_x)}")
    print(f"微调训练集样本数: {len(season_train_x)}")
    print(f"验证集样本数: {len(val_x)}")
    print(f"测试集样本数: {len(test_x)}")

    # 删除不再需要的数据以释放内存
    del samples_x, samples_y

    # 创建数据集
    train_dataset = TensorDataset(train_x, train_y)
    season_train_dataset = TensorDataset(season_train_x, season_train_y)
    valid_dataset = TensorDataset(val_x, val_y)
    test_dataset = TensorDataset(test_x, test_y)

    # 释放张量内存
    del train_x, train_y, season_train_x, season_train_y, val_x, val_y, test_x, test_y

    # 确保目录存在
    save_directory = '../clean_power_his60_future6_7features_season'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # 保存数据集
    torch.save(train_dataset.tensors, os.path.join(save_directory, 'train_dataset.pt'))
    torch.save(season_train_dataset.tensors, os.path.join(save_directory, 'season_train_dataset.pt'))
    torch.save(valid_dataset.tensors, os.path.join(save_directory, 'valid_dataset.pt'))
    torch.save(test_dataset.tensors, os.path.join(save_directory, 'test_dataset.pt'))

    # 保存归一化向量 (Max 和 Min)
    torch.save(Max, os.path.join(save_directory, 'max_values.pt'))
    torch.save(Min, os.path.join(save_directory, 'min_values.pt'))

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 主训练集可以shuffle
    season_train_loader = DataLoader(season_train_dataset, batch_size=batch_size, shuffle=False)  # 微调训练集不shuffle
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)  # 验证集不shuffle
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 测试集不shuffle

    return train_loader, valid_loader, test_loader, season_train_loader

# 差分验证和测试
def build_sample_diffblock(data, his=60, future=24, batch_size=32):
    lenght = len(data[1])
    site = len(data)
    sample = []
    Y = []
    i = 0

    while i < lenght - future - his - his:
        sub_sample = []
        sub_y = []
        valid = True  # 标记当前窗口有用
        j = 1
        while j <= site:
            # 检查窗口内所有数据点是否有效
            for k in range(i, i + his + future):
                if not is_null_sample(data[j][k]):
                    valid = False
                    break
            if not valid:
                break
            columns_to_select = [3, 4, 5, 6, 7, 8, 12]  # 选择7个特征
            sub_sample_x = np.array(data[j][i:i + his, columns_to_select].astype(np.float32))
            sub_sample.append(sub_sample_x)
            future_data = np.array(data[j][i + his:i + his + future, [columns_to_select[-1]]].astype(np.float32))
            sub_y.append(future_data)
            j = j + 1
        if valid:
            sub_sample = np.stack(sub_sample, axis=0)
            sub_y = np.stack(sub_y, axis=0)
            sample.append(sub_sample)
            Y.append(sub_y)
        i = i + 1

    sample = np.stack(sample, axis=0)
    Y = np.stack(Y, axis=0)
    del data

    # 将数据转换为 PyTorch Tensors
    samples_x = torch.tensor(sample, dtype=torch.float32)
    samples_y = torch.tensor(Y, dtype=torch.float32)
    print(f"总样本形状: {samples_x.shape}")
    del sample, Y

    total_samples = len(samples_x)

    # 训练集：前 70%
    train_end = int(total_samples * 0.7)
    train_x = samples_x[:train_end-his]
    train_y = samples_y[:train_end-his]

    # 剩余部分（30%）用于验证和测试
    remaining_x = samples_x[train_end:]
    remaining_y = samples_y[train_end:]
    remaining_count = len(remaining_x)
    print(f"测试和验证样本可用数量: {remaining_count}")

    # 将剩余部分划分为 10 个不相交的连续块
    block_size = remaining_count // 10  # 向下取整，确保不重叠
    # 如果不能整除，最后几个样本会被丢弃（可选：也可分配给最后一块，但为严格不相交，这里截断）
    usable_count = block_size * 10
    remaining_x = remaining_x[:usable_count]
    remaining_y = remaining_y[:usable_count]

    # 重塑为 (10, block_size, ...)
    val_blocks_x = []
    val_blocks_y = []
    test_blocks_x = []
    test_blocks_y = []

    for idx in range(10):
        start = idx * block_size
        end = start + block_size
        block_x = remaining_x[start:end-his]
        block_y = remaining_y[start:end-his]
        if idx % 2 == 0:  # 0,2,4,6,8 → 验证集
            val_blocks_x.append(block_x)
            val_blocks_y.append(block_y)
        else:  # 1,3,5,7,9 → 测试集
            test_blocks_x.append(block_x)
            test_blocks_y.append(block_y)

    # 合并所有验证块和测试块
    val_x = torch.cat(val_blocks_x, dim=0) if val_blocks_x else torch.empty(0, *samples_x.shape[1:])
    val_y = torch.cat(val_blocks_y, dim=0) if val_blocks_y else torch.empty(0, *samples_y.shape[1:])
    test_x = torch.cat(test_blocks_x, dim=0) if test_blocks_x else torch.empty(0, *samples_x.shape[1:])
    test_y = torch.cat(test_blocks_y, dim=0) if test_blocks_y else torch.empty(0, *samples_y.shape[1:])

    print(f"训练集样本数: {len(train_x)}")
    print(f"验证集样本数: {len(val_x)}")
    print(f"测试集样本数: {len(test_x)}")

    # 删除不再需要的数据以释放内存
    del samples_x, samples_y, remaining_x, remaining_y

    # 创建数据集
    train_dataset = TensorDataset(train_x, train_y)
    valid_dataset = TensorDataset(val_x, val_y)
    test_dataset = TensorDataset(test_x, test_y)

    del train_x, val_x, test_x, train_y, val_y, test_y

    # 确保目录存在
    save_directory = '../clean_power_his60_future6_7features_diffblock'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # 保存数据集
    torch.save(train_dataset.tensors, os.path.join(save_directory, 'train_dataset.pt'))
    torch.save(valid_dataset.tensors, os.path.join(save_directory, 'valid_dataset.pt'))
    torch.save(test_dataset.tensors, os.path.join(save_directory, 'test_dataset.pt'))

    # 保存归一化向量 (Max 和 Min)
    torch.save(Max, os.path.join(save_directory, 'max_values.pt'))
    torch.save(Min, os.path.join(save_directory, 'min_values.pt'))
    torch.save(site_adj, os.path.join(save_directory, 'adj_matrix.pt'))

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 训练集可以 shuffle
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)  # 验证集不 shuffle
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 测试集不 shuffle

    return train_loader, valid_loader, test_loader

# 不差分验证和测试，跟差分对应数据量
def build_sample_Nodiffblock(data, his=60, future=24, batch_size=32):
    lenght = len(data[1])
    site = len(data)
    sample = []
    Y = []
    i = 0

    while i < lenght - future - his - his:
        sub_sample = []
        sub_y = []
        valid = True  # 标记当前窗口有用
        j = 1
        while j <= site:
            # 检查窗口内所有数据点是否有效
            for k in range(i, i + his + future):
                if not is_null_sample(data[j][k]):
                    valid = False
                    break
            if not valid:
                break
            columns_to_select = [3, 4, 5, 6, 7, 8, 12]  # 选择7个特征
            sub_sample_x = np.array(data[j][i:i + his, columns_to_select].astype(np.float32))
            sub_sample.append(sub_sample_x)
            future_data = np.array(data[j][i + his:i + his + future, [columns_to_select[-1]]].astype(np.float32))
            sub_y.append(future_data)
            j = j + 1
        if valid:
            sub_sample = np.stack(sub_sample, axis=0)
            sub_y = np.stack(sub_y, axis=0)
            sample.append(sub_sample)
            Y.append(sub_y)
        i = i + 1

    sample = np.stack(sample, axis=0)
    Y = np.stack(Y, axis=0)
    del data

    # 将数据转换为 PyTorch Tensors
    samples_x = torch.tensor(sample, dtype=torch.float32)
    samples_y = torch.tensor(Y, dtype=torch.float32)
    print(f"总样本形状: {samples_x.shape}")
    del sample, Y

    total_samples = len(samples_x)

    # 训练集：前 70%
    train_end = int(total_samples * 0.7)
    train_x = samples_x[:train_end-his]
    train_y = samples_y[:train_end-his]

    # 剩余部分（30%）用于验证和测试
    remaining_x = samples_x[train_end:]
    remaining_y = samples_y[train_end:]
    remaining_count = len(remaining_x)
    print(f"测试和验证样本可用数量: {remaining_count}")

    # 将剩余部分划分为 10 个不相交的连续块
    block_size = remaining_count // 10  # 向下取整，确保不重叠
    # 如果不能整除，最后几个样本会被丢弃（可选：也可分配给最后一块，但为严格不相交，这里截断）
    usable_count = block_size * 10
    remaining_x = remaining_x[:usable_count]
    remaining_y = remaining_y[:usable_count]

    # 重塑为 (10, block_size, ...)
    val_blocks_x = []
    val_blocks_y = []
    test_blocks_x = []
    test_blocks_y = []

    for idx in range(10):
        start = idx * block_size
        end = start + block_size
        block_x = remaining_x[start:end-his]
        block_y = remaining_y[start:end-his]
        if idx < 5:  # 前5验证
            val_blocks_x.append(block_x)
            val_blocks_y.append(block_y)
        else:
            test_blocks_x.append(block_x)
            test_blocks_y.append(block_y)

    # 合并所有验证块和测试块
    val_x = torch.cat(val_blocks_x, dim=0) if val_blocks_x else torch.empty(0, *samples_x.shape[1:])
    val_y = torch.cat(val_blocks_y, dim=0) if val_blocks_y else torch.empty(0, *samples_y.shape[1:])
    test_x = torch.cat(test_blocks_x, dim=0) if test_blocks_x else torch.empty(0, *samples_x.shape[1:])
    test_y = torch.cat(test_blocks_y, dim=0) if test_blocks_y else torch.empty(0, *samples_y.shape[1:])

    print(f"训练集样本数: {len(train_x)}")
    print(f"验证集样本数: {len(val_x)}")
    print(f"测试集样本数: {len(test_x)}")

    # 删除不再需要的数据以释放内存
    del samples_x, samples_y, remaining_x, remaining_y

    # 创建数据集
    train_dataset = TensorDataset(train_x, train_y)
    valid_dataset = TensorDataset(val_x, val_y)
    test_dataset = TensorDataset(test_x, test_y)

    del train_x, val_x, test_x, train_y, val_y, test_y

    # 确保目录存在
    save_directory = '../clean_power_his60_future24_7features_Nodiffblock'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # 保存数据集
    torch.save(train_dataset.tensors, os.path.join(save_directory, 'train_dataset.pt'))
    torch.save(valid_dataset.tensors, os.path.join(save_directory, 'valid_dataset.pt'))
    torch.save(test_dataset.tensors, os.path.join(save_directory, 'test_dataset.pt'))

    # 保存归一化向量 (Max 和 Min)
    torch.save(Max, os.path.join(save_directory, 'max_values.pt'))
    torch.save(Min, os.path.join(save_directory, 'min_values.pt'))
    torch.save(site_adj, os.path.join(save_directory, 'adj_matrix.pt'))

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 训练集可以 shuffle
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)  # 验证集不 shuffle
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 测试集不 shuffle

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':

    # 创建时间窗口数据，创建的是历史60步，未来6步的数据，但是只是使用部分未来时间步
    # dir_data = "../sdwpf_kddcup/Cleaned_data_with_power_original_data_with_missing.csv"
    dir_data = "../sdwpf_kddcup/Cleaned_data_with_power.csv"
    # dir_data = "../sdwpf_kddcup/unclean_data.csv"
    dir_site = "../sdwpf_kddcup/sdwpf_baidukddcup2022_turb_location.csv"
    dict, site_adj, Max, Min = creat_dict(dir_data=dir_data, dir_site=dir_site, site=134)

    train_loader, valid_loader, test_loader = build_sample_Nodiffblock(dict, his=60, future=24, batch_size=32)
    # train_loader, valid_loader, test_loader, season_train_loader = build_sample_season(dict, his=60, future=6, batch_size=32)
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for inputs, targets in train_loader:
        break  # 获取第一个批次
    print(inputs.shape)
    print(targets.shape)

