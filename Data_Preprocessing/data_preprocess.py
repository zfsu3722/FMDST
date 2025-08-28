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
        if not (-720 <= row[Ndir] <= 720):
            j = j + 1
            a = a + 1
            # """
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
        # 检查风速和有功功率的关系
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
            # for k in range(i, i + his+1+future):
            for k in range(i, i + his + future):
                if not is_null_sample(data[j][k]):
                    valid = False
                    break
            if not valid:
                break
            columns_to_select = [3] + [4] + [5]+ [6] + [7] + [8] + [12]
            # columns_to_select = [3]  + [7] + [8]  + [12]
            sub_sample_x = np.array(data[j][i:i + his, columns_to_select].astype(np.float32))
            sub_sample.append(sub_sample_x)
            future_data = np.array(data[j][i + his:i + his + future, columns_to_select].astype(np.float32))
            # 要保留的列索引（注意：Python 索引从 0 开始）
            # future_data = future_data[5::6 ]
            # future_data = future_data[:]
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

    sample = np.array(sample)
    Y = np.array(Y)
    # 将数据转换为 PyTorch Tensors
    samples_x = torch.tensor(sample, dtype=torch.float32)
    samples_y = torch.tensor(Y, dtype=torch.float32)
    print(samples_x.shape)
    del sample, Y

    # 假设 samples_x 和 samples_y 是已经定义好的输入特征和目标变量
    # indices = np.random.permutation(len(samples_x))

    # 打乱后的索引用于重新排列 samples_x 和 samples_y
    # samples_x = samples_x[indices]
    # samples_y = samples_y[indices]

    train_x = samples_x[:int((len(samples_x) * 0.7))]
    train_y = samples_y[:int((len(samples_y) * 0.7))]

    remaining_x = samples_x[int((len(samples_x) * 0.7)):]
    remaining_y = samples_y[int((len(samples_y) * 0.7)):]
    # 假设 samples_x 和 samples_y 是已经定义好的输入特征和目标变量
    indices = np.random.permutation(len(remaining_x))

    # 打乱后的索引用于重新排列 samples_x 和 samples_y
    remaining_x = remaining_x[indices]
    remaining_y = remaining_y[indices]

    val_x = remaining_x[:int((len(remaining_x) * 0.33333))]
    test_x = remaining_x[int((len(remaining_x) * 0.33333)):int((len(remaining_x) * 0.66666))]

    val_y = remaining_y[:int((len(remaining_y) * 0.33333))]
    test_y = remaining_y[int((len(remaining_y) * 0.33333)):int((len(remaining_y) * 0.66666))]
    # 保留部分可以用于验证组合实验
    test_x_conbination = remaining_x[int((len(remaining_x) * 0.66666)):]
    test_y_conbination = remaining_y[int((len(remaining_y) * 0.66666)):]

    # 删除不再需要的数据以释放内存
    del samples_x, samples_y, remaining_x, remaining_y
    # 创建数据集
    train_dataset = TensorDataset(train_x, train_y)
    valid_dataset = TensorDataset(val_x, val_y)
    test_dataset = TensorDataset(test_x, test_y)
    test_dataset_conbination = TensorDataset(test_x_conbination, test_y_conbination)
    del train_x, val_x, test_x, train_y, val_y, test_y, test_x_conbination, test_y_conbination

    # 确保目录存在
    save_directory = '../clean_power_his60_future60_7features'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # 保存数据集
    torch.save(train_dataset.tensors, os.path.join(save_directory, 'train_dataset.pt'))
    torch.save(valid_dataset.tensors, os.path.join(save_directory, 'valid_dataset.pt'))
    torch.save(test_dataset.tensors, os.path.join(save_directory, 'test_dataset.pt'))
    torch.save(test_dataset_conbination.tensors, os.path.join(save_directory, 'test_dataset_conbination.pt'))

    # 保存归一化向量 (Max 和 Min)
    torch.save(Max, os.path.join(save_directory, 'max_values.pt'))
    torch.save(Min, os.path.join(save_directory, 'min_values.pt'))

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

if __name__ == '__main__':
    # 创建时间窗口数据，创建的是历史60步，未来60步的数据，但是只是使用部分未来时间步
    dir_data = "../sdwpf_kddcup/Cleaned_data_with_power.csv"
    dir_site = "../sdwpf_kddcup/sdwpf_baidukddcup2022_turb_location.csv"
    dict, site_adj, Max, Min = creat_dict(dir_data=dir_data, dir_site=dir_site, site=134)
    train_loader, valid_loader, test_loader = build_sample(dict, his=60, future=60, batch_size=32)
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for inputs, targets in train_loader:
        break  # 获取第一个批次
    print(inputs.shape)
    print(targets.shape)