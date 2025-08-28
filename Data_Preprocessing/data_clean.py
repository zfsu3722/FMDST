import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import cdist

def interpolate_and_clean_data_Ndir_Pab(data_slice, cols_to_interpolate, cols_to_clip):
    """对数据切片进行插值和清理"""
    # 确保使用的是副本
    data = data_slice.copy()

    # 插值
    for col in cols_to_interpolate:
        data.loc[:, col] = data[col].interpolate(method='linear', limit_direction='both')

    # 异常值处理
    for col, (lower, upper) in cols_to_clip.items():
        data.loc[data[col] > upper, col] = upper
        data.loc[data[col] < lower, col] = lower

    # 处理 Prtv 和 Patv 小于0的情况
    data.loc[data['Prtv'] < 0, 'Prtv'] = 0
    data.loc[data['Patv'] < 0, 'Patv'] = 0

    return data


def creat_dict_Ndir_Pab(dir_data, site):
    data = pd.read_csv(dir_data, header=0)
    data = data[data['TurbID'] <= site].copy()  # 添加 copy() 避免警告

    current_id = data.iloc[0, 0]
    dataset = {}
    i = 0
    min_idx = i
    max_idx = 0

    cols_to_interpolate = ['Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv']
    cols_to_clip = {
        'Ndir': (-720, 720),
        'Pab1': (-89, 89),
        'Pab2': (-89, 89),
        'Pab3': (-89, 89)
    }

    while i <= len(data) - 2:
        if len(dataset) == site:
            break

        if data.iloc[i, 0] == current_id:
            i += 1
        if data.iloc[i, 0] != current_id or i == len(data) - 2:
            max_idx = i
            # 对每个站点的数据进行插值和清理
            data_slice = data.iloc[min_idx:max_idx].copy()
            data_slice = interpolate_and_clean_data_Ndir_Pab(data_slice, cols_to_interpolate, cols_to_clip)
            dataset[current_id] = np.array(data_slice)
            current_id = data.iloc[i, 0] if i < len(data) else None
            min_idx = i

    del data

    # 将数据写回CSV文件
    output_df = pd.DataFrame()
    for turb_id, values in dataset.items():
        temp_df = pd.DataFrame(values,
                               columns=['TurbID', 'Day', 'Tmstamp', 'Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1',
                                        'Pab2', 'Pab3', 'Prtv', 'Patv'])
        output_df = pd.concat([output_df, temp_df], ignore_index=True)

    output_df.to_csv(dir_data.replace('.csv', '_Ndir_pab_processed.csv'), index=False)

    return dataset

def normalize_angle(angle):
    """将角度归一化到 -180 到 +180 范围"""
    return (angle + 180) % 360 - 180

def process_wind_direction(df):
    """
    将相对风向转换为绝对风向，并更新回 'Wdir' 列；
    同时将 'Ndir' 归一化。
    """
    # 检查是否包含所需列
    if 'Wdir' not in df.columns or 'Ndir' not in df.columns:
        raise ValueError("DataFrame 必须包含 'Wdir' 和 'Ndir' 列")

    # 计算绝对风向：Wdir += Ndir
    df['Wdir'] += df['Ndir']

    # 归一化 Wdir 和 Ndir
    df['Wdir'] = df['Wdir'].apply(normalize_angle)
    df['Ndir'] = df['Ndir'].apply(normalize_angle)

    return df

def assign_sites_to_groups(data_site, num_groups=10):
    """使用层次聚类将站点分配到指定数量的组中"""
    coords = data_site[['x', 'y']].values
    distances = pdist(coords, 'euclidean')
    distance_matrix = squareform(distances)

    # 层次聚类
    Z = linkage(distance_matrix, method='ward')

    # 分割成指定数量的组
    groups = fcluster(Z, num_groups, criterion='maxclust')

    # 将TurbID与组号对应起来
    site_groups = {i: [] for i in range(1, num_groups + 1)}
    for idx, group_id in enumerate(groups):
        site_groups[group_id].append(data_site.loc[idx, 'TurbID'])

    return site_groups

def mark_anomalies_as_nan(data, anomaly_rules):
    """将异常值标记为NaN"""
    for col, (min_val, max_val) in anomaly_rules.items():
        if min_val is not None and max_val is not None:
            # 将超出范围的数值标记为NaN
            data.loc[~data[col].between(min_val, max_val), col] = np.nan
    return data

def impute_with_group_means(data, site_groups, cols_to_impute):
    """
    使用组内同时间点的均值进行插补
    1. 将数据按组和时间点分组
    2. 计算每个组每个时间点的均值
    3. 用均值填补缺失值
    """
    # 创建反向映射：TurbID -> group_id
    turb_to_group = {turb: group_id for group_id, turb_list in site_groups.items() for turb in turb_list}
    data['group_id'] = data['TurbID'].map(turb_to_group)

    # 创建时间键（Day + Tmstamp）
    data['time_key'] = data['Day'].astype(str) + '_' + data['Tmstamp'].astype(str)

    # 计算每个组每个时间点的均值
    group_means = data.groupby(['group_id', 'time_key'])[cols_to_impute].mean().reset_index()

    print("均值计算完")

    # 合并均值回原始数据
    for col in cols_to_impute:
        # 为每个列创建均值列
        mean_col = f'{col}_mean'
        group_means_renamed = group_means.rename(columns={col: mean_col})[['group_id', 'time_key', mean_col]]
        data = data.merge(group_means_renamed, on=['group_id', 'time_key'], how='left')

        # 用均值填补缺失值
        data.loc[data[col].isna(), col] = data.loc[data[col].isna(), mean_col]

        # 删除临时列
        data.drop(columns=[mean_col], inplace=True)
        print(col)

    # 删除辅助列
    data.drop(columns=['group_id', 'time_key'], inplace=True)

    return data

def process_all_turbines_by_group(input_file, output_file, site_file, num_groups=10):
    # 读取数据
    data = pd.read_csv(input_file)
    data_site = pd.read_csv(site_file)

    # 分组
    site_groups = assign_sites_to_groups(data_site, num_groups=num_groups)

    # 异常标记
    anomaly_rules = {
        'Wspd': (-np.inf, np.inf),
        'Wdir': (-180, 180),
        'Etmp': (-21, 60),
        'Itmp': (-21, 70)
    }
    data = mark_anomalies_as_nan(data, anomaly_rules)

    # 插补
    cols_to_impute = ['Wspd', 'Wdir', 'Etmp', 'Itmp']
    processed_data = impute_with_group_means(data, site_groups, cols_to_impute)

    # 保存结果
    processed_data.to_csv(output_file, index=False)
    print(f"处理完成，结果已保存到 {output_file}")

def process_power_data_only(input_file, output_file, site_file, num_groups=10):
    """
    仅对功率数据进行异常检测与插补处理，并输出哪些站点、日期和时间点的 Patv 被填补。

    参数:
    - input_file: 输入CSV路径（已清洗完天气数据）
    - output_file: 输出CSV路径（包含修复后的功率数据）
    - site_file: 风机地理位置文件路径
    - num_groups: 分组数量（默认10组）
    """

    # Step 1: 读取数据
    data = pd.read_csv(input_file)
    data_site = pd.read_csv(site_file)

    # Step 2: 按地理距离聚类分组
    site_groups = assign_sites_to_groups(data_site, num_groups=num_groups)

    # Step 3: 特别处理 Patv 异常：Wspd > 2.5 且 Patv <= 0 的情况
    print("开始标记功率异常值...")
    data.loc[(data['Wspd'] > 2.5) & (data['Patv'] <= 0), 'Patv'] = np.nan
    print("功率异常标记完成。")

    # Step 4: 使用同组、同时刻均值插补 Patv 缺失值
    cols_to_impute = ['Patv']
    print("开始使用同组、同时刻均值插补 Patv 缺失值...")

    # 创建反向映射：TurbID -> group_id
    turb_to_group = {
        turb: group_id
        for group_id, turb_list in site_groups.items()
        for turb in turb_list
    }
    data['group_id'] = data['TurbID'].map(turb_to_group)

    # 构建时间键（Day + Tmstamp）
    data['time_key'] = data['Day'].astype(str) + '_' + data['Tmstamp'].astype(str)

    # 计算每组每个时间点的均值
    group_means = data.groupby(['group_id', 'time_key'])[cols_to_impute].mean().reset_index()
    group_means.rename(columns={'Patv': 'Patv_mean'}, inplace=True)

    # 合并回原始数据
    data = data.merge(group_means[['group_id', 'time_key', 'Patv_mean']], on=['group_id', 'time_key'], how='left')

    # 找出原本缺失的位置
    missing_mask = data['Patv'].isna()

    # 插补缺失值
    data.loc[missing_mask, 'Patv'] = data.loc[missing_mask, 'Patv_mean']

    # 删除辅助列
    data.drop(columns=['group_id', 'time_key', 'Patv_mean'], inplace=True)

    print("功率缺失值插补完成。")

    # Step 5: 打印所有被填补的数据详情（最多前100条）
    print("\n【插补记录】以下 Patv 缺失值已被填补（最多显示前100条）：")
    filled_entries = data[missing_mask].head(100)  # 只取前100条用于打印
    for idx, row in filled_entries.iterrows():
        turb_id = row['TurbID']
        day = row['Day']
        tmstamp = row['Tmstamp']
        patv_filled = row['Patv']
        print(f"→ TurbID={turb_id}, Day={day}, Tmstamp={tmstamp} → 已填补 Patv = {patv_filled:.1f}")

    # Step 6: 保存结果
    data.to_csv(output_file, index=False)
    print(f"\n功率处理完成，结果已保存到 {output_file}")



if __name__ == "__main__":
    # 此程序尽可能一步一步的运行

    # 1、清理机舱和叶片的异常缺失，并处理一下功率为负的变为0
    dir_data = "./sdwpf_kddcup/sdwpf_245days_v1.csv"
    dict= creat_dict_Ndir_Pab(dir_data=dir_data, site=134)

    # 2、处理周期性的风向和机舱方向数据，将相对风向变为绝对风向
    dir_data = "./sdwpf_kddcup/sdwpf_245days_v1_Ndir_pab_processed.csv"
    output_file = './sdwpf_kddcup/sdwpf_245days_v1_Ndir_pab_ababsolute_Wdir_processed.csv'

    # 加载数据
    df = pd.read_csv(dir_data)

    # 处理风向
    df = process_wind_direction(df)

    # 保存结果
    df.to_csv(output_file, index=False)
    print(f"处理完成，已保存至 {output_file}")

    # 3、处理风速、风向、内外温度的异常缺失进行聚类插补
    dir_data = './sdwpf_kddcup/sdwpf_245days_v1_Ndir_pab_ababsolute_Wdir_processed.csv'
    dir_site = "./sdwpf_kddcup/sdwpf_baidukddcup2022_turb_location.csv"
    output_file = './sdwpf_kddcup/No_power_clean_data_group_imputation.csv'

    process_all_turbines_by_group(dir_data, output_file, dir_site, num_groups=10)


    # 3、最后单独处理一下功率的数据
    input_file = './sdwpf_kddcup/No_power_clean_data_group_imputation.csv'
    output_file = './sdwpf_kddcup/Cleaned_data_with_power.csv'
    site_file = "./sdwpf_kddcup/sdwpf_baidukddcup2022_turb_location.csv"

    process_power_data_only(input_file, output_file, site_file, num_groups=10)
