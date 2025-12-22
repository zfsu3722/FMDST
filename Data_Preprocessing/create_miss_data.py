import pandas as pd
import numpy as np


def create_missing_data(
        dir_data,
        selected_sites=None,
        missing_ratio=0.1,
        target_cols=['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1','Patv'],
        output_path='data_with_missing.csv',
        random_seed=None
):
    """
    读取风力发电数据，对指定站点的指定列按比例引入随机缺失，并保存结果。

    参数：
    - dir_data (str): 原始 CSV 文件路径
    - selected_sites (int, list, or None):
        - int: 随机选 N 个站点
        - list: 指定站点 ID 列表
        - None: 默认随机选 1 个站点
    - missing_ratio (float): 缺失比例，如 0.1 表示 10%
    - target_cols (list): 要引入缺失的列名
    - output_path (str): 输出文件路径
    - random_seed (int or None): 随机种子，用于复现

    返回：
    - df_modified (pd.DataFrame): 处理后的数据
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # 1. 读取数据
    df = pd.read_csv(dir_data)

    # 2. 清洗 Patv 和 Prtv（你原有的逻辑）
    df.loc[df['Prtv'] < 0, 'Prtv'] = 0
    df.loc[df['Patv'] < 0, 'Patv'] = 0

    # 确保 TurbID 是整数
    df['TurbID'] = df['TurbID'].astype(int)

    all_sites = df['TurbID'].unique()

    # 3. 确定要处理的站点
    if selected_sites is None:
        selected_sites = np.random.choice(all_sites, size=1, replace=False).tolist()
    elif isinstance(selected_sites, int):
        if selected_sites > len(all_sites):
            raise ValueError(f"请求的站点数 {selected_sites} 超过总站点数 {len(all_sites)}")
        selected_sites = np.random.choice(all_sites, size=selected_sites, replace=False).tolist()
    elif isinstance(selected_sites, list):
        invalid = set(selected_sites) - set(all_sites)
        if invalid:
            raise ValueError(f"以下站点不存在于数据中: {invalid}")
    else:
        raise TypeError("`selected_sites` 必须是 int、list 或 None")

    print(f"✅ 将在以下站点引入缺失: {sorted(selected_sites)}")
    print(f"🎯 目标列: {target_cols}")
    print(f"📉 缺失比例: {missing_ratio:.1%}")

    # 4. 创建副本并引入缺失
    df_modified = df.copy()

    for site in selected_sites:
        site_mask = df_modified['TurbID'] == site
        site_indices = df_modified[site_mask].index

        for col in target_cols:
            if col not in df_modified.columns:
                print(f"⚠️ 警告：列 '{col}' 不存在，跳过。")
                continue

            n_total = len(site_indices)
            n_missing = int(np.round(n_total * missing_ratio))

            # 至少缺失一个（如果比例太小但 >0）
            if n_missing == 0 and missing_ratio > 0:
                n_missing = 1

            if n_missing > 0:
                missing_idx = np.random.choice(site_indices, size=n_missing, replace=False)
                df_modified.loc[missing_idx, col] = np.nan

    # 5. 保存结果
    df_modified.to_csv(output_path, index=False)
    print(f"💾 已保存处理后的数据到: {output_path}")

    return df_modified

if __name__ == '__main__':
    # 示例：从 '指定文件' 中随机选 3 个站点，在指定列引入 20% 缺失
    df_result = create_missing_data(
        dir_data='../sdwpf_kddcup/sdwpf_245days_v1.csv',
        selected_sites=20,
        missing_ratio=0.4,
        target_cols=['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1','Patv'],
        output_path='../sdwpf_kddcup/original_data_with_missing.csv',
        random_seed=2025
    )