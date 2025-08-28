import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy


def compute_mi(x, y, bins='auto'):
    """基于熵的单变量MI计算"""
    # 自动确定分箱数（Freedman-Diaconis规则）
    if bins == 'auto':
        q75, q25 = np.percentile(y, [75, 25])
        iqr = q75 - q25
        h = 2 * iqr * (len(y) ** (-1 / 3))
        bins = int((y.max() - y.min()) / h)
        bins = max(5, min(bins, 50))  # 限制分箱范围

    # 计算联合分布
    hist_xy, _, _ = np.histogram2d(x, y, bins=bins)
    prob_xy = hist_xy / (np.sum(hist_xy) + 1e-10)  # 添加小常数防止除零

    # 计算边缘分布
    prob_x = np.sum(prob_xy, axis=1)
    prob_y = np.sum(prob_xy, axis=0)

    # 计算各项熵值
    H_x = entropy(prob_x)
    H_y = entropy(prob_y)
    H_xy = entropy(prob_xy.flatten())

    # 互信息计算
    mi = H_x + H_y - H_xy
    return max(0, mi)  # 确保非负


def joint_entropy(X, bins=10):
    """计算联合熵（支持多维输入）"""
    # 添加微小噪声防止分箱边界效应
    X_noised = X + np.random.normal(0, 1e-10, X.shape)
    counts = np.histogramdd(X_noised, bins=bins)[0]
    prob = counts / (np.sum(counts) + 1e-10)  # 避免除零
    return entropy(prob.flatten())


def compute_joint_mi(X, y, bins=10):
    """基于联合概率分布的多变量MI计算"""
    # 计算各项熵值
    H_X = joint_entropy(X, bins)
    H_Y = entropy(np.histogram(y, bins=bins)[0])
    H_XY = joint_entropy(np.column_stack((X, y)), bins)

    # 互信息计算
    mi = H_X + H_Y - H_XY
    return max(0, mi)  # 确保非负


def compute_multi_mi(df, features, target_col='Patv', bins='auto'):
    """多变量MI计算接口"""
    X = df[features].values
    y = df[target_col].values

    # 自动确定分箱数（Freedman-Diaconis规则）
    if bins == 'auto':
        q75, q25 = np.percentile(y, [75, 25])
        iqr = q75 - q25
        h = 2 * iqr * (len(y) ** (-1 / 3))
        bins = int((y.max() - y.min()) / h)
        bins = max(5, min(bins, 50))  # 限制分箱范围

    return compute_joint_mi(X, y, bins)


if __name__ == '__main__':
    # 数据加载与预处理
    file_path = '../sdwpf_kddcup/Cleaned_data_with_power.csv'  # 替换为你的实际路径
    df = pd.read_csv(file_path)
    df = df.drop(columns=['TurbID', 'Day', 'Tmstamp', 'Prtv'])
    df = df.ffill().bfill()

    # 单变量MI计算（现在使用基于熵的方法）
    single_features = ['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3','Patv']
    single_mis = {f: compute_mi(df[f], df['Patv']) for f in single_features}

    # 多变量MI计算（联合分布方法）
    multi_combinations = {
        "[Etmp, Itmp]": ['Etmp', 'Itmp'],
        "[Wspd,Etmp, Itmp]": ['Wspd','Etmp', 'Itmp'],
        "[Wspd,Etmp]": ['Wspd', 'Etmp'],
        "[Wspd, Itmp]": ['Wspd', 'Itmp'],
        "[Pab1, Pab2, Pab3]": ['Pab1', 'Pab2', 'Pab3'],
        "[Ndir, Pab1]": ['Ndir', 'Pab1'],
        "[Wdir, Ndir]": ['Wdir', 'Ndir'],
        "[Ndir, Pab1, Pab2, Pab3]": ['Ndir', 'Pab1', 'Pab2', 'Pab3'],
        "[Wdir, Ndir, Pab1]": ['Wdir', 'Ndir', 'Pab1']
    }
    multi_mis = {k: compute_multi_mi(df, v) for k, v in multi_combinations.items()}

    # 结果合并与展示
    all_mis = {**single_mis, **multi_mis}
    mis_df = pd.DataFrame(list(all_mis.items()), columns=['Feature Combination', 'Mutual Information'])
    mis_df = mis_df.sort_values(by='Mutual Information', ascending=False).reset_index(drop=True)

    print("\n--- 单变量 MI (基于熵) ---")
    for k, v in sorted(single_mis.items(), key=lambda x: x[1], reverse=True):
        print(f"{k}: {v:.4f}")

    print("\n--- 多变量组合 MI (联合分布法) ---")
    for k, v in sorted(multi_mis.items(), key=lambda x: x[1], reverse=True):
        print(f"{k}: {v:.4f}")

    # 可视化
    plt.figure(figsize=(12, 8))

    plt.rcParams.update({
        'font.size': 20,  # 20 默认 10 → 15（1.5倍）
        'axes.titlesize': 20,  # 默认 12 → 18
        'axes.labelsize': 20,  # 默认 10 → 15
        'xtick.labelsize': 20,  # 默认 10 → 15
        'ytick.labelsize': 20,  # 默认 10 → 15
        'legend.fontsize': 20,  # 默认 10 → 15
    })

    ax = sns.barplot(x='Mutual Information', y='Feature Combination', data=mis_df, palette='viridis')

    # 标记多变量组合
    for i, row in mis_df.iterrows():
        if '[' in row['Feature Combination']:
            ax.text(row['Mutual Information'] + 0.02, i, 'Joint', ha='left', va='center', color='red')

    plt.title('Mutual Information (Entropy-based Method)')
    plt.tight_layout()

    plt.savefig('mi_analysis.png')  # 文件名
    plt.show()