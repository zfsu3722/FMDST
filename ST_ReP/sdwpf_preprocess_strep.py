import pandas as pd
import numpy as np
import os


def create_sdwpf_dataset(csv_path='./sdwpf_kddcup/sdwpf_245days_v1.csv', save_dir='./SDWPF'):
    """从CSV文件创建SDWPF数据集"""
    print(f"正在处理数据...")

    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 异常值处理
    def handle_exceptions(df):
        mask = pd.Series(index=df.index, data=0)
        mask |= (df['Wspd'] > 2.5) & (df['Patv'] <= 0)
        mask |= (df['Pab1'] > 89) | (df['Pab2'] > 89) | (df['Pab3'] > 89)
        mask |= df['Patv'].isna()
        mask |= (df['Patv'] < 0)
        mask |= (df['Ndir'] > 720) | (df['Ndir'] < -720)
        mask |= (df['Wdir'] > 180) | (df['Wdir'] < -180)
        df['Patv_clear'] = df['Patv'].where(~mask, 0)
        return mask

    handle_exceptions(df)

    # 提取功率数据
    df_ready = df[['TurbID', 'Patv_clear']]
    df_list = df_ready.groupby(['TurbID'])
    data = []

    for _, v in df_list:
        data.append(v['Patv_clear'].values)

    # 堆叠数据
    data = np.stack(data, 1)
    data = np.expand_dims(data, axis=-1)

    # 保存数据
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/SDWPF.npz'
    np.savez_compressed(save_path, data=data)

    print(f"数据已保存到 {save_path}")
    print(f"数据形状: {data.shape}")
    return save_path

if __name__ == "__main__":
    create_sdwpf_dataset()