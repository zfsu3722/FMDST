from Newest_Ablation_model.LSTM_Transformer import LSTM_Transformer
from Newest_Ablation_model.TCN_Transformer import TCN_Transformer
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

print("当前工作目录:", os.getcwd())
print("脚本所在目录:", CURRENT_DIR)
print("项目根目录:", PROJECT_DIR)

# =========================================================
# 1. 配置参数
# =========================================================
save_directory_data = os.path.join(PROJECT_DIR, 'clean_power_his60_future24_7features_diffblock')
model_save_dir = os.path.join(PROJECT_DIR, 'model_save')

batch_size = 64
input_dim = 7
output_dim = 12
num_workers = 0

# 模型文件名
lstm_model_file = 'Power_Pred_LSTM_Transformer_his60_Pred12_134site_7feature_best_model.pth'
tcn_model_file = 'Power_Pred_TCN_Transformer_his60_Pred12_134site_7feature_best_model.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Selected device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

# 采样间隔
# 如果每个时间步是 1 小时，就写 1.0
# 如果每个时间步是 15 分钟，就写 0.25
dt = 1.0

# 低/高频阈值，可一次看多个，便于验证结论稳不稳
freq_thresholds = [0.05, 0.10, 0.15]

# 低频局部放大上限
zoom_freq_max = 0.15


# =========================================================
# 2. 加载数据
# =========================================================
def load_test_loader(save_directory_data, batch_size, num_workers=0):
    if not os.path.exists(save_directory_data):
        raise FileNotFoundError(f"The directory {save_directory_data} does not exist.")

    dataset_path = os.path.join(save_directory_data, 'test_dataset.pt')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Test dataset not found: {dataset_path}")

    # 仅你自己的本地可信文件时这样用；如果来源不可信要谨慎
    test_data = torch.load(dataset_path, map_location='cpu')
    test_dataset = TensorDataset(test_data[0], test_data[1])
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    print("Test X shape:", test_data[0].shape)
    print("Test Y shape:", test_data[1].shape)
    return test_loader


# =========================================================
# 3. 加载模型
# =========================================================
def load_model(model_class, model_path, input_dim, output_dim, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = model_class(input_dim=input_dim, output_dim=output_dim).to(device)

    state_dict = torch.load(model_path, map_location=device)

    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)

    model.eval()
    print(f"Loaded model: {model_path}")
    return model


# =========================================================
# 4. 低内存流式 FFT 统计
# 只累计平均 PSD / 平均 Ratio，不保留全量频谱
# =========================================================
def compute_psd_streaming(model, data_loader, device, dt=1.0, remove_mean=True):
    model.eval()

    sum_psd = None
    sum_ratio = None
    total_count = 0
    freqs = None
    feature_shape_example = None

    with torch.no_grad():
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(device)

            # 期望: [B, N, T, C]
            feat = model.extract_features(batch_x)

            if feat.dim() != 4:
                raise ValueError(
                    f"model.extract_features(batch_x) must return [B, N, T, C], "
                    f"but got shape {tuple(feat.shape)}"
                )

            if feature_shape_example is None:
                feature_shape_example = tuple(feat.shape)

            feat = feat.detach().cpu().numpy().astype(np.float32)

            # [B, N, T, C] -> [B, N, C, T]
            x = np.transpose(feat, (0, 1, 3, 2))

            if remove_mean:
                x = x - x.mean(axis=-1, keepdims=True)

            # FFT along temporal axis
            fft_result = np.fft.rfft(x, axis=-1)
            psd = np.abs(fft_result) ** 2  # [B, N, C, F]

            if freqs is None:
                T = feat.shape[2]
                freqs = np.fft.rfftfreq(T, d=dt)

            ratio = psd / (psd.sum(axis=-1, keepdims=True) + 1e-12)

            # 对 B,N,C 三维求和
            batch_sum_psd = psd.sum(axis=(0, 1, 2))
            batch_sum_ratio = ratio.sum(axis=(0, 1, 2))
            batch_count = psd.shape[0] * psd.shape[1] * psd.shape[2]

            if sum_psd is None:
                sum_psd = batch_sum_psd
                sum_ratio = batch_sum_ratio
            else:
                sum_psd += batch_sum_psd
                sum_ratio += batch_sum_ratio

            total_count += batch_count

            del feat, x, fft_result, psd, ratio

    if total_count == 0:
        raise RuntimeError("No features were processed. Please check the test loader.")

    avg_psd = sum_psd / total_count
    avg_ratio = sum_ratio / total_count

    return freqs, avg_psd, avg_ratio, feature_shape_example


# =========================================================
# 5. 统计指标
# =========================================================
def low_high_energy_summary(freqs, avg_ratio, threshold=0.1):
    low_mask = freqs <= threshold
    high_mask = freqs > threshold

    low_ratio = float(np.sum(avg_ratio[low_mask]))
    high_ratio = float(np.sum(avg_ratio[high_mask]))
    low_high_ratio = float(low_ratio / (high_ratio + 1e-12))

    return {
        'threshold': float(threshold),
        'low_ratio': low_ratio,
        'high_ratio': high_ratio,
        'low_high_ratio': low_high_ratio
    }


def spectral_centroid(freqs, avg_ratio):
    return float(np.sum(freqs * avg_ratio) / (np.sum(avg_ratio) + 1e-12))


def spectral_entropy(avg_ratio):
    p = np.clip(avg_ratio, 1e-12, None)
    return float(-np.sum(p * np.log(p)))


# =========================================================
# 6. 画图
# =========================================================
def plot_ratio_comparison(freqs1, ratio1, freqs2, ratio2,
                          label1='LSTM', label2='TCN',
                          save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(freqs1, ratio1, label=f'{label1} energy ratio')
    plt.plot(freqs2, ratio2, label=f'{label2} energy ratio')
    plt.xlabel('Frequency')
    plt.ylabel('Energy Ratio')
    plt.title('Temporal Spectral Energy Ratio Comparison')
    plt.legend()
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_cumulative_ratio(freqs1, ratio1, freqs2, ratio2,
                          label1='LSTM', label2='TCN',
                          save_path=None):
    cum1 = np.cumsum(ratio1)
    cum2 = np.cumsum(ratio2)

    plt.figure(figsize=(10, 6))
    plt.plot(freqs1, cum1, label=f'{label1} cumulative ratio')
    plt.plot(freqs2, cum2, label=f'{label2} cumulative ratio')
    plt.xlabel('Frequency')
    plt.ylabel('Cumulative Energy Ratio')
    plt.title('Cumulative Temporal Energy Ratio')
    plt.legend()
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_ratio_difference(freqs, ratio_lstm, ratio_tcn, save_path=None):
    diff = ratio_tcn - ratio_lstm
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, diff, label='TCN - LSTM')
    plt.axhline(0.0, linestyle='--')
    plt.xlabel('Frequency')
    plt.ylabel('Energy Ratio Difference')
    plt.title('Temporal Spectral Ratio Difference (TCN - LSTM)')
    plt.legend()
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_ratio_zoom(freqs1, ratio1, freqs2, ratio2,
                    freq_max=0.15,
                    label1='LSTM', label2='TCN',
                    save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(freqs1, ratio1, label=f'{label1} energy ratio')
    plt.plot(freqs2, ratio2, label=f'{label2} energy ratio')
    plt.xlim(0.0, freq_max)
    plt.xlabel('Frequency')
    plt.ylabel('Energy Ratio')
    plt.title(f'Temporal Spectral Energy Ratio (Zoomed, f <= {freq_max})')
    plt.legend()
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# =========================================================
# 7. 打印汇总结果
# =========================================================
def print_summary_block(freqs_lstm, avg_ratio_lstm, freqs_tcn, avg_ratio_tcn, thresholds):
    print("\n========== Temporal Spectral Statistics ==========")
    print(f"LSTM spectral centroid: {spectral_centroid(freqs_lstm, avg_ratio_lstm):.6f}")
    print(f"TCN  spectral centroid: {spectral_centroid(freqs_tcn, avg_ratio_tcn):.6f}")
    print(f"LSTM spectral entropy : {spectral_entropy(avg_ratio_lstm):.6f}")
    print(f"TCN  spectral entropy : {spectral_entropy(avg_ratio_tcn):.6f}")

    for th in thresholds:
        lstm_stats = low_high_energy_summary(freqs_lstm, avg_ratio_lstm, threshold=th)
        tcn_stats = low_high_energy_summary(freqs_tcn, avg_ratio_tcn, threshold=th)

        print(f"\n===== Low / High Frequency Energy Summary (threshold = {th:.2f}) =====")
        print(f"LSTM low-frequency ratio  (f <= {th:.2f}): {lstm_stats['low_ratio']:.6f}")
        print(f"LSTM high-frequency ratio (f >  {th:.2f}): {lstm_stats['high_ratio']:.6f}")
        print(f"TCN  low-frequency ratio  (f <= {th:.2f}): {tcn_stats['low_ratio']:.6f}")
        print(f"TCN  high-frequency ratio (f >  {th:.2f}): {tcn_stats['high_ratio']:.6f}")

        print(f"LSTM low/high ratio: {lstm_stats['low_high_ratio']:.6f}")
        print(f"TCN  low/high ratio: {tcn_stats['low_high_ratio']:.6f}")

        print(f"Low-frequency difference  (LSTM - TCN): {lstm_stats['low_ratio'] - tcn_stats['low_ratio']:.6f}")
        print(f"High-frequency difference (LSTM - TCN): {lstm_stats['high_ratio'] - tcn_stats['high_ratio']:.6f}")


# =========================================================
# 8. 保存数值结果
# =========================================================
def save_summary_npz(save_path,
                     freqs_lstm, avg_psd_lstm, avg_ratio_lstm,
                     freqs_tcn, avg_psd_tcn, avg_ratio_tcn,
                     lstm_feature_shape, tcn_feature_shape,
                     thresholds):
    save_dict = {
        'freqs_lstm': freqs_lstm,
        'avg_psd_lstm': avg_psd_lstm,
        'avg_ratio_lstm': avg_ratio_lstm,
        'freqs_tcn': freqs_tcn,
        'avg_psd_tcn': avg_psd_tcn,
        'avg_ratio_tcn': avg_ratio_tcn,
        'lstm_feature_shape': np.array(lstm_feature_shape if lstm_feature_shape is not None else []),
        'tcn_feature_shape': np.array(tcn_feature_shape if tcn_feature_shape is not None else []),
    }

    for th in thresholds:
        lstm_stats = low_high_energy_summary(freqs_lstm, avg_ratio_lstm, threshold=th)
        tcn_stats = low_high_energy_summary(freqs_tcn, avg_ratio_tcn, threshold=th)

        key_suffix = str(th).replace('.', '_')
        save_dict[f'lstm_low_ratio_{key_suffix}'] = np.array(lstm_stats['low_ratio'])
        save_dict[f'lstm_high_ratio_{key_suffix}'] = np.array(lstm_stats['high_ratio'])
        save_dict[f'lstm_low_high_ratio_{key_suffix}'] = np.array(lstm_stats['low_high_ratio'])

        save_dict[f'tcn_low_ratio_{key_suffix}'] = np.array(tcn_stats['low_ratio'])
        save_dict[f'tcn_high_ratio_{key_suffix}'] = np.array(tcn_stats['high_ratio'])
        save_dict[f'tcn_low_high_ratio_{key_suffix}'] = np.array(tcn_stats['low_high_ratio'])

    np.savez(save_path, **save_dict)


# =========================================================
# 9. 主程序
# =========================================================
def main():
    test_loader = load_test_loader(save_directory_data, batch_size, num_workers)

    lstm_model_path = os.path.join(model_save_dir, lstm_model_file)
    tcn_model_path = os.path.join(model_save_dir, tcn_model_file)

    lstm_model = load_model(LSTM_Transformer, lstm_model_path, input_dim, output_dim, device)
    tcn_model = load_model(TCN_Transformer, tcn_model_path, input_dim, output_dim, device)

    print("\nComputing LSTM temporal FFT statistics...")
    freqs_lstm, avg_psd_lstm, avg_ratio_lstm, lstm_feature_shape = compute_psd_streaming(
        lstm_model, test_loader, device, dt=dt, remove_mean=True
    )

    print("Computing TCN temporal FFT statistics...")
    freqs_tcn, avg_psd_tcn, avg_ratio_tcn, tcn_feature_shape = compute_psd_streaming(
        tcn_model, test_loader, device, dt=dt, remove_mean=True
    )

    print("LSTM feature example shape:", lstm_feature_shape)
    print("TCN  feature example shape:", tcn_feature_shape)

    # 输出统计量
    print_summary_block(
        freqs_lstm, avg_ratio_lstm,
        freqs_tcn, avg_ratio_tcn,
        thresholds=freq_thresholds
    )

    # 画图：只保留更有解释价值的图
    plot_ratio_comparison(
        freqs_lstm, avg_ratio_lstm,
        freqs_tcn, avg_ratio_tcn,
        label1='LSTM', label2='TCN',
        save_path='temporal_ratio_comparison.png'
    )

    plot_cumulative_ratio(
        freqs_lstm, avg_ratio_lstm,
        freqs_tcn, avg_ratio_tcn,
        label1='LSTM', label2='TCN',
        save_path='temporal_cumulative_ratio.png'
    )

    plot_ratio_difference(
        freqs_lstm, avg_ratio_lstm, avg_ratio_tcn,
        save_path='temporal_ratio_difference_tcn_minus_lstm.png'
    )

    plot_ratio_zoom(
        freqs_lstm, avg_ratio_lstm,
        freqs_tcn, avg_ratio_tcn,
        freq_max=zoom_freq_max,
        label1='LSTM', label2='TCN',
        save_path='temporal_ratio_zoom.png'
    )

    # 保存数值结果
    save_summary_npz(
        save_path='temporal_fft_ratio_summary.npz',
        freqs_lstm=freqs_lstm,
        avg_psd_lstm=avg_psd_lstm,
        avg_ratio_lstm=avg_ratio_lstm,
        freqs_tcn=freqs_tcn,
        avg_psd_tcn=avg_psd_tcn,
        avg_ratio_tcn=avg_ratio_tcn,
        lstm_feature_shape=lstm_feature_shape,
        tcn_feature_shape=tcn_feature_shape,
        thresholds=freq_thresholds
    )

    print("\nDone.")
    print("Saved figures:")
    print(" - temporal_ratio_comparison.png")
    print(" - temporal_cumulative_ratio.png")
    print(" - temporal_ratio_difference_tcn_minus_lstm.png")
    print(" - temporal_ratio_zoom.png")
    print("Saved data:")
    print(" - temporal_fft_ratio_summary.npz")


if __name__ == '__main__':
    main()