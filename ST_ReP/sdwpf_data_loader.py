import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class StandardScaler():
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, data):
        self.mean = data.mean()
        self.std = data.std()
        return self

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class SDWPF_Dataset(Dataset):
    def __init__(self, root_path='./SDWPF', flag='train',
                 seq_len=60, pred_len=6, data_path='SDWPF.npz'):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.flag = flag

        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.__read_data__(root_path, data_path)

    def __read_data__(self, root_path, data_path):
        self.scaler = StandardScaler()

        data_file = os.path.join(root_path, data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data']

        # 数据集划分
        train_ratio = 0.6
        test_ratio = 0.2
        num_train = int(data.shape[0] * train_ratio)
        num_test = int(data.shape[0] * test_ratio)
        num_vali = data.shape[0] - num_train - num_test

        border1s = [0, num_train - self.seq_len, data.shape[0] - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, data.shape[0]]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        data_slice = data[border1:border2]

        # 标准化
        self.scaler.fit(data[border1s[0]:border2s[0]])
        self.data_x = self.scaler.transform(data_slice)
        self.data_y = self.data_x.copy()

        # 简单的时间特征（一天中的时间点，周几）
        n_samples = data_slice.shape[0]
        t_of_d = (np.arange(n_samples) % 144).reshape(-1, 1)  # 144个10分钟/天
        d_of_w = ((np.arange(n_samples) // 144) % 7).reshape(-1, 1)  # 7天一周
        self.data_stamp = np.concatenate([t_of_d, d_of_w], axis=1).astype(np.int64)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]

        return (torch.FloatTensor(seq_x),
                torch.LongTensor(seq_x_mark),
                torch.FloatTensor(seq_y))

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


def get_sdwpf_dataloaders(seq_len=60, pred_len=6, batch_size=32):
    train_dataset = SDWPF_Dataset(flag='train', seq_len=seq_len, pred_len=pred_len)
    val_dataset = SDWPF_Dataset(flag='val', seq_len=seq_len, pred_len=pred_len)
    test_dataset = SDWPF_Dataset(flag='test', seq_len=seq_len, pred_len=pred_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.scaler