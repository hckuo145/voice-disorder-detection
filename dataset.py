import os
import numpy as np
from collections import defaultdict

import torch
import torchaudio
from torch.utils.data import Dataset

from typing import Dict


def wave_to_spectrum(wave, n_fft, hop_length=None, win_length=None, log_fn=None):
    window = torch.hamming_window(win_length).to(wave.device)
    stft = torch.stft(wave, n_fft, hop_length, win_length, window, return_complex=True)
    
    sptrm = torch.abs(stft)
    phase = torch.angle(stft)     
    
    if log_fn == 'log10':
        sptrm = torch.log10(sptrm + 1e-12)
    elif log_fn == 'log1p':
        sptrm = torch.log1p(sptrm)

    return sptrm, phase


class AudioDataset(Dataset):
    def __init__(self, data_dir, data_list, stft_args, frame_size, mode):
        super(AudioDataset, self).__init__()


        with open(data_list) as infile:
            lines = infile.readlines()

        self.path_dict = defaultdict(list)
        for line in lines:
            cls, name = line.strip().split('/')
            self.path_dict[cls] += [ os.path.join(data_dir, cls, name) ] 

        if mode == 'train':
            self.min_len = min([ len(paths) for paths in self.path_dict.values() ])

        self.mode       = mode
        self.stft_args  = stft_args
        self.frame_size = frame_size

        self.resample()

    def resample(self):
        self.paths, self.labels = [], []

        for label, paths in enumerate(self.path_dict.values()):
            if self.mode == 'train':
                paths = list(np.random.permutation(paths)[:self.min_len])
                
            self.paths  += paths
            self.labels += [label] * len(paths)
        
        self.labels = torch.tensor(self.labels, dtype=torch.int64)

    def __getitem__(self, idx):
        path  = self.paths[idx]
        label = self.labels[idx]
        
        wave = torchaudio.load(path)[0][0]
        sptrm, _ = wave_to_spectrum(wave, **self.stft_args)

        offset = 0
        if self.mode == 'train':
            offset = np.random.randint(sptrm.size(1) - self.frame_size + 1)
        sptrm = sptrm[:, offset:offset+self.frame_size]

        return sptrm, label

    def __len__(self):
        return len(self.paths)



if __name__ == '__main__':
    from torch.utils.data import DataLoader

    data_dir  = 'data/clean'
    data_list = 'data/label/train_list_s0f0.txt'
    stft_args = {
        'n_fft'     : 500,
        'hop_length': 250,
        'win_length': 500,
        'log_fn'    : 'log10'
    }

    dataset = AudioDataset(data_dir, data_list, stft_args, 127, 'train')
    loader  = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    for i, (batch_x, batch_y) in enumerate(loader):
        print(i, batch_x.size())