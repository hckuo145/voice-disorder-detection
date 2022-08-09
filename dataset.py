import numpy as np

import torch
import torchaudio
from torch.utils.data import Dataset

from typing import Dict


def wave_to_spectrum(wave, n_fft, hop_length=None, win_length=None, log_fn=None, norm=None):
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
    def __init__(self, path_dict: Dict, stft_args, frame_size, mode):
        super(AudioDataset, self).__init__()

        if mode == 'train':
            self.min_len = min([ len(paths) for paths in self.path_dict.values() ])

        self.mode       = mode
        self.path_dict  = path_dict
        self.stft_args  = stft_args
        self.frame_size = frame_size

        self.resample()

    def resample(self):
        self.paths, self.labels = [], []

        for label, paths in enumerate(self.path_dict.values()):
            if self.mode == 'train':
                paths = np.random.permutation(paths)[:self.min_len]

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
            offset = np.random.randint(len(wave) - self.frame_size + 1)
        sptrm = sptrm[:, offset:offset+self.frame_size]

        return sptrm, label

    def __len__(self):
        return len(self.paths)