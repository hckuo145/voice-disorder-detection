import os
import numpy as np
from collections import defaultdict

import torch
import torchaudio
from torch.utils.data import Dataset

from typing import Dict

LABEL_DICT  = { 'normal': 0, 'neoplasm': 1, 'phonotrauma': 2 }
DOMAIN_DICT = { 'clean': 0, 'airconditioner': 1, 'street':2, 'babycry':3 }


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
    def __init__(self, data_dir, data_list, stft_args, frame_size, phase):
        super(AudioDataset, self).__init__()

        with open(data_list) as infile:
            lines = infile.readlines()
        
        self.info = {}
        self.name_dict = defaultdict(list)
        for line in lines:
            name, domain = line.strip().split()
            label = name.split('/')[0]

            self.name_dict[label] += [ name ]
            self.info[name] = {
                'path'  : os.path.join(data_dir, name),
                'label' : LABEL_DICT[label],
                'domain': DOMAIN_DICT[domain] if 'target' in phase else 0
            }

        if 'train' in phase:
            self.min_len = min([ len(names) for names in self.name_dict.values() ])

        self.phase      = phase
        self.stft_args  = stft_args
        self.frame_size = frame_size

        self.resample()

    def resample(self):
        self.names = []

        for _, names in enumerate(self.name_dict.values()):
            names
            if 'train' in self.phase:
                names = list(np.random.permutation(names)[:self.min_len])

            self.names += names

    def __getitem__(self, idx):
        name = self.names[idx]

        wave = torchaudio.load(self.info[name]['path'])[0][0]
        sptrm, _ = wave_to_spectrum(wave, **self.stft_args)

        offset = 0
        if 'train' in self.phase:
            offset = np.random.randint(sptrm.size(1) - self.frame_size + 1)
        sptrm = sptrm[:, offset:offset+self.frame_size]

        label  = torch.tensor(self.info[name]['label'], dtype=torch.int64)
        domain = torch.tensor(self.info[name]['domain'], dtype=torch.int64)

        return sptrm, label, domain

    def __len__(self):
        return len(self.names)


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
    
    def __getitem__(self, idx):
        return tuple(dataset[idx % len(dataset)] for dataset in self.datasets)

    def __len__(self):
        return max(len(dataset) for dataset in self.datasets)

    def resample(self):
        for dataset in self.datasets:
            dataset.resample()


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    source_data_dir = 'data/clean'
    target_data_dir = 'data/airconditioner'
    train_data_list = 'data/label/train_list_t0f0.txt'
    adapt_data_list = 'data/label/adapt_list.txt'

    stft_args = {
        'n_fft'     : 500,
        'hop_length': 250,
        'win_length': 500,
        'log_fn'    : 'log10'
    }
    frame_size = 127

    dataset = ConcatDataset(
        AudioDataset(source_data_dir, train_data_list, stft_args, frame_size, 'train_source'),
        AudioDataset(target_data_dir, adapt_data_list, stft_args, frame_size, 'train_target')
    )
    loader  = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    for i, (source_batch, target_batch) in enumerate(loader):
        source_x, source_cls, source_dmn = source_batch
        target_x, target_cls, target_dmn = target_batch
        print(i, source_x.size(), source_cls.size(), source_dmn.size())