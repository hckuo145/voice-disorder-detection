import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import hamming


def extract_spectrum(path, sr=16000, fft_size=500):
    hop_size = fft_size // 2
    win_size = fft_size

    sig, _ = librosa.load(path, sr=sr)
    if (len(sig) - fft_size) % hop_size != 0:
        sig = librosa.util.fix_length(sig, len(sig) + hop_size)

    stft  = librosa.stft(sig, fft_size, hop_size, win_size, window=hamming, center=False)
    sptrm = np.abs(stft)


    sptrm = np.log10(sptrm + 1.5e-2)
    # sptrm = np.log1p(sptrm)


    # _std  = np.std(sptrm, axis=0, keepdims=True)
    # _mean = np.mean(sptrm, axis=0, keepdims=True)
    # sptrm = (sptrm - _mean) / _std

    # _min = np.amin(sptrm, axis=0, keepdims=True)
    # _max = np.amax(sptrm, axis=0, keepdims=True)
    # sptrm = (sptrm - _min) / (_max - _min)

    # _min = np.amin(sptrm)
    # _max = np.amax(sptrm)
    # sptrm = (sptrm - _min) / (_max - _min)

    return sptrm


src_dir = '../data/clean'
tgt_dir = '../data/noisy'

cls_order = ['Normal', 'Neoplasm', 'Structural']

filenames = [
    'normal/023.wav', 'neoplasm/001.wav', 'phonotrauma/022.wav'
]

# neoplasm: 001, 100
# structural: 155, 14

# normal      : A/C 023.wav street
# neoplasm    : A/C         street 040.wav
# phonotrauma : A/C         street 032.wav


plt.rcParams.update({'font.size': 12})

fig = plt.figure(figsize=(24, 16))
outer = gridspec.GridSpec(4, 3, wspace=0.2, hspace=0.3)

for i, filename in enumerate(filenames):
    scale = 16000

    ax = fig.add_subplot(outer[0, i])
    ax.set_title(f'{cls_order[i]}\n\n', fontweight='bold')
    ax.axis('off')


    inner = gridspec.GridSpecFromSubplotSpec(1, 2, \
        subplot_spec=outer[0, i], wspace=0.2, hspace=0.2)
        
    filepath = os.path.join(src_dir, filename)
    wave, _, = librosa.load(filepath, sr=16000)
    time = np.arange(len(wave)) / scale
    
    ax = fig.add_subplot(inner[0, 0])
    ax.plot(time, wave, linewidth=1)
    ax.set_xlim([0, len(wave) / scale])
    ax.set_ylim([-1, 1])
    ax.set_title('Waveform')
    ax.set_xlabel('Time (sec)')
    if i == 0:
        ax.set_ylabel(r'$\bf{Clean}$' + '\n\nAmplitude')
    else:
        ax.set_ylabel('Amplitude')

    sptrm = extract_spectrum(filepath)
    ax = fig.add_subplot(inner[0, 1])
    ax.imshow(sptrm, aspect='auto', cmap='hot', origin='lower', extent=[0, len(wave)/scale, 0, 8])
    ax.set_title('Spectrum')
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Frequency (kHz)')



    inner = gridspec.GridSpecFromSubplotSpec(1, 2, \
        subplot_spec=outer[1, i], wspace=0.2, hspace=0.2)
        
    filepath = os.path.join(tgt_dir, filename)
    wave, _, = librosa.load(filepath, sr=16000)
    time = np.arange(len(wave)) / scale
    
    ax = fig.add_subplot(inner[0, 0])
    ax.plot(time, wave, linewidth=1)
    ax.set_xlim([0, len(wave) / scale])
    ax.set_ylim([-1, 1])
    ax.set_title('Waveform')
    ax.set_xlabel('Time (sec)')
    if i == 0:
        ax.set_ylabel(r'$\bf{A/C}$' + '\n\nAmplitude')
    else:
        ax.set_ylabel('Amplitude')

    sptrm = extract_spectrum(filepath)
    ax = fig.add_subplot(inner[0, 1])
    ax.imshow(sptrm, aspect='auto', cmap='hot', origin='lower', extent=[0, len(wave)/scale, 0, 8])
    ax.set_title('Spectrum')
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Frequency (kHz)')

filenames = [
    'normal/055.wav', 'neoplasm/040.wav', 'phonotrauma/032.wav'
]
# normal 041, 

for i, filename in enumerate(filenames):
    scale = 16000

    ax = fig.add_subplot(outer[2, i])
    ax.set_title(f'{cls_order[i]}\n\n', fontweight='bold')
    ax.axis('off')


    inner = gridspec.GridSpecFromSubplotSpec(1, 2, \
        subplot_spec=outer[2, i], wspace=0.2, hspace=0.2)
        
    filepath = os.path.join(src_dir, filename)
    wave, _, = librosa.load(filepath, sr=16000)
    time = np.arange(len(wave)) / scale
    
    ax = fig.add_subplot(inner[0, 0])
    ax.plot(time, wave, linewidth=1)
    ax.set_xlim([0, len(wave) / scale])
    ax.set_ylim([-1, 1])
    ax.set_title('Waveform')
    ax.set_xlabel('Time (sec)')
    if i == 0:
        ax.set_ylabel(r'$\bf{Clean}$' + '\n\nAmplitude')
    else:
        ax.set_ylabel('Amplitude')

    sptrm = extract_spectrum(filepath)
    ax = fig.add_subplot(inner[0, 1])
    ax.imshow(sptrm, aspect='auto', cmap='hot', origin='lower', extent=[0, len(wave)/scale, 0, 8])
    ax.set_title('Spectrum')
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Frequency (kHz)')



    inner = gridspec.GridSpecFromSubplotSpec(1, 2, \
        subplot_spec=outer[3, i], wspace=0.2, hspace=0.2)
        
    filepath = os.path.join(tgt_dir, filename)
    wave, _, = librosa.load(filepath, sr=16000)
    time = np.arange(len(wave)) / scale
    
    ax = fig.add_subplot(inner[0, 0])
    ax.plot(time, wave, linewidth=1)
    ax.set_xlim([0, len(wave) / scale])
    ax.set_ylim([-1, 1])
    ax.set_title('Waveform')
    ax.set_xlabel('Time (sec)')
    if i == 0:
        ax.set_ylabel(r'$\bf{Street}$' + '\n\nAmplitude')
    else:
        ax.set_ylabel('Amplitude')

    sptrm = extract_spectrum(filepath)
    ax = fig.add_subplot(inner[0, 1])
    ax.imshow(sptrm, aspect='auto', cmap='hot', origin='lower', extent=[0, len(wave)/scale, 0, 8])
    ax.set_title('Spectrum')
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Frequency (kHz)')


outer.tight_layout(fig)
plt.savefig('input.png', dpi=600)