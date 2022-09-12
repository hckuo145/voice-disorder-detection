import sys
sys.path.append('/home/hckuo/Project/voice-disorder-detection/')

import yaml
import numpy             as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
from torch.utils.data import DataLoader

from model   import CNN
from dataset import AudioDataset, ConcatDataset


device = torch.device('cuda:0')

source_data_dir = '../data/clean' 
target_data_dir = '../data/noisy'
test_data_list  = '../data/label/test_list_k11f2.txt'
# 0 145  44
# 1 153  145
# 2 83   24
# 3 55   130
# 4 194  195

stft_args = {
    'n_fft'     : 500,
    'hop_length': 250,
    'win_length': 500,
    'log_fn'    : 'log10',
    'norm'      : 'mean_std'
}
frame_size = 127

dataset = ConcatDataset(
    AudioDataset(source_data_dir, test_data_list, stft_args, frame_size, phase='test_source', device=device),
    AudioDataset(target_data_dir, test_data_list, stft_args, frame_size, phase='test_target', device=device)
)

loader = DataLoader(dataset, batch_size=len(dataset))


model_config = '../config/model/SepCNN.yaml'
model_params = '../exp/SepCNN_naive_k11f2/seed83/ckpt/best_valid_src_uar.pt'
with open(model_config) as conf:
    model_args = yaml.load(conf, Loader=yaml.Loader)

ref_model = CNN(**model_args)
ref_model.load_state_dict(torch.load(model_params, map_location=torch.device('cpu'))['model'])
ref_model.to(device)
ref_model.eval()

model_config = '../config/model/SepDACNN_a05.yaml'
model_params = '../exp/SepDACNN_a05_adapt_k11f2/seed24/ckpt/best_valid_avg_uar.pt'
with open(model_config) as conf:
    model_args = yaml.load(conf, Loader=yaml.Loader)

dat_model = CNN(**model_args)
dat_model.load_state_dict(torch.load(model_params, map_location=torch.device('cpu'))['model'])
dat_model.to(device)
dat_model.eval()


s = 20
c0 = '#2ca02c'
c1 = '#fc5a50'
# c1 = '#dc143c'
c2 = '#ff7f0e'
fig, ax = plt.subplots(2, 1, figsize=(5, 10))


with torch.no_grad():
    for source_data, target_data in loader:
        source_x, source_cls, source_dmn = source_data
        target_x, target_cls, target_dmn = target_data
        
        source_h = ref_model.return_hidden(source_x).cpu().numpy()
        target_h = ref_model.return_hidden(target_x).cpu().numpy()
        
        hiddens = np.concatenate([source_h, target_h], axis=0)
        
        source_cls, source_dmn = source_cls.cpu().numpy(), source_dmn.cpu().numpy()
        target_cls, target_dmn = target_cls.cpu().numpy(), target_dmn.cpu().numpy()
        idx = [
            [ np.where((source_cls == 0) * (source_dmn == 0)), np.where((source_cls == 1) * (source_dmn == 0)), np.where((source_cls == 2) * (source_dmn == 0)) ],
            [ np.where((target_cls == 0) * (target_dmn == 1)), np.where((source_cls == 1) * (target_dmn == 1)), np.where((source_cls == 2) * (target_dmn == 1)) ],
            [ np.where((target_cls == 0) * (target_dmn == 2)), np.where((source_cls == 1) * (target_dmn == 2)), np.where((source_cls == 2) * (target_dmn == 2)) ]
        ]

hidden_tsne = TSNE(n_components=2, perplexity=15, init='pca', n_iter=10000).fit_transform(hiddens)
hidden_tsne = (hidden_tsne - hidden_tsne.min()) / (hidden_tsne.max() - hidden_tsne.min())

source_tsne, target_tsne = np.split(hidden_tsne, 2)
ax[0].scatter(source_tsne[idx[0][0]][:, 0], source_tsne[idx[0][0]][:, 1], s=s, c=c0, marker='o')
ax[0].scatter(source_tsne[idx[0][1]][:, 0], source_tsne[idx[0][1]][:, 1], s=s, c=c0, marker='^')
ax[0].scatter(source_tsne[idx[0][2]][:, 0], source_tsne[idx[0][2]][:, 1], s=s, c=c0, marker='x')
ax[0].scatter(target_tsne[idx[1][0]][:, 0], target_tsne[idx[1][0]][:, 1], s=s, c=c1, marker='o')
ax[0].scatter(target_tsne[idx[1][1]][:, 0], target_tsne[idx[1][1]][:, 1], s=s, c=c1, marker='^')
ax[0].scatter(target_tsne[idx[1][2]][:, 0], target_tsne[idx[1][2]][:, 1], s=s, c=c1, marker='x')
ax[0].scatter(target_tsne[idx[2][0]][:, 0], target_tsne[idx[2][0]][:, 1], s=s, c=c2, marker='o')
ax[0].scatter(target_tsne[idx[2][1]][:, 0], target_tsne[idx[2][1]][:, 1], s=s, c=c2, marker='^')
ax[0].scatter(target_tsne[idx[2][2]][:, 0], target_tsne[idx[2][2]][:, 1], s=s, c=c2, marker='x')

ax[0].set_xlim([0, 1])
ax[0].set_ylim([0, 1])



with torch.no_grad():
    for source_data, target_data in loader:
        source_x, source_cls, source_dmn = source_data
        target_x, target_cls, target_dmn = target_data
        
        source_h = dat_model.return_hidden(source_x).cpu().numpy()
        target_h = dat_model.return_hidden(target_x).cpu().numpy()
        
        hiddens = np.concatenate([source_h, target_h], axis=0)
        
        source_cls, source_dmn = source_cls.cpu().numpy(), source_dmn.cpu().numpy()
        target_cls, target_dmn = target_cls.cpu().numpy(), target_dmn.cpu().numpy()
        idx = [
            [ np.where((source_cls == 0) * (source_dmn == 0)), np.where((source_cls == 1) * (source_dmn == 0)), np.where((source_cls == 2) * (source_dmn == 0)) ],
            [ np.where((target_cls == 0) * (target_dmn == 1)), np.where((source_cls == 1) * (target_dmn == 1)), np.where((source_cls == 2) * (target_dmn == 1)) ],
            [ np.where((target_cls == 0) * (target_dmn == 2)), np.where((source_cls == 1) * (target_dmn == 2)), np.where((source_cls == 2) * (target_dmn == 2)) ]
        ]

hidden_tsne = TSNE(n_components=2, perplexity=15, init='pca', n_iter=10000).fit_transform(hiddens)
hidden_tsne = (hidden_tsne - hidden_tsne.min()) / (hidden_tsne.max() - hidden_tsne.min())

source_tsne, target_tsne = np.split(hidden_tsne, 2)
sca0 = ax[1].scatter(source_tsne[idx[0][0]][:, 0], source_tsne[idx[0][0]][:, 1], s=s, c=c0, marker='o')
sca1 = ax[1].scatter(source_tsne[idx[0][1]][:, 0], source_tsne[idx[0][1]][:, 1], s=s, c=c0, marker='^')
sca2 = ax[1].scatter(source_tsne[idx[0][2]][:, 0], source_tsne[idx[0][2]][:, 1], s=s, c=c0, marker='x')
sca3 = ax[1].scatter(target_tsne[idx[1][0]][:, 0], target_tsne[idx[1][0]][:, 1], s=s, c=c1, marker='o')
sca4 = ax[1].scatter(target_tsne[idx[1][1]][:, 0], target_tsne[idx[1][1]][:, 1], s=s, c=c1, marker='^')
sca5 = ax[1].scatter(target_tsne[idx[1][2]][:, 0], target_tsne[idx[1][2]][:, 1], s=s, c=c1, marker='x')
sca6 = ax[1].scatter(target_tsne[idx[2][0]][:, 0], target_tsne[idx[2][0]][:, 1], s=s, c=c2, marker='o')
sca7 = ax[1].scatter(target_tsne[idx[2][1]][:, 0], target_tsne[idx[2][1]][:, 1], s=s, c=c2, marker='^')
sca8 = ax[1].scatter(target_tsne[idx[2][2]][:, 0], target_tsne[idx[2][2]][:, 1], s=s, c=c2, marker='x')

ax[1].set_xlim([0, 1])
ax[1].set_ylim([0, 1])

ax[0].set_ylabel('(a) SepConv')
ax[1].set_ylabel('(b) SepConv-dat')

fig.legend(
    [sca0, sca1, sca2, sca3, sca4, sca5, sca6, sca7, sca8], 
    ['clean\nnormal' , 'clean\nneoplasm' , 'clean\nstructural', \
     'A/C\nnormal'   , 'A/C\nneoplasm'   , 'A/C\nstructural'  , \
     'street\nnormal', 'street\nneoplasm', 'street\nstructural'], 
    loc='center right'
)
plt.savefig('tsne.png', bbox_inches='tight', dpi=600)