import yaml
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model   import CNN
from runner  import Runner
from dataset import AudioDataset, ConcatDataset


parser = argparse.ArgumentParser()
parser.add_argument('--test'  , action='store_true', default=False)
parser.add_argument('--train' , action='store_true', default=False)
    
parser.add_argument('--seed'  , type=int, default=0)
parser.add_argument('--batch' , type=int, default=32)
parser.add_argument('--title' , type=str, default='Untitled')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--params', type=str, default=None)

parser.add_argument('--kfold_idx', type=int, default=0)
parser.add_argument('--test_fold', type=int, default=0)

parser.add_argument('--model_conf', type=str, default='../config/model/SeNet.yaml')
parser.add_argument('--hyper_conf', type=str, default='../config/hyper/hyper.yaml')
args = parser.parse_args()

with open(args.hyper_conf) as conf:
    vars(args).update(yaml.load(conf, Loader=yaml.Loader))

with open(args.model_conf) as conf:
    model_args = yaml.load(conf, Loader=yaml.Loader)


np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
device = torch.device(args.device)


print(f'{args.title}, seed {args.seed}', flush=True)

model = CNN(**model_args).to(device)
n_params = sum( p.numel() for p in model.parameters() )
print(f'[Model] -# params: {n_params}', flush=True)

if args.train:
    args.train_data_list = args.train_data_list.format(kfold_idx=args.kfold_idx, test_fold=args.test_fold)
    args.valid_data_list = args.valid_data_list.format(kfold_idx=args.kfold_idx, test_fold=args.test_fold)

    dataset = {
        'train': ConcatDataset(
            AudioDataset(args.source_data_dir, args.train_data_list, **args.dataset_args, phase='train_source'),
            AudioDataset(args.target_data_dir, args.adapt_data_list, **args.dataset_args, phase='train_target')
        ),
        'valid': ConcatDataset(
            AudioDataset(args.source_data_dir, args.valid_data_list, **args.dataset_args, phase='valid_source'),
            AudioDataset(args.target_data_dir, args.valid_data_list, **args.dataset_args, phase='valid_target')
        )
    }

    loader = {
        'train': DataLoader(dataset['train'], batch_size=args.batch, pin_memory=True, shuffle=True, drop_last=True),
        'valid': DataLoader(dataset['valid'], batch_size=args.batch, pin_memory=True)
    }

    criterion = {
        'cls': nn.CrossEntropyLoss().to(device),
        'dmn': nn.CrossEntropyLoss(torch.FloatTensor(args.weight) if args.weight is not None else None).to(device)
    }
    optimizer = getattr(optim, args.optim['name'])(model.parameters(), **args.optim['args'])

    print('Start Training ...', flush=True)
    runner = Runner(model, loader, device, criterion, optimizer, args=args)
    runner.train()

if args.test:
    args.test_data_list = args.test_data_list.format(kfold_idx=args.kfold_idx, test_fold=args.test_fold)

    dataset = {
        'test': ConcatDataset(
            AudioDataset(args.source_data_dir, args.test_data_list, **args.dataset_args, phase='test_source'),
            AudioDataset(args.target_data_dir, args.test_data_list, **args.dataset_args, phase='test_target')
        )
    }

    loader = {
        'test': DataLoader(dataset['test'], batch_size=args.batch, num_workers=4, pin_memory=True)
    }

    runner = Runner(model, loader, device, args=args)
    runner.test(args.params)