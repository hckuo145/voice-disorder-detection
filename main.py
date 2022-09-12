import yaml
import time
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
parser.add_argument('--test'    , action='store_true', default=False)
parser.add_argument('--train'   , action='store_true', default=False)
parser.add_argument('--finetune', action='store_true', default=False)
    
parser.add_argument('--seed'   , type=int, default=0)
parser.add_argument('--batch'  , type=int, default=32)
parser.add_argument('--title'  , type=str, default='Untitled')
parser.add_argument('--device' , type=str, default='cuda')
parser.add_argument('--params' , type=str, default=None)
parser.add_argument('--logfile', type=str, default=None)

parser.add_argument('--kfold_idx', type=int, default=0)
parser.add_argument('--test_fold', type=int, default=0)

parser.add_argument('--model_conf', type=str, default='../config/model/SeNet.yaml')
parser.add_argument('--hyper_conf', type=str, default='../config/hyper/hyper.yaml')
args = parser.parse_args()

with open(args.hyper_conf) as conf:
    vars(args).update(yaml.load(conf, Loader=yaml.Loader))

with open(args.model_conf) as conf:
    model_args = yaml.load(conf, Loader=yaml.Loader)


torch.backends.cudnn.deterministic = True
device = torch.device(args.device)


if args.train:
    t1 = time.time()
    print('Preprocess Data ... ', end='', flush=True)
    
    args.train_data_list = args.train_data_list.format(kfold_idx=args.kfold_idx, test_fold=args.test_fold)
    args.valid_data_list = args.valid_data_list.format(kfold_idx=args.kfold_idx, test_fold=args.test_fold)

    dataset = {
        'train': ConcatDataset(
            AudioDataset(args.source_data_dir, args.train_data_list, **args.dataset_args, phase='train_source', device=device) \
                if not args.finetune else AudioDataset(args.target_data_dir, args.train_data_list, **args.dataset_args, phase='train_target', device=device),
            AudioDataset(args.target_data_dir, args.adapt_data_list, **args.dataset_args, phase='train_target', device=device)
        ),
        'valid': ConcatDataset(
            AudioDataset(args.source_data_dir, args.valid_data_list, **args.dataset_args, phase='valid_source', device=device),
            AudioDataset(args.target_data_dir, args.valid_data_list, **args.dataset_args, phase='valid_target', device=device)
        )
    }

    loader = {
        'train': DataLoader(dataset['train'], batch_size=args.batch, shuffle=True, drop_last=True),
        'valid': DataLoader(dataset['valid'], batch_size=args.batch)
    }
    
    t2 = time.time()
    print(f'cost {int(t2 - t1)}s', flush=True)


    criterion = {
        'cls': nn.CrossEntropyLoss().to(device),
        'dmn': nn.CrossEntropyLoss(torch.FloatTensor(args.weight) if args.weight is not None else None).to(device)
    }

    for seed in range(200):
        args.seed = seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        print(f'\n{args.title}, seed {args.seed}', flush=True)

        model = CNN(**model_args).to(device)
        n_params = sum( p.numel() for p in model.parameters() )
        print(f'[Model] -# params: {n_params}', flush=True)

        optimizer = getattr(optim, args.optim['name'])(model.parameters(), **args.optim['args'])

        t1 = time.time()
        print('Start Training ... ', end='', flush=True)
        
        runner = Runner(model, loader, device, criterion, optimizer, args=args)
        if args.finetune:
            params = args.params.format(exp_dir=args.exp_dir, seed=args.seed)
            runner.load_checkpoint(params, params_only=True)
        runner.train()
        
        t2 = time.time()
        print(f'cost {int(t2 - t1)}s', flush=True)


if args.test:
    t1 = time.time()
    print('Preprocess Data ... ', end='', flush=True)

    args.test_data_list = args.test_data_list.format(kfold_idx=args.kfold_idx, test_fold=args.test_fold)
    args.logfile = args.logfile.format(exp_dir=args.exp_dir)

    dataset = {
        'test': ConcatDataset(
            AudioDataset(args.source_data_dir, args.test_data_list, **args.dataset_args, phase='test_source', device=device),
            AudioDataset(args.target_data_dir, args.test_data_list, **args.dataset_args, phase='test_target', device=device)
        )
    }

    loader = {
        'test': DataLoader(dataset['test'], batch_size=args.batch)
    }

    t2 = time.time()
    print(f'cost {int(t2 - t1)}s', flush=True)


    model = CNN(**model_args).to(device)
    n_params = sum( p.numel() for p in model.parameters() )
    print(f'[Model] -# params: {n_params}', flush=True)

    for seed in range(200):
        args.seed = seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        print(f'\n{args.title}, seed {args.seed}', flush=True)

        t1 = time.time()
        print('Start Testing ... ', end='', flush=True)
        
        runner = Runner(model, loader, device, args=args)
        params = args.params.format(exp_dir=args.exp_dir, seed=args.seed)
        runner.test(params)

        t2 = time.time()
        print(f'cost {int(t2 - t1)}s', flush=True)