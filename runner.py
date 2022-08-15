from audioop import mul
import os
import torch
import numpy as np
from tqdm            import tqdm
from collections     import defaultdict
from tensorboardX    import SummaryWriter
from sklearn.metrics import multilabel_confusion_matrix

import pdb


class Runner():
    def __init__(self, model, loader, device, criterion=None, optimizer=None, args=None):
        self.epoch   = 0
        self.metrics = defaultdict(float)

        self.model     = model
        self.loader    = loader
        self.device    = device
        self.criterion = criterion
        self.optimizer = optimizer

        if args.train:
            self.writer = SummaryWriter(os.path.join(args.log_dir, args.title))

        if args.patience != -1 and args.max_epoch == -1:
            args.max_epoch = np.iinfo(int).max
        
        for mtr in args.monitor.keys():
            if args.monitor[mtr]['mode'] == 'min':
                args.monitor[mtr]['record'] = np.finfo(float).max
            else:
                args.monitor[mtr]['record'] = np.finfo(float).min

            args.monitor[mtr]['cnt'] = 0

        vars(self).update({ key: val for key, val in vars(args).items() 
                if key not in list(vars(self).keys()) + dir(self) })


    def save_checkpoint(self, checkpoint='ckpt.pt'):
        state_dict = {
            'epoch'    : self.epoch,
            'model'    : self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        
        os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
        torch.save(state_dict, checkpoint)


    def load_checkpoint(self, checkpoint='ckpt.pt', params_only=False):
        state_dict = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state_dict['model'])

        if not params_only:
            self.epoch = state_dict['epoch']
            self.optimizer.load_state_dict(state_dict['optimizer'])


    def _update_callback(self, save_best_only=True):
        if not save_best_only:
            self.save_checkpoint(os.path.join(self.exp_dir, self.title, f'seed{self.seed}', f'ckpt/epoch_{self.epoch}.pt')) 

        for mtr in self.monitor.keys():
            if (self.monitor[mtr]['mode'] == 'min' and self.metrics[mtr] < self.monitor[mtr]['record']) or \
               (self.monitor[mtr]['mode'] == 'max' and self.metrics[mtr] > self.monitor[mtr]['record']):
                
                self.monitor[mtr]['record'] = self.metrics[mtr]
                self.monitor[mtr]['cnt'] = 0
                
                self.save_checkpoint(os.path.join(self.exp_dir, self.title, f'seed{self.seed}', f'ckpt/best_{"_".join(mtr.split("/"))}.pt')) 
                
            else:
                self.monitor[mtr]['cnt'] += 1


    def _check_early_stopping(self):
        return self.patience != -1          and \
               self.epoch >= self.min_epoch and \
               all([ info['cnt'] >= self.patience for info in self.monitor.values() ])


    def _write_to_tensorboard(self, iteration):
        for key, val in self.metrics.items():
            self.writer.add_scalar(key, val, iteration)


    @staticmethod
    def _display(phase='Train', iteration=None, **kwargs):
        disp = f'[{phase}]'

        if iteration is not None:
            disp += f" Iter {iteration}"
        
        for key, value in kwargs.items():
            if key.endswith('loss'):
                disp += f" - {'_'.join(key.split('/'))}: {value:4.3e}"
            else:
                disp += f" - {'_'.join(key.split('/'))}: {value * 100:4.2f}"
        
        print(disp, flush=True)


    def _train_step(self, source_batch, target_batch, mode='naive'):
        source_x, source_cls, source_dmn = source_batch
        target_x, target_cls, target_dmn = target_batch

        pred_source_cls, pred_source_dmn = self.model(source_x)
        if mode != 'naive':
            pred_target_cls, pred_target_dmn = self.model(target_x)

        source_cls_loss = self.criterion['cls'](pred_source_cls, source_cls)
        if   mode == 'joint':    
            target_cls_loss = self.criterion['cls'](pred_target_cls, target_cls)
        elif mode == 'adapt':            
            source_dmn_loss = self.criterion['dmn'](pred_source_dmn, source_dmn)
            target_dmn_loss = self.criterion['dmn'](pred_target_dmn, target_dmn)

        if   mode == 'naive':
            loss = source_cls_loss
        elif mode == 'joint':
            loss = (source_cls_loss + target_cls_loss) / 2
        elif mode == 'adapt':
            loss = source_cls_loss + (source_dmn_loss + target_dmn_loss) / 2
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.metrics['train/loss']         += loss.item()
        self.metrics['train/src_cls_loss'] += source_cls_loss.item()   
        if   mode == 'joint':    
            self.metrics['train/tgt_cls_loss'] += target_cls_loss.item()
        elif mode == 'adapt':            
            self.metrics['train/src_dmn_loss'] += source_dmn_loss.item()
            self.metrics['train/tgt_dmn_loss'] += target_dmn_loss.item()


    @torch.no_grad()
    def _valid_step(self, source_batch, target_batch, mode='naive'):
        source_x, source_cls, source_dmn = source_batch
        target_x, target_cls, target_dmn = target_batch

        pred_source_cls, pred_source_dmn = self.model(source_x)
        pred_target_cls, pred_target_dmn = self.model(target_x)

        source_cls_loss = self.criterion['cls'](pred_source_cls, source_cls)
        source_dmn_loss = self.criterion['dmn'](pred_source_dmn, source_dmn)
        target_cls_loss = self.criterion['cls'](pred_target_cls, target_cls)
        target_dmn_loss = self.criterion['dmn'](pred_target_dmn, target_dmn)

        if   mode == 'naive':
            loss = source_cls_loss
        elif mode == 'adapt':
            loss = source_cls_loss + (source_dmn_loss + target_dmn_loss) / 2
        elif mode == 'joint':
            loss = (source_cls_loss + target_cls_loss) / 2

        self.metrics['valid/loss']         += loss.item() * len(source_x)
        self.metrics['valid/src_cls_loss'] += source_cls_loss.item() * len(source_x)
        self.metrics['valid/src_dmn_loss'] += source_dmn_loss.item() * len(source_x)
        self.metrics['valid/tgt_cls_loss'] += target_cls_loss.item() * len(source_x)
        self.metrics['valid/tgt_dmn_loss'] += target_dmn_loss.item() * len(source_x)

        return pred_source_cls, pred_target_cls


    def train(self):
        while self.epoch < self.max_epoch:
            self.epoch  += 1
            self.metrics = defaultdict(float)


            self.model.train()
            self.loader['train'].dataset.resample()
            for i, (source_batch, target_batch) in enumerate(tqdm(self.loader['train'])):
                source_batch = list(map(lambda item: item.to(self.device), source_batch))
                target_batch = list(map(lambda item: item.to(self.device), target_batch))
                self._train_step(source_batch, target_batch, self.mode)


            self.model.eval()
            source_true, source_pred = [], []
            target_true, target_pred = [], []
            for i, (source_batch, target_batch) in enumerate(tqdm(self.loader['valid'])):
                source_batch = list(map(lambda item: item.to(self.device), source_batch))
                target_batch = list(map(lambda item: item.to(self.device), target_batch))
                pred_source_cls, pred_target_cls = self._valid_step(source_batch, target_batch, self.mode)

                source_pred += list(torch.argmax(pred_source_cls, dim=1).cpu().numpy())
                target_pred += list(torch.argmax(pred_target_cls, dim=1).cpu().numpy())
                source_true += list(source_batch[1].cpu().numpy())
                target_true += list(target_batch[1].cpu().numpy())

            source_confusions = multilabel_confusion_matrix(source_true, source_pred)
            target_confusions = multilabel_confusion_matrix(target_true, target_pred)
            source_recalls = [ mat[1, 1] / (mat[1, 0] + mat[1, 1]) for mat in source_confusions ]
            target_recalls = [ mat[1, 1] / (mat[1, 0] + mat[1, 1]) for mat in target_confusions ]

            self.metrics['valid/avg_uar'] = (np.mean(source_recalls)+ np.mean(target_recalls)) / 2
            self.metrics['valid/src_uar'] = np.mean(source_recalls)
            self.metrics['valid/src_nor'] = source_recalls[0]
            self.metrics['valid/src_neo'] = source_recalls[1]
            self.metrics['valid/src_pho'] = source_recalls[2]
            self.metrics['valid/tgt_uar'] = np.mean(target_recalls)
            self.metrics['valid/tgt_nor'] = target_recalls[0]
            self.metrics['valid/tgt_neo'] = target_recalls[1]
            self.metrics['valid/tgt_pho'] = target_recalls[2]


            for key, value in self.metrics.items():
                if key.endswith('loss'):
                    if key.startswith('train'):
                        self.metrics[key] = value / len(self.loader['train'])
                    else:       
                        self.metrics[key] = value / len(self.loader['valid'].dataset)

            self._display('Train', self.epoch, **self.metrics)
            
            self._write_to_tensorboard(self.epoch)
            
            self._update_callback(self.save_best_only)

            if self._check_early_stopping(): break


    @torch.no_grad()
    def test(self, checkpoint=None):
        self.load_checkpoint(checkpoint, params_only=True)

        self.model.eval()
        source_true, source_pred = [], []
        target_true, target_pred = [], []
        for source_batch, target_batch in tqdm(self.loader['test']):
            source_x, source_cls, _ = source_batch
            target_x, target_cls, _ = target_batch
    
            source_x = source_x.to(self.device)
            target_x = target_x.to(self.device)
            
            pred_source_cls, _ = self.model(source_x)
            pred_target_cls, _ = self.model(target_x)
            
            source_pred += list(torch.argmax(pred_source_cls, dim=1).cpu().numpy())
            target_pred += list(torch.argmax(pred_target_cls, dim=1).cpu().numpy())
            source_true += list(source_cls.numpy())
            target_true += list(target_cls.numpy())
        
        source_confusions = multilabel_confusion_matrix(source_true, source_pred)
        target_confusions = multilabel_confusion_matrix(target_true, target_pred)
        source_recalls = [ mat[1, 1] / (mat[1, 0] + mat[1, 1]) for mat in source_confusions ]
        target_recalls = [ mat[1, 1] / (mat[1, 0] + mat[1, 1]) for mat in target_confusions ]
        
        self.metrics['test/avg_uar'] = (np.mean(source_recalls)+ np.mean(target_recalls)) / 2
        self.metrics['test/src_uar'] = np.mean(source_recalls)
        self.metrics['test/src_nor'] = source_recalls[0]
        self.metrics['test/src_neo'] = source_recalls[1]
        self.metrics['test/src_pho'] = source_recalls[2]
        self.metrics['test/tgt_uar'] = np.mean(target_recalls)
        self.metrics['test/tgt_nor'] = target_recalls[0]
        self.metrics['test/tgt_neo'] = target_recalls[1]
        self.metrics['test/tgt_pho'] = target_recalls[2]

        self._display('Test', **self.metrics)