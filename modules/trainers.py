import os
import gc
import sys
import time

import pickle
import numpy as np

from tqdm import tqdm
from termcolor import colored, cprint
from base import BaseTrainer

import torch
from torchvision.utils import make_grid

from modules.datasets import prepare_sample

# ---------------------------- Helpers ----------------------------------
def create_collector(items):
    data = { item: [] for item in items }
    return data

def collect_batches(batch_data, sample_items, pred_items):
    epoch_dict = {'samples' : {}, 'preds' : {}}

    for item in pred_items:
        ll = [ batch[item].squeeze() for batch in batch_data['preds'] ]
        epoch_dict['preds'][item] = torch.cat(ll)

    for item in sample_items:
        if (item != 'rgb') and (batch_data['samples'][0][item] is not None):
            ll = [ batch[item].squeeze() for batch in batch_data['samples'] ]
            epoch_dict['samples'][item] = torch.cat(ll)

    return epoch_dict


# --------------------------- Trainers ----------------------------------
class Trainer(BaseTrainer):
    """
    Trainer class: Inherited from BaseTrainer.
    """
    def __init__(self, model, criterion, metrics, resume, config, data_loaders, train_logger = None, inference=False):
        super(Trainer, self).__init__(model, criterion, metrics, resume, config, train_logger, inference)

        self.config             = config
        self.data_loaders       = data_loaders
        self.do_validation      = data_loaders['val'] is not None

    # -------------------------- train() ------------------------------------
    def _train_epoch(self, epoch):
        """
        Training routine for an epoch.
        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        print(colored('Epoch [ {} ] : '.format(epoch), 'magenta', attrs=['bold']))

        self.model.train()
        loader = self.data_loaders['train']
        self.writer.set_step(epoch, 'training')

        optimizer_update_rate = self.config['optimizer']['update_rate']

        batch_collector = create_collector(['samples','preds'])

        n_batches      = len(loader)
        batch_iterator = tqdm(enumerate(loader), total=n_batches, leave=False)

        # ----- iterate batches ----- #
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            for batch_idx, sample in batch_iterator:
                # ----- prepare inputs ----- #
                sample = prepare_sample(sample, self.device, self.config)

                # ----- forward ----- #
                pred = self.model(sample)

                # ----- backward & weights update ----- #
                loss = self._eval_losses({'preds':pred, 'samples':sample})
                loss['total'].backward()

                if (batch_idx+1) % optimizer_update_rate  == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # ----- collect batch data ----- #
                sample.pop('rgb', None)
                sample.pop('aux_inputs', None)

                batch_collector['samples'].append(sample)
                batch_collector['preds'].append(pred)

        # ----- stack batches ----- #
        epoch_data = collect_batches(
            batch_collector,
            sample.keys(),
            pred.keys()
        )

        # ----- epoch loss & metrics ----- #
        loss    = self._eval_losses(epoch_data, tensor_board=True)
        metrics = self._eval_metrics(epoch_data)

        log = {
            'loss'    : loss['total'].item(),
            'metrics' : metrics
        }

        gc.collect()
        # ----- validation ----- #
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        # ----- lr scheduler ----- #
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    # ------------------------- validate() ----------------------------------
    def _valid_epoch(self, epoch):
        """
        Validation routine for an epoch.
        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        loader = self.data_loaders['val']
        self.writer.set_step(epoch, 'validation')

        batch_collector = create_collector(['samples','preds'])

        n_batches      = len(loader)
        batch_iterator = tqdm(enumerate(loader), total=n_batches, leave=False)

        with torch.no_grad():
            for batch_idx, sample in batch_iterator:
                # ----- prepare inputs ----- #
                sample = prepare_sample(sample, self.device, self.config)

                # ----- forward ----- #
                pred = self.model(sample)

                # ----- collect batch data ----- #
                sample.pop('rgb', None)
                sample.pop('aux_inputs', None)

                batch_collector['samples'].append(sample)
                batch_collector['preds'].append(pred)

        # ----- stack batches ----- #
        epoch_data = collect_batches(
            batch_collector,
            sample.keys(),
            pred.keys()
        )

        # ----- loss & metrics ----- #
        loss    = self._eval_losses(epoch_data, tensor_board=True)
        metrics = self._eval_metrics(epoch_data)

        return {
            'val_loss'    : loss['total'].item(),
            'val_metrics' : metrics
        }
    # ------------------------- predict() -----------------------------------
    def _predict(self, vrbs=True):
        if vrbs:
            print(colored('Inference... ', 'magenta', attrs=['bold']))

        self.model.eval()

        # ----- disable grads for all layers ----- #
        for param in self.model.module.encoder.parameters():
            param.requires_grad = False

        loader = self.data_loaders['val']

        batch_collector = create_collector(['samples','preds'])

        n_batches      = len(loader)
        batch_iterator = tqdm(enumerate(loader), total=n_batches, leave=True)

        with torch.no_grad():
            for batch_idx, sample in batch_iterator:
                # ----- prepare inputs ----- #
                sample = prepare_sample(sample, self.device, self.config)

                # ----- forward propagation ----- #
                pred  = self.model(sample)

                # ----- gather bacth data ----- #
                sample.pop('rgb', None)
                sample.pop('aux_inputs', None)

                batch_collector['samples'].append(sample)
                batch_collector['preds'].append(pred)

        # ----- stack batches ----- #
        epoch_data = collect_batches(
            batch_collector,
            sample.keys(),
            pred.keys()
        )

        # ----- prepare results ----- #
        results_dict = {
            'id' : epoch_data['samples']['id'].cpu().numpy()
        }

        for item in epoch_data['preds'].keys():
            results_dict[item] = epoch_data['preds'][item].detach().cpu().numpy()

        return results_dict

    # ------------------------ evaluations ----------------------------------
    def _eval_losses(self, data_dict, tensor_board=False):
        loss_dict = {}

        totalLoss = 0
        for node_name, criterion in self.criterions.items():
            loss_dict[node_name] = criterion(
                data_dict['preds'][node_name].squeeze(),
                data_dict['samples'][node_name]
            )

            totalLoss += loss_dict[node_name]

        loss_dict['total'] = totalLoss

        # ----- tensorboardX ----- #
        if tensor_board:
            for k, v in loss_dict.items():
                self.writer.add_scalar('{}-loss'.format(k), v.item())

        return loss_dict

    def _eval_metrics(self, data_dict):
        acc_metrics = list()

        for item in data_dict['preds'].keys():
            data_dict['preds'][item] = data_dict['preds'][item].detach().cpu().numpy()
        for item in data_dict['samples'].keys():
            data_dict['samples'][item] = data_dict['samples'][item].cpu().numpy()

        for node_name, met_list in self.metrics.items():
            for metric in met_list:
                if metric.__name__ in [ 'auc', 'aver_prec']:
                    y_pred = data_dict['preds'][node_name]
                    y_true = data_dict['samples'][node_name].astype(np.uint8)

                    acc = metric(y_true, y_pred)
                    self.writer.add_scalar('{}_{}'.format(node_name, metric.__name__), acc)
                    self.writer.add_pr_curve('{}_PR'.format(node_name), labels=y_true, predictions=y_pred)


        return acc_metrics
