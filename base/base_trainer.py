import os
import math
import json
import shutil
import datetime
import numpy as np
import logging, coloredlogs
from tqdm import tqdm
from termcolor import colored, cprint

import torch
from utils.visualization import WriterTensorboardX

# -------------------------- misc ---------------------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

# ------------------------ BaseTrainer ----------------------------------
class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion_dict, metrics_dict, resume, config, train_logger=None, inference=False):
        self.inference = inference
        self.config    = config
        self.logger    = logging.getLogger(self.__class__.__name__)
        coloredlogs.install(level='INFO', logger=self.logger)

        # ----- setup GPU device if available  ----- #
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model              = model.to(self.device)

        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model)
        else:
            self.model = torch.nn.DataParallel(model, device_ids=[0])

        if self.inference:
            pass
        else:
            # ---  optimizer & lr scheduling  ---
            self.optimizer = get_instance(torch.optim, 'optimizer', config, filter(lambda p: p.requires_grad, model.parameters()) )

            self.lr_scheduler  = None
            if config['lr_scheduler']['enable']:
                self.lr_scheduler  = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, self.optimizer)

            self.criterions    = criterion_dict
            self.metrics       = metrics_dict
            self.train_logger  = train_logger

            cfg_trainer        = config['trainer']
            self.epochs        = cfg_trainer['epochs']
            self.save_period   = cfg_trainer['save_period']
            self.verbosity     = cfg_trainer['verbosity']
            self.monitor       = cfg_trainer.get('monitor', 'off')

            # ----- initialize directory for checkpoint saving ----- #
            start_time           = datetime.datetime.now().strftime('%b%d-%H-%M-%S')
            exper_name           = os.path.join(('--').join([config['architecture']['type'], config['name']]), start_time)
            self.checkpoint_dir  = os.path.join(cfg_trainer['save_dir'], exper_name)
            ensure_dir(self.checkpoint_dir)

            # ----- initialize directory for experiment setup information ----- #
            setup_info_dir       = os.path.join(self.checkpoint_dir, 'setup-info')
            ensure_dir(setup_info_dir)

            # ----- initialize tensorboard writer instance ----- #
            writer_dir = os.path.join(self.checkpoint_dir, 'tensorboardX')
            self.writer = WriterTensorboardX(writer_dir, self.logger, cfg_trainer['tensorboardX'])

            # ----- dump configuration file into checkpoint directory ----- #
            config_save_path = os.path.join(setup_info_dir, 'config.json')
            with open(config_save_path, 'w') as handle:
                json.dump(config, handle, indent = 4, sort_keys = False)

            # ----- copy data-frame into checkpoint directory  ----- #
            shutil.copy(config['dataset']['data_frame'], setup_info_dir)

            # ----- monitor model performance and save best ----- #
            if self.monitor == 'off':
                self.mnt_mode = 'off'
                self.mnt_best = 0
            else:
                self.mnt_mode, self.mnt_metric = self.monitor.split()
                assert self.mnt_mode in ['min', 'max']

                self.mnt_best = math.inf if self.mnt_mode == 'min' else -math.inf
                self.early_stop = cfg_trainer.get('early_stop', math.inf)

            self.start_epoch = 1

        if resume:
            self._resume_checkpoint(resume)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()

        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: Configured to use {} GPU(s), but only {} are available.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu

        device   = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))

        return device, list_ids

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            # prepare for epoch
            if epoch == self.config['optimizer']['unfreeze_encoder']:

                #for param in self.model.module.encoder[-4:-1].parameters():
                #    param.requires_grad = True

                for param in self.model.module.encoder2.parameters():
                    param.requires_grad = True
                for param in self.model.module.encoder3.parameters():
                    param.requires_grad = True

                #self.optimizer.add_param_group({'params': self.model.module.encoder[-4:-1].parameters()})
                self.optimizer.add_param_group({'params': self.model.module.encoder2.parameters()})
                self.optimizer.add_param_group({'params': self.model.module.encoder3.parameters()})

                model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
                trainable_parameters = float( sum([np.prod(p.size()) for p in model_parameters]) / 1e3)
                print(colored('Encoder released, learning {} K parameters.'.format(trainable_parameters), 'blue'))

            # train
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            if self.verbosity >= 1:
                for key, value in result.items():
                    if key == 'metrics':
                        log.update({mtr.__name__ : value[i] for i, mtr in enumerate(self.metrics)})
                    elif key == 'val_metrics':
                        log.update({'val_' + mtr.__name__ : value[i] for i, mtr in enumerate(self.metrics)})
                    else:
                        log[key] = value

            # print logged informations to the screen
            if self.train_logger is not None:
                self.train_logger.add_entry(log)
                if self.verbosity >= 1:
                    for key, value in log.items():
                        self.logger.info('{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] < self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] > self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False
                    not_improved_count = 0

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch  = type(self.model).__name__

        state = {
            'architecture' : arch,
            'epoch'        : epoch,
            'logger'       : self.train_logger,
            'state_dict'   : self.model.state_dict(),
            'optimizer'    : self.optimizer.state_dict(),
            'monitor_best' : self.mnt_best,
            'config'       : self.config
        }

        filename = os.path.join(self.checkpoint_dir, 'epoch-{}.pth'.format(epoch))
        torch.save(state, filename)
        if self.verbosity >= 1:
            self.logger.info("Checkpoint: {} ".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Current best: {} ".format('model_best.pth'))

    def _resume_checkpoint(self, resume_path, vrbs=False):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        if vrbs:
            self.logger.info("Loading checkpoint: {} ...".format(resume_path))

        checkpoint        = torch.load(resume_path)
        self.start_epoch  = checkpoint['epoch'] + 1
        self.mnt_best     = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['architecture'] != self.config['architecture']:
            self.logger.warning('Warning: Architecture configuration given in config file is different from that of checkpoint. ' + \
                                'This may yield an exception while state_dict is being loaded.')
        self.model.load_state_dict(checkpoint['state_dict'])

        if not self.inference:
            # load optimizer state from checkpoint only when optimizer type is not changed.
            if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
                self.logger.warning('Warning: Optimizer type given in config file is different from that of checkpoint. ' + \
                                    'Optimizer parameters not being resumed.')
            else:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.train_logger = checkpoint['logger']
        if vrbs:
            self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))
