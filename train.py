import os
import sys
import time
import json
import argparse
from utils import Logger, print_summary_header

import numpy as np
import pandas as pd

from tqdm import tqdm
from termcolor import colored, cprint

import torch
import modules.losses as losses_
import modules.metrics as metrics_
import modules.trainers as trainers_

from modules.datasets import get_data_loaders
from modules.networks import instantiate_network

def main(config, resume):
    train_logger = Logger()
    
    # --- loaders ---
    data_loaders = get_data_loaders(config)
    
    # ---  model  ---
    model = instantiate_network(config)
    
    # ---  loss & metrics  ---
    loss_dict   = dict()
    metric_dict = dict()
    
    for d_ in config['architecture']['args']['aux_outputs']:
        loss_dict[d_['name']]   = getattr(losses_, d_['loss']['type'])(**d_['loss']['args'])
        metric_dict[d_['name']] = [getattr(metrics_, met) for met in d_['metrics']]        
    
    # ---  print info before training  ---
    print_summary_header(config, data_loaders, model)
    
    trainer = trainers_.Trainer(model, loss_dict, metric_dict,  
                      resume            = resume,
                      config            = config,
                      data_loaders      = data_loaders,
                      train_logger      = train_logger)

    trainer.train()
    

# ------------------------- __main__-------------------------------------
if __name__ == '__main__':
    print(colored('----- Training Routines -----', 'blue', attrs=['bold']))
    
    parser = argparse.ArgumentParser(description='Main training script')
    parser.add_argument('-c', '--config', default=None, type=str,
                           help='json config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')

    args = parser.parse_args()

    # --- validate arguments  --- #
    if args.config:
        # load config file
        config = json.load(open(args.config))
        path   = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume)['config']
    else:
        print(colored('Configuration file needs to be specified: --config', 'red'))
        sys.exit(1)    

    # ---  reproducibility  --- #
    seeds = config['reproducibility']
    if seeds['enable']:
        np.random.seed(seeds['numpySeed'])
        torch.manual_seed(seeds['torchSeed'])
        torch.backends.cudnn.deterministic = seeds['cudnn.deterministic']
        torch.backends.cudnn.benchmark     = seeds['cudnn.benchmark']        

    main(config, args.resume)    
    print(colored('\n--------------- Done ----------------', 'blue', attrs=['bold']))

# ----------------------  Done -----------------------------------------
