import os
import sys
import json
import argparse
from tqdm import tqdm
from termcolor import colored, cprint
from utils import print_inference_header

import numpy as np
import pandas as pd

import torch
import modules.losses as losses_
import modules.metrics as metrics_
import modules.trainers as trainers_
from torch.utils.data import DataLoader

from modules.datasets import FinHerTiles
from modules.networks import instantiate_network

def main(config):
    models = config['models']

    # ---  prepare main data-frame  ---
    root = config["root"]

    # get first-level subdirs:
    dir_list = [ o for o in os.listdir(root) if os.path.isdir(os.path.join(root,o)) and o.startswith('FinHer')]

    for slide_name in tqdm(dir_list, leave=True):
        slide_dir  = os.path.join(root, slide_name)

        tile_dir   = os.path.join(slide_dir, 'Normalized-Tiles')

        # read & filter data_frame with tiles
        df = 'Normalized-{}-Tiles.csv'.format(slide_name)
        df = pd.read_csv(os.path.join(slide_dir, df))
        df['id'] = df.index

        for model_dict in models:
            #print(colored('> {}'.format(model_dict['snapshot']), 'magenta', attrs=['bold']))

            model_config = torch.load(model_dict['snapshot'])['config']

            # --- loaders ---
            data_set     = FinHerTiles(df, tile_dir, t_size=config['tile_size'])
            data_loaders = dict()
            data_loaders['val'] = DataLoader(
                data_set,
                batch_size  = config["batch_size"],
                shuffle     = False,
                num_workers = config["num_workers"],
                drop_last   = config["drop_last"]
            )

            # ---  model  ---
            model = instantiate_network(model_config)

            # ---  loss & metrics  ---
            loss_dict   = dict()
            metric_dict = dict()

            for d_ in model_config['architecture']['args']['aux_outputs']:
                loss_dict[d_['name']]   = getattr(losses_, d_['loss']['type'])(**d_['loss']['args'])
                metric_dict[d_['name']] = [getattr(metrics_, met) for met in d_['metrics']]

            # ---  instantiate model instance  ---
            trainer = trainers_.Trainer(model, loss_dict, metric_dict,
                              resume            = model_dict['snapshot'],
                              config            = model_config,
                              data_loaders      = data_loaders,
                              train_logger      = None,
                              inference         = True )

            # ---  inference  ---
            result_dict = trainer._predict(vrbs=False)
            
            # --- drop attention maps when applicabel ---
            _ = result_dict.pop("a1", None)
            _ = result_dict.pop("a2", None)
            
            # --- rename column ---
            cc_ = list(result_dict.keys())
            cc_.remove('id')
            for col in cc_:
                ccnew_ = '{}{}'.format(col,model_dict['name'])
                result_dict[ccnew_] = result_dict.pop(col)

            # ---  append results to a data-frame  ---
            df = pd.merge(
                left      = df,
                right     = pd.DataFrame(result_dict, index=None),
                on        = 'id',
                how       = 'left',
                validate  = 'one_to_one'
            )

        # ---  save results to csv-file  ---
        csv_name = '{}-{}.csv'.format(slide_name, config['analysis'])
        df.to_csv(
            os.path.join(slide_dir, csv_name),
            index=False
        )

# ------------------------- __main__-------------------------------------
if __name__ == '__main__':
    print(colored('----- Inference on tiles -----', 'blue', attrs=['bold']))

    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('-c', '--config', default=None, type=str,
                           help='json config file path (default: None)')

    args = parser.parse_args()

    # --- validate arguments  --- #
    if args.config:
        # load config file
        config = json.load(open(args.config))
    else:
        print(colored('Configuration file needs to be specified: --config', 'red'))
        sys.exit(1)

    main(config)
    print(colored('\n--------------- Done ----------------', 'blue', attrs=['bold']))

# ----------------------  Done -----------------------------------------
