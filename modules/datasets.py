from __future__ import print_function, division
import os
import sys
import h5py
import numpy as np
import pandas as pd
from termcolor import colored, cprint

from utils.augmentation import *
import matplotlib.pyplot as plt
from skimage import io, transform

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import solt.core as slc
import solt.data as sld
import solt.transforms as slt

def get_data_loaders(conf):
    datset_cnfg       = conf['dataset']
    loaders_cnfg      = conf['loaders']
    transforms_cnfg   = conf['transformations']

    data_frames       = split_samples(datset_cnfg)
    transformations   = compose_transforms(transforms_cnfg)

    data_sets         = instantiate_dataset(
                                    datset_cnfg,
                                    data_frames,
                                    transformations)

    data_loaders = {
        split : DataLoader(data_sets[split], **loaders_cnfg[split]) for split in ['train', 'val']
    }

    return data_loaders

# ------------------------- helpers ---------------------------------------
def split_samples(conf):
    ''' prepare taining and validation splits '''
    df   = pd.read_csv(conf['data_frame'])
    col  = ('.').join(['cv', conf['split_col']])

    df.dropna(subset=[col], inplace=True)
    df[col]   = df[col].astype('uint32')

    train_df  = df.loc[ df[col] != int(conf['split_num']) ]
    val_df    = df.loc[ df[col] == int(conf['split_num']) ]

    return { 'train' : train_df, 'val' : val_df }

def compose_transforms(conf):
    ''' prepare transformations '''
    mean_vector = np.array(conf['mean_vector'], dtype=np.float32)
    std_vector  = np.array(conf['std_vector'], dtype=np.float32)

    # mean() std() scaling
    normTransform = transforms.Normalize(
        torch.from_numpy(mean_vector).float(),
        torch.from_numpy(std_vector).float()
    )

    # train transforms:
    source_crop   = conf['crop_size']
    scale_factor  = conf['rescale']

    scale_range   = scale_factor * conf['scale_range']
    shear_range   = conf['shear_range']
    gamma_range   = conf['gamma_range']

    adj_angle     = int(np.ceil(np.sqrt(2*source_crop**2)))
    adj_scale     = int(source_crop*scale_range)

    extended_crop = adj_angle + adj_scale
    final_crop    = int(source_crop*scale_factor)

    train_transforms = transforms.Compose([
        toSOLT,
        slc.Stream([
            slt.PadTransform((extended_crop, extended_crop), padding='r'),
            slt.CropTransform(extended_crop, 'r'),

            slt.RandomProjection(
                slc.Stream([
                    slt.RandomScale(range_x=(scale_factor-scale_range, scale_factor+scale_range), p=0.5),
                    slt.RandomRotate(rotation_range=(-90, 90), p=0.5),
                    slt.RandomShear(range_x=(-shear_range, shear_range), range_y=(-shear_range, shear_range), p=0.5),
                ]), v_range=(1e-6, 3e-5), p=1.0),

            slt.ImageGammaCorrection(p=0.6, gamma_range=gamma_range),
            slt.PadTransform((final_crop, final_crop), padding='r'),
            slt.CropTransform(final_crop, crop_mode='c'),
        ], padding='r'),
        fromSOLT,
        transforms.ToTensor(),
        normTransform
    ])

    # validation transforms
    source_crop = conf['val_crop']
    val_transforms = transforms.Compose([
        toSOLT,
        slc.Stream([
            slt.PadTransform((source_crop, source_crop), padding='r'),
            slt.CropTransform(source_crop, 'c'),
            slt.RandomScale(range_x=(scale_factor,scale_factor), p=1.0)
        ], padding='r'),
        fromSOLT,
        transforms.ToTensor(),
        normTransform
    ])

    return { 'train' : train_transforms, 'val' : val_transforms }

def instantiate_dataset(ds_conf, dframes, tsfms):
    ''' instantiate dataset '''
    dset_module = getattr(sys.modules[__name__], ds_conf['type'])

    d_sets = {
        split : dset_module(ds_conf['hdf5_file'], dframes[split], tsfms[split], aux_vars=ds_conf['aux_vars']) for split in ['train', 'val']
    }

    return d_sets

def toSOLT(img):
    return sld.DataContainer(img, 'I')

def fromSOLT(dc: sld.DataContainer):
    return dc.data[0]

# ------------------------- datasets --------------------------------------
class FinProg(Dataset):
    def __init__(self, hdf5_file, data_frame, tsfrm=None, aux_vars=None):
        self.data_frame  = data_frame
        self.hdf5_file   = hdf5_file
        self.tsfrm       = tsfrm
        self.aux         = aux_vars

    def __len__(self):
        return int(self.data_frame.shape[0])

    def __getitem__(self, idx):
        df_row = self.data_frame.iloc[idx]

        with h5py.File(self.hdf5_file, "r") as hdf:
            img = hdf[ str(int(df_row['id'])) ][()]

        if self.tsfrm:
            img = self.tsfrm(img)

        # main covariates:
        sample = {
            'id'     : int(df_row['id']),
            'img'    : img,
            'event'  : int(df_row['dss.new']),
            'fu'     : float(df_row['fu.new']),
            'grank'  : float(df_row['grank'])
        }

        # auxiliary variables:
        if self.aux is not None:
            for var in self.aux:
                sample[var] = float(df_row[var])

        return sample

def prepare_sample(sample, device, config):
    data = {
        'id'     : sample['id'].long(),
        'rgb'    : sample['img'].float().to(device),
        'grank'  : sample['grank'].float().to(device),
        'event'  : sample['event'].float().to(device),
        'fu'     : sample['fu'].float().to(device)
    }

    # prepare aux covarates:
    aux = config['dataset']['aux_vars']
    if aux is not None:
        for var in aux:
            data[var] = sample[var].float().to(device)

    # concat data for the network aux inputs (mlp):
    data['aux_inputs'] = None
    aux_inputs = config['architecture']['args']['aux_input']

    if aux_inputs is not None:
        aux_list = list()
        for aux in aux_inputs['covars']:
            aux_list.append(data[aux])

        data['aux_inputs'] = torch.stack(aux_list, dim=1)

    return data

# ------------------------ FinHer -----------------------------------------
class FinHerTiles(Dataset):
    def __init__(self, df, tile_dir, t_size=1000, scale_factor=0.99):
        self.df       = df
        self.tile_dir = tile_dir

        # filter data_frame with tiles
        self.df = self.df.loc[
            (self.df['NORMALIZED'] == 1) &
            (self.df['HEIGHT'] == t_size) &
            (self.df['WIDTH'] == t_size)
        ]

        # prepare normalization stats from FinProg
        mean_vector = np.array([ 0.8198558, 0.78990823, 0.91205645 ])
        std_vector  = np.array([ 0.1421396, 0.15343277, 0.07634846 ])

        normTransform = transforms.Normalize(
            torch.from_numpy(mean_vector).float(),
            torch.from_numpy(std_vector).float()
        )

        self.torch_compose = transforms.Compose([
            toSOLT,
            slt.RandomScale(range_x=(scale_factor,scale_factor), p=1.0),
            fromSOLT,
            transforms.ToTensor(),
            normTransform
        ])

    def __len__(self):
        return int(self.df.shape[0])

    def __getitem__(self, idx):
        t_name = '{}.jpg'.format(self.df.iloc[idx]['TILE_NAME'])

        img = cv2.imread(os.path.join(self.tile_dir, t_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.torch_compose(img)

        sample = {
            'id'     : int(self.df.iloc[idx]['id']),
            'img'    : img,
            'event'  : int(1),
            'fu'     : float(1.0),
            'grank'  : float(1.0),
            'ER'     : int(1),
            'PR'     : int(1),
            'HER2'   : int(1)
        }

        return sample
# ------------------------- eof -------------------------------------------
