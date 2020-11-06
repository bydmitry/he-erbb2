import json
import numpy as np
from tqdm import tqdm
from termcolor import colored, cprint
import logging, coloredlogs

logging.basicConfig( level = logging.INFO )

coloredlogs.DEFAULT_DATE_FORMAT = '%H:%M:%S'

coloredlogs.DEFAULT_FIELD_STYLES = {
    'asctime'      : {'color': 'white'}, 
    'hostname'     : {'color': 'magenta'}, 
    'levelname'    : {'color': 'black', 'bold': True}, 
    'name'         : {'color': 'blue'}, 'programname': {'color': 'cyan'}
}

coloredlogs.DEFAULT_LEVEL_STYLES = {
    'critical'     : {'color': 'red', 'bold': True}, 
    'debug'        : {'color': 'green'}, 
    'error'        : {'color': 'red'}, 
    'info'         : {'color': 'blue'}, 
    'notice'       : {'color': 'magenta'}, 
    'spam'         : {'color': 'green', 'faint': True}, 
    'success'      : {'color': 'green', 'bold': True}, 
    'verbose'      : {'color': 'blue', 'bold': True}, 
    'warning'      : {'color': 'yellow'}
}

coloredlogs.DEFAULT_LOG_FORMAT  = '%(asctime)s %(message)s'

class Logger:
    """
    Training process logger
    Note:
        Used by BaseTrainer to save training history.
    """
    def __init__(self):
        self.entries = {}

    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry

    def __str__(self):
        return json.dumps(self.entries, sort_keys=True, indent=4)
    
def print_summary_header(config, loaders, model):
    # architecture
    architecture = '{} {}'.format(
        config['architecture']['type'], 
        config['architecture']['args'].get("layers", None))
    trainable_parameters = float( model.__trainable_params__() / 1e3)
    print(colored('\n     :: {} :: '.format(architecture), 'blue', attrs=['bold']))
    print(colored('\nTrainable parameters: {} K\n'.format(trainable_parameters), 'blue'))
    
    # data
    print(colored('Training:', 'blue', attrs=['bold']))
    ldr = loaders['train']
    print_str = '  samples:    {}\n  batches:    {}\n  batch size: {}\n  optimizer step: every {} batches / {} samples\n'.format(
        ldr.dataset.__len__(), len(ldr), ldr.batch_size, 
        config['optimizer']['update_rate'], 
        config['optimizer']['update_rate'] * config['loaders']['train']['batch_size']      )
    print(colored(print_str, 'blue'))
    
    print(colored('Validation:', 'blue', attrs=['bold']))
    ldr = loaders['val']
    print_str = '  samples:    {}\n  batches:    {}\n  batch size: {}\n '.format(
        ldr.dataset.__len__(), len(ldr), ldr.batch_size )
    print(colored(print_str, 'blue'))
    
    
    print(colored('Mean: ', 'blue'), np.round(config['transformations']['mean_vector'],3))
    print(colored('Std:  ', 'blue'), np.round(config['transformations']['std_vector'],3))

    print(colored('\nReproducibility: {}\n'.format(config['reproducibility']['enable']), 'blue', attrs=['bold']))
    
def print_inference_header(config, loaders, model):
    # architecture
    architecture = '{} {}'.format(
        config['architecture']['type'], 
        config['architecture']['args'].get("layers", None))
    print(colored('\n:: {} :: '.format(architecture), 'blue', attrs=['bold']))
    
    # data   
    ldr = loaders['val']
    for k, v in config['dataset'].items():
        print(colored('{:15s}: {}'.format(str(k),str(v)), 'blue'))
    print_str = '{:15s}: samples: {}   batches: {}   batch size: {}\n '.format(
        'Evaluate', ldr.dataset.__len__(), len(ldr), ldr.batch_size )
    print(colored(print_str, 'blue'))
    
    print(colored('Mean: {}  Std: {}', 'blue').format(
        np.round(config['transformations']['mean_vector'],3),
        np.round(config['transformations']['std_vector'],3)) )