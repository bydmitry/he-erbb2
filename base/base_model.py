import os
import logging
import numpy as np

import torch.nn as nn
from termcolor import colored, cprint

class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params       = sum([np.prod(p.size()) for p in model_parameters])
        show_params  = float( params / 1e3 )
        print(colored('\n     :: {} :: '.format(self.__class__.__name__), 'blue', attrs=['bold']))
        print(colored('\nTrainable parameters: {} K\n'.format(show_params), 'blue'))

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        #return super(BaseModel, self).__name__() + '\nTrainable parameters: {}'.format(params)
        return '\nTrainable parameters: {}'.format(params)
        
    def __trainable_params__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])