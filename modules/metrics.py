import torch

import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

def aver_prec(observed, predicted_scores):
    return average_precision_score(observed, predicted_scores)

def auc(observed, predicted):
    return roc_auc_score(observed, predicted)

def misc_plots(event_observed, predicted_scores):
    pass

