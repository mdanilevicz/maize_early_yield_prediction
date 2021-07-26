
from fastai.vision.all import *
import fastai
from fastai.tabular.all import *
from fastai.data.load import _FakeLoader, _loaders
from glob import glob

import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import os

# Custom functions
from msi_utils import *
from fold_utils import * 

class MixedDL():
    def __init__(self, tab_dl:TabDataLoader, vis_dl:TfmdDL, device='cuda:0'):
        "Stores away `tab_dl` and `vis_dl`, and overrides `shuffle_fn`"
        self.device = device
        tab_dl.shuffle_fn = self.shuffle_fn
        vis_dl.shuffle_fn = self.shuffle_fn
        
        self.dls = [tab_dl, vis_dl]
        self.count = 0
        self.fake_l = _FakeLoader(self, False, 0, 0, 1)
    
    def __len__(self): return len(self.dls[0])
        
    def shuffle_fn(self, idxs):
        "Generates a new `rng` based upon which `DataLoader` is called"
        if self.count == 0: # if we haven't generated an rng yet
            self.rng = self.dls[0].rng.sample(idxs, len(idxs))
            self.count += 1
            return self.rng
        else:
            self.count = 0
            return self.rng
        
    def to(self, device): self.device = device
        
        
@patch
def __iter__(dl:MixedDL):
    "Iterate over your `DataLoader`"
    z = zip(*[_loaders[i.fake_l.num_workers==0](i.fake_l) for i in dl.dls])
    for b in z:
        if dl.device is not None: 
            b = to_device(b, dl.device)
        batch = []
        batch.extend(dl.dls[0].after_batch(b[0])[:2]) # tabular cat and cont
        batch.append(dl.dls[1].after_batch(b[1][0])) # Image
#         try: # In case the data is unlabelled
        batch.append(b[1][1]) # y
        yield tuple(batch)
@patch
def one_batch(x:MixedDL):
    "Grab a batch from the `DataLoader`"
    with x.fake_l.no_multiproc(): res = first(x)
    if hasattr(x, 'it'): delattr(x, 'it')
    return res

@patch
def show_batch(x:MixedDL, channels=None):
    "Show a batch from multiple `DataLoaders`"
    for dl in x.dls:
        # added this test because my custom class MSI needs the argument channels to be passed in the show_batch fn
        if type(dl) == TabDataLoader:
            dl.show_batch()
        else:
            # type(dl) == TfmdDL
            dl.show_batch(channels=3)



