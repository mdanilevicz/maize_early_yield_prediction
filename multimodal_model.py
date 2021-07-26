# +
# Import the libraries
from fastai.vision.all import *
import fastai
from fastai.tabular.all import *
from fastai.data.load import _FakeLoader, _loaders
from glob import glob
import torch
import pandas as pd
import numpy as np
import os

# Custom functions
from msi_utils import *
from fold_utils import * 
from multimodal_utisl import *
# -


global glb_tab_logits
def get_tab_logits(self, inp, out):
    global glb_tab_logits
    glb_tab_logits = inp

global glb_vis_logits
def get_vis_logits(self, inp, out):
    global glb_vis_logits
    glb_vis_logits = inp


class TabVis(nn.Module):
    # Modify the architecture here if you want more or less layers at the fusion module
    def __init__(self, tab_model, vis_model, num_classes=1): 
        super(TabVis, self).__init__()
        self.tab_model = tab_model
        self.vis_model = vis_model
        
        # Add the fusion module
        self.mixed_reg = nn.Sequential(nn.Linear(612,612),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(612, num_classes))
        
        # receive the weights from tab and spectral modules
        self.tab_reg = nn.Linear(100, num_classes)
        self.vis_reg = nn.Linear(512, num_classes)
        
        # register hook that will grab the module's weights
        self.tab_handle = self.tab_model.layers[2][0].register_forward_hook(get_tab_logits)
        self.vis_handle = self.vis_model[11].register_forward_hook(get_vis_logits)
   
    def remove_my_hooks(self):
        self.tab_handle.remove()
        self.vis_handle.remove()
        return None    
        
    def forward(self, x_cat, x_cont, x_im):
        # Tabular Regressor
        tab_pred = self.tab_model(x_cat, x_cont) 
        # Spectral Regressor
        vis_pred = self.vis_model(x_im)
        # Logits
        tab_logits = glb_tab_logits[0]   # Only grabbling weights, not bias'
        vis_logits = glb_vis_logits[0]   # Only grabbling weights, not bias'
        mixed = torch.cat((tab_logits, vis_logits), dim=1)
        # Mixed classifier block
        mixed_pred = self.mixed_reg(mixed) 
        return (tab_pred, vis_pred, mixed_pred)

class GradientBlending(nn.Module):
    def __init__(self, tab_weight=0.0, visual_weight=0.0, tab_vis_weight=1.0, loss_scale=1.0):
        "Expects weights for each model, the combined model, and an overall scale"
        super(myGradientBlending, self).__init__()
        self.tab_weight = tab_weight
        self.visual_weight = visual_weight
        self.tab_vis_weight = tab_vis_weight
        self.scale = loss_scale
        
    def remove_my_hooks(self):
        self.tab_handle.remove()
        self.vis_handle.remove()
        #self.print_handle.remove()
        return None
        
    def forward(self, xb, yb):
        tab_out, visual_out, tv_out = xb
        targ = yb
        
        # Add some hook here to log the modules losses in a csv
        "Gathers `self.loss` for each model, weighs, then sums"
        t_loss = root_mean_squared_error(tab_out, targ) * self.scale
        v_loss = root_mean_squared_error(visual_out, targ) * self.scale
        tv_loss = root_mean_squared_error(tv_out, targ) * self.scale

        weighted_t_loss = t_loss * self.tab_weight
        weighted_v_loss = v_loss * self.visual_weight
        weighted_tv_loss = tv_loss * self.tab_vis_weight
        
        loss = weighted_t_loss + weighted_v_loss + weighted_tv_loss
        return loss

# Metrics
def t_rmse(inp, targ):
    "Compute rmse with `targ` and `pred`"
    pred = inp[0].flatten()
    return root_mean_squared_error(*flatten_check(pred,targ))

def v_rmse(inp, targ):
    "Compute rmse with `targ` and `pred`"
    pred = inp[1].flatten()
    return root_mean_squared_error(*flatten_check(pred,targ))

def tv_rmse(inp, targ):
    "Compute rmse with `targ` and `pred`"
    pred = inp[2].flatten()
    return root_mean_squared_error(*flatten_check(pred,targ))

def weighted_RMSEp(inp, targ, w_t=0.333, w_v=0.333, w_tv=0.333):
    # normalised by the max -min
    delta = df['Yield'].max() - df['Yield'].min()
    tv_inp = (inp[2].flatten()) 
    rmsep = root_mean_squared_error(*flatten_check(tv_inp,targ)) / delta    
    return rmsep * 100
