from fastai.vision.all import *
import fastai
from fastai.tabular.all import *
from fastai.data.load import _FakeLoader, _loaders
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import random


# CUSTOM VIS DATABLOCK FUNCTIONS
def get_npy(dataframe):
    "Get the images (.npy) that will be used as input for the model"
    # get sample names from the dataframe
    samples = dataframe['Barcode'] 
    fnames = []
    # for each sample in the dataframe
    for sp in samples:
        img_getter = lambda x: path/f'images/{sp}.npy'
        fnames.append(img_getter(sp))
    # returns a list of the image paths
    return fnames

def get_y(fname):
    "Get the target yield value"
    fname = str(fname)
    fname = fname.split(sep='/')[-1]
    fname = fname.replace('.npy', '')
    y_target = mixed_df[mixed_df["Barcode"] == fname]
    y_target = float(y_target['Yield'])
    return y_target

def mix_npy_blocks(img):
    "This function will be used to build the plot image and add transforms"
    # Cut the image in half and stack the chunks side-by-side
    chunk0 = img[:40, :20, :]
    chunk1 = img[40:80, :20, :]  

    if random.choice([True,False]):
        chunk0 = np.flip(chunk0[:,:,:], axis=0) # Flip vertically equals img[X,:,:]
    if random.choice([True,False]):
        chunk1 = np.flip(chunk1[:,:,:], axis=0) # Flip vertically equals img[X,:,:]
    if random.choice([True,False]):
        chunk0 = np.flip(chunk0[:,:,:], axis=1) # Flip horizontally equals img[:,X,:]
    if random.choice([True,False]):
        chunk1 = np.flip(chunk1[:,:,:], axis=1) # Flip horizontally equals img[:,X,:]

    if random.choice([True,False]):
        new_img = np.hstack((chunk0, chunk1))
    else:
        new_img =np.hstack((chunk1, chunk0))
    
    return  new_img

def vegetation_idxs(img):
    "Calculate VI and add as new bands"
    e = 0.00015 # Add a small value to avoid division by zero
    im = img
    
    # Calculate the VIs - change to np functions
    ndvi = np.divide(np.subtract(im[:,:,4], im[:,:,2]), (np.add(im[:,:,4], im[:,:,2])+e))
    ndvi_re = (im[:,:,4] - im[:,:,3]) / ((im[:,:,4] + im[:,:,3]) + e)
    ndre = (im[:,:,3] - im[:,:,2]) / ((im[:,:,3] + im[:,:,3]) + e) 
    envi = ((im[:,:,4] + im[:,:,1]) - (2 * im[:,:,0])) / (((im[:,:,4] - im[:,:,1]) + (2 * im[:,:,0])) + e)
    ccci = ndvi_re / (ndvi + e)
    gndvi = (im[:,:,4] - im[:,:,1])/ ((im[:,:,4] + im[:,:,1]) + e)
    gli = ((2* im[:,:,1]) - im[:,:,0] - im[:,:,2]) / (((2* im[:,:,1]) + im[:,:,0] + im[:,:,2]) + e)
    osavi = ((im[:,:,4] - im[:,:,3])/ ((im[:,:,4] + im[:,:,3] + 0.16)) *(1 + 0.16) + e)
    
    vi_list = [ndvi, ndvi_re, ndre, envi, ccci, gndvi , gli, osavi]
    vis = np.zeros((40,40,13)) 
    
    vis_stacked = np.stack(vi_list, axis=2)
    vis[:,:,:5] = im
    vis[:,:,5:] = vis_stacked
    
    return vis

def load_npy(fn):
    im = np.load(str(fn), allow_pickle=True)
    im = im*3 # increase image signal
    
    # Padding with zeros
    w, h , c = im.shape
    im = np.pad(im, ((0, 100-w), (0, 100-h), (0,0)),mode='constant', constant_values=0)    
    im = mix_npy_blocks(im) # Add transforms and stacking
    im = vegetation_idxs(im) # Add vegetation indexes bands
    # Normalise bands by deleting no-data values
    for band in range(13):
        im[:,:,band] = np.clip(im[:,:,band], 0, 1)
    
    # Swap axes because np is:  width, height, channels
    # and torch wants        :  channel, width , height
    im = np.swapaxes(im, 2, 0)
    im = np.swapaxes(im, 1, 2) 
    im = np.nan_to_num(im)
    return torch.from_numpy(im)

class MSITensorImage(TensorImage):
    _show_args = {'cmap':'Rdb'}
    
    def show(self, channels=3, ctx=None, vmin=None, vmax=None, **kwargs):
        "Visualise the images"
        if channels == 3 :
            return show_composite(self, 3, ctx=ctx, **{**self._show_args, **kwargs}) 
    
        else:
            return show_single_channel(self, channels, ctx=ctx, **{**self._show_args, **kwargs} )
    
    @classmethod
    def create(cls, fn:(Path, str), **kwargs) -> None:
        " Uses the load fn the array and turn into tensor"
        return cls(load_npy(fn))
        
    def __repr__(self): return f'{self.__class__.__name__} size={"x".join([str(d) for d in self.shape])}'

def MSITensorBlock(cls=MSITensorImage):
    " A `TransformBlock` for numpy array images"
    # Calls the class create function to transform the x input using custom functions
    return TransformBlock(type_tfms=cls.create, batch_tfms=None)

def root_mean_squared_error(p, y): 
    return torch.sqrt(F.mse_loss(p.view(-1), y.view(-1)))

def create_rgb(img):
    # make RGB plot to visualise the "show batch"
    RGB = np.zeros((3, 40, 40))
    RGB[0] = img[2]
    RGB[2] = img[0]
    RGB[1] = img[1]
    #Change from tensor format to pyplot
    RGB = np.swapaxes(RGB, 0, 2)
    RGB = np.swapaxes(RGB, 1, 0)
    RGB = RGB 
    return RGB

def show_composite(img, channels, ax=None,figsize=(3,3), title=None, scale=True,
                   ctx=None, vmin=0, vmax=1, scale_axis=(0,1), **kwargs)->plt.Axes:
    "Show three channel composite"
    ax = ifnone(ax, ctx)
    dims = img.shape[0]
    RGBim = create_rgb(img)
    ax.imshow(RGBim)
    ax.axis('off')
    if title is not None: ax.set_title(title)
    return ax

def show_single_channel(img, channel, ax=None, figsize=(3,3), ctx=None, 
                        title=None, **kwargs) -> plt.Axes:
    ax = ifnone(ax, ctx)
    if ax is None: _, ax = plt.subplots(figsize=figsize)    
    
    tempim = img.data.cpu().numpy()
    
    if tempim.ndim >2:
        ax.imshow(tempim[channel,:,:])
        ax.axis('off')
        if title is not None: ax.set_title(f'{fname} with {title}')
    else:
        ax.imshow(tempim)
        ax.axis('off')
        if title is not None: ax.set_title(f'{fname} with {title}')
        
    return ax
