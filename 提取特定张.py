import tifffile as tiff
import torch
import numpy as np
from skimage import io

path='./datasets/test/9_550Vx575H_FOV_30Hz_0.2power_00001_lowSNR_MC.tif'
img=tiff.imread(path)
im=[]
for z in range(0,256):
    im_=img[z*20,:,:]
    im.append(im_)
im=np.array(im)
name='每20帧提取一张.tif'
io.imsave(name,im)