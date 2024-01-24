import torch
import numpy as np
import tifffile as tiff
from skimage import io


path='./datasets/test/9_550Vx575H_FOV_30Hz_0.2power_00001_lowSNR_MC.tif'
img=tiff.imread(path)
input=np.zeros(shape=(256,64,64),dtype='uint16')
target=np.zeros(shape=(256,64,64),dtype='uint16')
for i in range(256):
    input[i]=img[i*2,0:64,0:64]
    target[i]=img[i*2+1,0:64,0:64]
name_input='./datasets/切片/input.tif'
name_target='./datasets/切片/target.tif'
io.imsave(name_input,input)
io.imsave(name_target,target)
'''
path1='./datasets/切片/input.tif'
path2='./datasets/切片/target.tif'
img1=tiff.imread(path1)
img2=tiff.imread(path2)
print(img1)
print(img2)
'''