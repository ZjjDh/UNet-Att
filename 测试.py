import cv2
import numpy as np
import math
import tifffile as tiff

path='./datasets/ground_trues/平均噪声29张.tif' 
img=tiff.imread(path)
img=img[0:256,:,:]
img_max=np.max(img)
print(img_max)