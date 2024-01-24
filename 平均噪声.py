import tifffile as tiff
import torch
import numpy as np
from skimage import io

# path='./datasets/highSNR/9_550Vx575H_FOV_30Hz_0.2power_00001_highSNR_MC.tif'
# img=tiff.imread(path)
# im = np.zeros((256,512,512))
# a=img.shape[0]
# for p in range(1,257):
#     for z in range(256):
#         if z<p:
#             im[z]=np.mean(img[0:z+p+1,:,:],0)
#         else:
#             im[z]=np.mean(img[z-p:z+p+1,:,:],0)

#     '''
#     for z in range(int(img.shape[0]/p)):
#         if z==int(img.shape[0]/p)-1:
#             im[z]=np.mean(img[z*p:-1,:,:],0)
#         else:
#             im[z]=np.mean(img[z*p:z*p+p,:,:],0)
#     '''
#     #im=im[0:256,:,:]
#     name='./datasets/ground_trues_2/平均噪声'+str(p*2+1)+'张.tif'
#     io.imsave(name,im)
#     print(p*2+1)

path_high = r'./datasets/highSNR/5_550Vx575H_FOV_30Hz_0.2power_00001_highSNR_MC.tif'
img = tiff.imread(path_high)
#im = tiff.imread(path_low)
im = np.zeros((256,512,512))
print(img.shape,im.shape)
p = 14
for z in range(256):
    if z<p:
        im[z]=np.mean(img[0:z+p+1,:,:],0)
    else:
        im[z]=np.mean(img[z-p:z+p+1,:,:],0)

#im=im[0:256,:,:]
path_ave = './datasets/ground_trues_2/5_平均噪声29张.tif'
io.imsave(path_ave,im)


# path_high = r'./datasets/test0/ForTestPlugin_256x256x320.tif'
# img = tiff.imread(path_high)
# #im = tiff.imread(path_low)
# im = np.zeros((256,256,256))
# print(img.shape,im.shape)
# p = 14
# for z in range(256):
#     if z<p:
#         im[z]=np.mean(img[0:z+p+1,:,:],0)
#     else:
#         im[z]=np.mean(img[z-p:z+p+1,:,:],0)

# #im=im[0:256,:,:]
# path_ave = './datasets/ground_trues_2/test0_平均噪声29张.tif'
# io.imsave(path_ave,im)