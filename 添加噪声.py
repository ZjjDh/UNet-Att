import tifffile as tiff
import torch
import numpy as np
from skimage import io
from skimage import util

# for i in range(1,31):
#     if i<10:
#         path='./results/DataFolderIs_test_unet_30_32/'+'E_0'+str(i)+'_Iter_0993/9_550Vx575H_FOV_30Hz_0.2power_00001_lowSNR_MC_E_0'+str(i)+'_Iter_0993_output.tif'
#     else:
#         path='./results/DataFolderIs_test_unet_30_32/'+'E_'+str(i)+'_Iter_0993/9_550Vx575H_FOV_30Hz_0.2power_00001_lowSNR_MC_E_'+str(i)+'_Iter_0993_output.tif'

#     img=tiff.imread(path)
#     noise_img_1=util.random_noise(img,mode='gaussian',var=0.0001)   #var为方差，也就是标准差的平方，因此需要是0.01, 0.04, 0.09, 0.16
#     noise_img_2=util.random_noise(img,mode='gaussian',var=0.0004)
#     noise_img_3=util.random_noise(img,mode='gaussian',var=0.0009)
#     noise_img_4=util.random_noise(img,mode='gaussian',var=0.0016)

#     noise_img_1=noise_img_1*65535
#     noise_img_1=noise_img_1.astype('uint16')
#     noise_img_2=noise_img_2*65535
#     noise_img_2=noise_img_2.astype('uint16')
#     noise_img_3=noise_img_3*65535
#     noise_img_3=noise_img_3.astype('uint16')
#     noise_img_4=noise_img_4*65535
#     noise_img_4=noise_img_4.astype('uint16')

#     name1='./datasets/noise/unet/'+'E'+str(i)+'N1.tif'
#     io.imsave(name1,noise_img_1)
#     name2='./datasets/noise/unet/'+'E'+str(i)+'N2.tif'
#     io.imsave(name2,noise_img_2)
#     name3='./datasets/noise/unet/'+'E'+str(i)+'N3.tif'
#     io.imsave(name3,noise_img_3)
#     name4='./datasets/noise/unet/'+'E'+str(i)+'N4.tif'
#     io.imsave(name4,noise_img_4)


path_dir = './datasets/test/'
file_name = '9_550Vx575H_FOV_30Hz_0.2power_00001_lowSNR_MC'
img=tiff.imread(path_dir + file_name + '.tif')
img = img[0:1000,:,:]

noise_img_1=util.random_noise(img,mode='gaussian',var=0.0001)   #var为方差，也就是标准差的平方，因此需要是0.01, 0.04, 0.09, 0.16
noise_img_2=util.random_noise(img,mode='gaussian',var=0.0004)
noise_img_3=util.random_noise(img,mode='gaussian',var=0.0009)
noise_img_4=util.random_noise(img,mode='gaussian',var=0.0016)

noise_img_1=noise_img_1*65535
noise_img_1=noise_img_1.astype('uint16')
noise_img_2=noise_img_2*65535
noise_img_2=noise_img_2.astype('uint16')
noise_img_3=noise_img_3*65535
noise_img_3=noise_img_3.astype('uint16')
noise_img_4=noise_img_4*65535
noise_img_4=noise_img_4.astype('uint16')

name1='./datasets/test_noise/' + file_name + '_N1.tif'
io.imsave(name1,noise_img_1)
name2='./datasets/test_noise/' + file_name + '_N2.tif'
io.imsave(name2,noise_img_2)
name3='./datasets/test_noise/' + file_name + '_N3.tif'
io.imsave(name3,noise_img_3)
name4='./datasets/test_noise/' + file_name + '_N4.tif'
io.imsave(name4,noise_img_4)
