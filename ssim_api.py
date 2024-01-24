import skimage.metrics
import numpy as np
import tifffile as tiff




path1='./datasets/ground_trues_2/平均噪声29张.tif'
img1 = tiff.imread(path1)
img1 = img1[14:256,:,:]

#path2='./datasets/test/9_550Vx575H_FOV_30Hz_0.2power_00001_lowSNR_MC.tif'

# path2='./datasets/test/9_550Vx575H_FOV_30Hz_0.2power_00001_lowSNR_MC.tif'
# img2 = tiff.imread(path2)
# img2 = img2[14:256,:,:]


print('n2n_2')

for i in range(1,31):
    if i<10:
        path2='./results/DataFolderIs_test_n2n_2/'+'E_0'+str(i)+'_Iter_4800/9_550Vx575H_FOV_30Hz_0.2power_00001_lowSNR_MC_E_0'+str(i)+'_Iter_4800_output.tif'
    else:
        path2='./results/DataFolderIs_test_n2n_2/'+'E_'+str(i)+'_Iter_4800/9_550Vx575H_FOV_30Hz_0.2power_00001_lowSNR_MC_E_'+str(i)+'_Iter_4800_output.tif'
   
    #img=img*255/img_max
    outimg=tiff.imread(path2)
    outimg=outimg[14:256,:,:]

    ssim = skimage.metrics.structural_similarity(
        img1.astype(np.float), outimg.astype(np.float), multichannel=True, data_range=65535)

    print(ssim)

    
