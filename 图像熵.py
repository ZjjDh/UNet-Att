
import numpy as np
import math
import tifffile as tiff
import skimage.measure

# tmp = []
# for i in range(65536):
#     tmp.append(0)
# val = 0
# k = 0
# res = 0
# path1='./datasets/ground_trues_2/平均噪声29张.tif'
# img1 = tiff.imread(path1)
# img = img1[97,:,:]

# for i in range(len(img)):
#     for j in range(len(img[i])):
#         val = img[i][j]
#         tmp[val] = float(tmp[val] + 1)
#         k =  float(k + 1)
# for i in range(len(tmp)):
#     tmp[i] = float(tmp[i] / k)
# for i in range(len(tmp)):
#     if(tmp[i] == 0):
#         res = res
#     else:
#         res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
# print(res)

frame = 97
files = ['2_550Vx575H_FOV_30Hz_0.2power_00001_lowSNR_MC', 
         '4_550Vx575H_FOV_30Hz_0.2power_00001_lowSNR_MC', 
         '9_550Vx575H_FOV_30Hz_0.2power_00001_lowSNR_MC', 
         '13_550Vx575H_FOV_30Hz_0.3power_00001_lowSNR_MC']
files_dir = ['test_2', 'test_4', 'test', 'test_13']


entropys_raw = []
for i in range(4):
    path_raw = './datasets/' + files_dir[i] + '/' + files[i] + '.tif'
    img_raw = tiff.imread(path_raw)
    img_raw_slice = img_raw[frame,:,:]
    entropy_raw = skimage.measure.shannon_entropy(img_raw_slice)
    entropys_raw.append(entropy_raw)
print(entropys_raw)

entropys_dncnn = []
for i in range(4):
    path_dncnn = './results/DataFolderIs_' +files_dir[i] + '_dncnn/E_24_Iter_4800/' + files[i] + '_E_24_Iter_4800_output.tif'
    img_dncnn = tiff.imread(path_dncnn)
    img_dncnn_slice = img_dncnn[frame,:,:]
    entropy_dncnn = skimage.measure.shannon_entropy(img_dncnn_slice)
    entropys_dncnn.append(entropy_dncnn)
print(entropys_dncnn)

entropys_n2n = []
for i in range(4):
    path_n2n = './results/DataFolderIs_' +files_dir[i] + '_n2n/E_08_Iter_4800/' + files[i] + '_E_08_Iter_4800_output.tif'
    img_n2n = tiff.imread(path_n2n)
    img_n2n_slice = img_n2n[frame,:,:]
    entropy_n2n = skimage.measure.shannon_entropy(img_n2n_slice)
    entropys_n2n.append(entropy_n2n)
print(entropys_n2n)

entropys_deepCAD = []
for i in range(4):
    path_deepCAD = './results/DataFolderIs_' +files_dir[i] + '_deepCAD_1/E_25_Iter_4800/' + files[i] + '_E_25_Iter_4800_output.tif'
    img_deepCAD = tiff.imread(path_deepCAD)
    img_deepCAD_slice = img_deepCAD[frame,:,:]
    entropy_deepCAD = skimage.measure.shannon_entropy(img_deepCAD_slice)
    entropys_deepCAD.append(entropy_deepCAD)
print(entropys_deepCAD)

entropys_attention_unet2 = []
for i in range(4):
    path_attention_unet2 = './results/DataFolderIs_' +files_dir[i] + '_attention_unet++_v1_1_2_old/E_29_Iter_4800/' + files[i] + '_E_29_Iter_4800_output.tif'
    img_attention_unet2 = tiff.imread(path_attention_unet2)
    img_attention_unet2_slice = img_attention_unet2[frame,:,:]
    entropy_attention_unet2 = skimage.measure.shannon_entropy(img_attention_unet2_slice)
    entropys_attention_unet2.append(entropy_attention_unet2)
print(entropys_attention_unet2)

# entropys_gt = []
# path_gt='./datasets/ground_trues_2/平均噪声29张.tif'
# img_gt = tiff.imread(path_gt)
# for frame in frames:
#     img_gt_slice = img_gt[frame,:,:]
#     entropy_gt = skimage.measure.shannon_entropy(img_gt_slice)
#     entropys_gt.append(entropy_gt)
# print(entropys_gt)



# entropys_raw = []
# path_raw = './datasets/test_4/4_550Vx575H_FOV_30Hz_0.2power_00001_lowSNR_MC.tif'
# img_raw = tiff.imread(path_raw)
# for frame in frames:
#     img_raw_slice = img_raw[frame,:,:]
#     entropy_raw = skimage.measure.shannon_entropy(img_raw_slice)
#     entropys_raw.append(entropy_raw)
# print(entropys_raw)

# entropys_n2n = []
# path_n2n = './results/DataFolderIs_test_4_n2n/E_08_Iter_4800/4_550Vx575H_FOV_30Hz_0.2power_00001_lowSNR_MC_E_08_Iter_4800_output.tif'
# img_n2n = tiff.imread(path_n2n)
# for frame in frames:
#     img_n2n_slice = img_n2n[frame,:,:]
#     entropy_n2n = skimage.measure.shannon_entropy(img_n2n_slice)
#     entropys_n2n.append(entropy_n2n)
# print(entropys_n2n)

# entropys_deepCAD = []
# path_deepCAD = './results/DataFolderIs_test_4_deepCAD_1/E_25_Iter_4800/4_550Vx575H_FOV_30Hz_0.2power_00001_lowSNR_MC_E_25_Iter_4800_output.tif'
# img_deepCAD = tiff.imread(path_deepCAD)
# for frame in frames:
#     img_deepCAD_slice = img_deepCAD[frame,:,:]
#     entropy_deepCAD = skimage.measure.shannon_entropy(img_deepCAD_slice)
#     entropys_deepCAD.append(entropy_deepCAD)
# print(entropys_deepCAD)

# entropys_attention_unet2 = []
# path_attention_unet2 = './results/DataFolderIs_test_4_attention_unet++_v1_1_2_old/E_29_Iter_4800/4_550Vx575H_FOV_30Hz_0.2power_00001_lowSNR_MC_E_29_Iter_4800_output.tif'
# img_attention_unet2 = tiff.imread(path_attention_unet2)
# for frame in frames:
#     img_attention_unet2_slice = img_attention_unet2[frame,:,:]
#     entropy_attention_unet2 = skimage.measure.shannon_entropy(img_attention_unet2_slice)
#     entropys_attention_unet2.append(entropy_attention_unet2)
# print(entropys_attention_unet2)

# entropys_gt = []
# path_gt='./datasets/ground_trues_2/4_平均噪声29张.tif'
# img_gt = tiff.imread(path_gt)
# for frame in frames:
#     img_gt_slice = img_gt[frame,:,:]
#     entropy_gt = skimage.measure.shannon_entropy(img_gt_slice)
#     entropys_gt.append(entropy_gt)
# print(entropys_gt)
