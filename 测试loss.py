import torch
import numpy as np
import tifffile as tiff

path='./datasets/test/9_550Vx575H_FOV_30Hz_0.2power_00001_lowSNR_MC.tif'
img=tiff.imread(path)
img = (img-img.min()).astype(np.float32)

path_output='./results/DataFolderIs_test_pth_unet_24_3900/E_24_Iter_3900/9_550Vx575H_FOV_30Hz_0.2power_00001_lowSNR_MC_E_24_Iter_3900_output.tif'
outimg=tiff.imread(path_output)
outimg = (outimg-outimg.min()).astype(np.float32)

#测试img与outimg的loss
'''
输出
820.25439453125 756561.1875 378690.71875
1245.9580078125 1869032.875 935139.4375
1383.661865234375 2538790.75 1270087.25
1478.470947265625 2467861.5 1234670.0
1274.919189453125 1838867.0 920070.9375
1347.049072265625 2295351.75 1148349.375
1074.862548828125 1291290.0 646182.4375
566.7265625 351678.8125 176122.765625
'''
for p in range(8):

    for i in range(256):
        img_=img[i*2+1,p*64:p*64+64,p*64:p*64+64]
        outimg_=outimg[i,p*64:p*64+64,p*64:p*64+64]
        img_=torch.from_numpy(np.expand_dims(img_, 0))
        outimg_=torch.from_numpy(np.expand_dims(outimg_, 0))

    img_=img_.cuda()
    outimg_ = outimg_.cuda()

    L1_pixelwise = torch.nn.L1Loss()   
    L2_pixelwise = torch.nn.MSELoss()
    L1_loss = L1_pixelwise(img_.float(),outimg_.float())
    L2_loss = L2_pixelwise(img_.float(),outimg_.float())
    Total_loss =  0.5*L1_loss + 0.5*L2_loss
    print(L1_loss.item(), L2_loss.item(), Total_loss.item())

'''
#测试input与target之间的loss
#输出125.224853515625 61184.2265625 30654.7265625
188.8955078125 107419.2109375 53804.0546875
304.22265625 268416.34375 134360.28125
261.830078125 188594.59375 94428.2109375
228.85595703125 158284.328125 79256.59375
243.473388671875 206942.421875 103592.9453125
186.27783203125 120375.921875 60281.1015625
49.902099609375 17930.572265625 8990.2373046875
for p in range(8):

    for i in range(256):
        input=img[i*2,p*64:p*64+64,p*64:p*64+64]
        target=img[i*2+1,p*64:p*64+64,p*64:p*64+64]
        input=torch.from_numpy(np.expand_dims(input, 0))
        target=torch.from_numpy(np.expand_dims(target, 0))

    #img1=torch.from_numpy(input)
    #img2=torch.from_numpy(target)

    L1_pixelwise = torch.nn.L1Loss()   
    L2_pixelwise = torch.nn.MSELoss()
    L1_loss = L1_pixelwise(input.float(),target.float())
    L2_loss = L2_pixelwise(input.float(),target.float())
    Total_loss =  0.5*L1_loss + 0.5*L2_loss
    print(L1_loss.item(), L2_loss.item(), Total_loss.item())
'''