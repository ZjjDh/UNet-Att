import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import datetime
from numpy import *
import csv
from network import Network
from data_process import train_preprocess, shuffle_datasets, trainset
from utils import save_yaml

#############################################################################################################################################
# 参数设置
parser = argparse.ArgumentParser() 

parser.add_argument("--n_epochs", type=int, default=40, help="number of training epochs")
parser.add_argument('--GPU', type=str, default='0,1', help="the index of GPU you will use for computation")

parser.add_argument('--batch_size', type=int, default=2, help="batch size")
parser.add_argument('--img_w', type=int, default=150, help="the width of image sequence")
parser.add_argument('--img_h', type=int, default=150, help="the height of image sequence")
parser.add_argument('--img_s', type=int, default=150, help="the slices of image sequence")

parser.add_argument('--lr', type=float, default=0.00005, help='initial learning rate')  
parser.add_argument("--b1", type=float, default=0.9, help="Adam: bata1")    
parser.add_argument("--b2", type=float, default=0.999, help="Adam: bata2")  
parser.add_argument('--normalize_factor', type=int, default=1, help='normalize factor')
parser.add_argument('--fmap', type=int, default=16, help='number of feature maps')

parser.add_argument('--output_dir', type=str, default='./results', help="output directory")
parser.add_argument('--datasets_folder', type=str, default='train', help="A folder containing files for training")
parser.add_argument('--datasets_path', type=str, default='datasets', help="dataset root path")
parser.add_argument('--pth_path', type=str, default='pth', help="pth file root path")
parser.add_argument('--select_img_num', type=int, default=100000, help='select the number of images used for training')
parser.add_argument('--train_datasets_size', type=int, default=4000, help='datasets size for training')  
opt = parser.parse_args() 

# 数据在切分时有25%的重叠
opt.overlap_s=int(opt.img_s*0.75)   
opt.overlap_w=int(opt.img_w*0.75)
opt.overlap_h=int(opt.img_h*0.75)
opt.ngpu=str(opt.GPU).count(',')+1    
print('\033[1;31mTraining parameters -----> \033[0m') 
print(opt)

########################################################################################################################
# 路径设置
if not os.path.exists(opt.output_dir):   
    os.mkdir(opt.output_dir)   

# 模型保存设置
current_time = opt.datasets_folder+'_'+datetime.datetime.now().strftime("%Y%m%d%H%M")   
output_path = opt.output_dir + '/' + current_time  
pth_dir = 'pth'
if not os.path.exists(pth_dir): 
    os.mkdir(pth_dir)
pth_path = pth_dir+'//'+ current_time   
if not os.path.exists(pth_path): 
    os.mkdir(pth_path)

yaml_name = pth_path+'//para.yaml'
save_yaml(opt, yaml_name)

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.GPU)  
batch_size = opt.batch_size
lr = opt.lr

name_list, noise_img, coordinate_list = train_preprocess(opt)
# print('name_list -----> ',name_list)

########################################################################################################################
# 损失函数
L1_pixelwise = torch.nn.L1Loss()   
L2_pixelwise = torch.nn.MSELoss()  

# 网络
denoise_generator = Network(in_channels = 1, out_channels = 1, fea_maps=opt.fmap, final_sig = True)   

# GPU
if torch.cuda.is_available():
    denoise_generator = denoise_generator.cuda()
    denoise_generator = nn.DataParallel(denoise_generator, device_ids=range(opt.ngpu))
    print('\033[1;31mUsing {} GPU for training -----> \033[0m'.format(torch.cuda.device_count()))
    L2_pixelwise.cuda()
    L1_pixelwise.cuda()
########################################################################################################################
#模型优化器
optimizer_G = torch.optim.Adam(denoise_generator.parameters(),
                                lr=opt.lr, betas=(opt.b1, opt.b2))

#学习率衰减
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, opt.n_epochs*opt.train_datasets_size, eta_min=0, last_epoch=-1)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.99999)   

########################################################################################################################
#判断使用cuda还是cpu
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
prev_time = time.time()

########################################################################################################################
time_start=time.time()

# 开始训练
Total_Loss_mean=list(range(0,opt.n_epochs))   
L1_Loss_mean=list(range(0,opt.n_epochs))   
L2_Loss_mean=list(range(0,opt.n_epochs))   
for epoch in range(0, opt.n_epochs):
    name_list = shuffle_datasets(name_list)   
    train_data = trainset(name_list, coordinate_list, noise_img) 
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    for iteration, (input, target) in enumerate(trainloader):  
        input=input.cuda()
        target = target.cuda()
        real_A=input
        real_B=target
        real_A = Variable(real_A)
        #print('real_A shape -----> ', real_A.shape)
        #print('real_B shape -----> ',real_B.shape)
        fake_B = denoise_generator(real_A)  
        L1_loss = L1_pixelwise(fake_B, real_B)
        L2_loss = L2_pixelwise(fake_B, real_B)
        ################################################################################################################
        print(optimizer_G.param_groups[0]['lr'])
        optimizer_G.zero_grad() 
        # Total loss
        #Total_loss =  L2_loss
        Total_loss =  0.5*L1_loss + 0.5*L2_loss   
        Total_loss.backward()   #反向传播
        optimizer_G.step()
        scheduler.step()  #学习率更新
        ################################################################################################################
        batches_done = epoch * len(trainloader) + iteration   
        batches_left = opt.n_epochs * len(trainloader) - batches_done  
        time_left = datetime.timedelta(seconds=int(batches_left * (time.time() - prev_time))) 
        prev_time = time.time()   

        Total_loss_=[]  
        L1_loss_=[]     
        L2_loss_=[]     
        if iteration%1 == 0:
            time_end=time.time()
            print('\r[Epoch %d/%d] [Batch %d/%d] [Total loss: %.2f, L1 Loss: %.2f, L2 Loss: %.2f] [ETA: %s] [Time cost: %.2d s]        ' 
            % (
                epoch+1,
                opt.n_epochs,
                iteration+1,
                len(trainloader),
                Total_loss.item(),
                L1_loss.item(),
                L2_loss.item(),
                time_left,
                time_end-time_start 
            ), end=' ')             
            Total_loss_.append(Total_loss.item())  
            L1_loss_.append(L1_loss.item())        
            L2_loss_.append(L2_loss.item())         
          
            rows_=zip(Total_loss_,L1_loss_,L2_loss_)
            loss_path=pth_path+'//loss_detail.csv'
            with open(loss_path,'a',newline='') as csvfile:
                writer=csv.writer(csvfile)
                for row_ in rows_:
                    writer.writerow(row_)

        if (iteration+1)%len(trainloader) == 0:
            print('\n', end=' ')
            
        ################################################################################################################
        # 每周期保存一次模型
        if (iteration + 1) % (len(trainloader)) == 0:   
            model_save_name = pth_path + '//E_' + str(epoch+1).zfill(2) + '_Iter_' + str(iteration+1).zfill(4) + '.pth'
            if isinstance(denoise_generator, nn.DataParallel): 
                torch.save(denoise_generator.module.state_dict(), model_save_name) 
            else:
                torch.save(denoise_generator.state_dict(), model_save_name)        
            Total_Loss_mean[epoch]=mean(Total_loss_)   
            L1_Loss_mean[epoch]=mean(L1_loss_)          
            L2_Loss_mean[epoch]=mean(L2_loss_)          
    

# 保存每周期loss
rows=zip(Total_Loss_mean,L1_Loss_mean,L2_Loss_mean)
loss_path_=pth_path+'//loss.csv'
with open(loss_path_,'w',newline='') as csvfile:
    writer=csv.writer(csvfile)
    for row in rows:
        writer.writerow(row)
