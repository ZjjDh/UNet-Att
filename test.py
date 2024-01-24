import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules import module
from torch.utils.data import DataLoader
import argparse
import time
import datetime
from network import Network
import numpy as np
from utils import save_yaml, read_yaml
from data_process import test_preprocess, testset, multibatch_test_save, singlebatch_test_save
from skimage import io

#############################################################################################################################################
# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of training epochs")
parser.add_argument('--GPU', type=str, default='0,1', help="the index of GPU you will use for computation")

parser.add_argument('--batch_size', type=int, default=2, help="batch size")
parser.add_argument('--img_w', type=int, default=64, help="the width of image sequence")
parser.add_argument('--img_h', type=int, default=64, help="the height of image sequence")
parser.add_argument('--img_s', type=int, default=512, help="the slices of image sequence")
parser.add_argument('--overlap_w', type=int, default=48, help='the width of image gap')
parser.add_argument('--overlap_h', type=int, default=48, help='the height of image gap')
parser.add_argument('--overlap_s', type=int, default=384, help='the slices of image gap')

parser.add_argument('--lr', type=float, default=0.00005, help='initial learning rate')
parser.add_argument("--b1", type=float, default=0.9, help="Adam: bata1")
parser.add_argument("--b2", type=float, default=0.999, help="Adam: bata2")
parser.add_argument('--normalize_factor', type=int, default=1, help='normalize factor')
parser.add_argument('--fmap', type=int, default=16, help='number of feature maps')

parser.add_argument('--output_dir', type=str, default='./results', help="output directory")
parser.add_argument('--datasets_path', type=str, default='datasets', help="dataset root path")
parser.add_argument('--pth_path', type=str, default='pth', help="pth file root path")
parser.add_argument('--datasets_folder', type=str, default='test', help="A folder containing files to be tested")
parser.add_argument('--denoise_model', type=str, default='train_model', help='A folder containing models to be tested')
parser.add_argument('--test_datasize', type=int, default=300, help='dataset size to be tested')
parser.add_argument('--train_datasets_size', type=int, default=100000, help='datasets size for training')

opt = parser.parse_args()
opt.ngpu=str(opt.GPU).count(',')+1
print('\033[1;31mParameters -----> \033[0m')
print(opt)
batch_size = opt.batch_size

########################################################################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.GPU)
model_path = opt.pth_path + '//' + opt.denoise_model
# print(model_path)
model_list = list(os.walk(model_path, topdown=False))[-1][-1]
model_list.sort()
# print(model_list)

# 从文件中读取参数
for i in range(len(model_list)):
    aaa = model_list[i]
    if '.yaml' in aaa:
        yaml_name = model_list[i]
        del model_list[i]
# print(yaml_name)
read_yaml(opt, model_path + '//' + yaml_name)
# print(opt.datasets_folder)

# 数据预处理
im_folder = opt.datasets_path + '//' + opt.datasets_folder
img_list = list(os.walk(im_folder, topdown=False))[-1][-1]
img_list.sort()
print('\033[1;31mpiles for processing -----> \033[0m')
print('Total number -----> ', len(img_list))
for img in img_list: print(img)

if not os.path.exists(opt.output_dir):
    os.mkdir(opt.output_dir)
current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
output_path1 = opt.output_dir + '//' + 'DataFolderIs_' + opt.datasets_folder + '_' + current_time + '_ModelFolderIs_' + opt.denoise_model
if not os.path.exists(output_path1):
    os.mkdir(output_path1)

yaml_name = output_path1 + '//para.yaml'
save_yaml(opt, yaml_name)

##############################################################################################################################################################
# 网络架构和GPU设置
denoise_generator = Network(in_channels=1,out_channels=1,fea_maps=opt.fmap,final_sig=True)

if torch.cuda.is_available():
    print('\033[1;31mUsing {} GPU for testing -----> \033[0m'.format(torch.cuda.device_count()))
    denoise_generator = denoise_generator.cuda()
    denoise_generator = nn.DataParallel(denoise_generator, device_ids=range(opt.ngpu))
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
##############################################################################################################################################################

# 开始处理
for pth_index in range(len(model_list)):
    aaa = model_list[pth_index]
    if '.pth' in aaa:
        pth_name = model_list[pth_index]
        output_path = output_path1 + '//' + pth_name.replace('.pth', '')
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # 加载模型
        model_name = opt.pth_path + '//' + opt.denoise_model + '//' + pth_name
        if isinstance(denoise_generator, nn.DataParallel):
            #model=Network_3D_Unet().cuda()
            #state_dict = torch.load(model_name, map_location="gpu")
            #torch.save(state_dict, model_name ,_use_new_zipfile_serialization=False)
            denoise_generator.module.load_state_dict(torch.load(model_name))  # parallel
            denoise_generator.eval()
        else:
            denoise_generator.load_state_dict(torch.load(model_name))  # not parallel
            denoise_generator.eval()
        denoise_generator.cuda()

        # 预测所有堆栈
        for N in range(len(img_list)):
            name_list, noise_img, coordinate_list = test_preprocess(opt, N)
            #print(len(name_list))
            prev_time = time.time()
            time_start = time.time()
            denoise_img = np.zeros(noise_img.shape)
            input_img = np.zeros(noise_img.shape)

            test_data = testset(name_list, coordinate_list, noise_img)
            testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
            for iteration, (noise_patch,cut_slices) in enumerate(testloader):
                noise_patch=noise_patch.cuda()
                real_A = noise_patch
                #print('real_A -----> ',real_A.shape)
                #input_name = name_list[index]
                # print(' input_name -----> ',input_name)
                #print(' cut_slices -----> ',cut_slices)
                # print('real_A -----> ',real_A.shape)
                real_A = Variable(real_A)
                fake_B = denoise_generator(real_A)
                ################################################################################################################
                # 计算模型预测剩余时间
                batches_done = iteration
                batches_left = 1 * len(testloader) - batches_done
                time_left_seconds = int(batches_left * (time.time() - prev_time))
                time_left = datetime.timedelta(seconds=time_left_seconds)
                prev_time = time.time()
                ################################################################################################################
                if iteration % 1 == 0:
                    time_end = time.time()
                    time_cost = time_end - time_start  # datetime.timedelta(seconds= (time_end - time_start))
                    print(
                        '\r[Model %d/%d, %s] [pile %d/%d, %s] [Patch %d/%d] [Time Cost: %.0d s] [ETA: %s s]     '
                        % (
                            pth_index + 1,
                            len(model_list),
                            pth_name,
                            N + 1,
                            len(img_list),
                            img_list[N],
                            iteration + 1,
                            len(testloader),
                            time_cost,
                            time_left_seconds
                        ), end=' ')

                if (iteration + 1) % len(testloader) == 0:
                    print('\n', end=' ')
                ################################################################################################################
                output_image = np.squeeze(fake_B.cpu().detach().numpy())
                raw_image = np.squeeze(real_A.cpu().detach().numpy())
                if(output_image.ndim==3):
                    turn=1
                else:
                    turn=output_image.shape[0]
                #print(turn)
                if(turn>1):
                   for id in range(turn):
                      #print('shape of output_image -----> ',output_image.shape)
                      out_a,out_b,pile_start_w,pile_cut_end_w,pile_start_h,pile_cut_end_h,pile_start_s,pile_cut_end_s=multibatch_test_save(cut_slices,id,output_image,raw_image)
                      denoise_img[pile_start_s:pile_cut_end_s, pile_start_h:pile_cut_end_h, pile_start_w:pile_cut_end_w] \
                       = out_a * (np.sum(out_b) / np.sum(out_a)) ** 0.5
                      input_img[pile_start_s:pile_cut_end_s, pile_start_h:pile_cut_end_h, pile_start_w:pile_cut_end_w] \
                       = out_b
                else:
                    out_a, out_b, pile_start_w, pile_cut_end_w, pile_start_h, pile_cut_end_h, pile_start_s, pile_cut_end_s = singlebatch_test_save(
                        cut_slices, output_image, raw_image)
                    denoise_img[pile_start_s:pile_cut_end_s, pile_start_h:pile_cut_end_h, pile_start_w:pile_cut_end_w] \
                        = out_a * (np.sum(out_b) / np.sum(out_a)) ** 0.5
                    input_img[pile_start_s:pile_cut_end_s, pile_start_h:pile_cut_end_h, pile_start_w:pile_cut_end_w] \
                        = out_b

            output_img = denoise_img.squeeze().astype(np.float32) * opt.normalize_factor
            del denoise_img
            # output_img = output_img1[0:raw_noise_img.shape[0],0:raw_noise_img.shape[1],0:raw_noise_img.shape[2]]
            output_img = output_img - output_img.min()
            output_img = output_img / output_img.max() * 65535
            output_img = np.clip(output_img, 0, 65535).astype('uint16')
            output_img = output_img - output_img.min()
            # output_img = output_img.astype('uint16')

            result_name = output_path + '//' + img_list[N].replace('.tif', '') + '_' + pth_name.replace('.pth', '') + '_output.tif'
            #io.imsave(result_name, output_img, check_contrast=False)
            io.imsave(result_name, output_img)


