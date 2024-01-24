import numpy as np
import os
import tifffile as tiff
import random
import math
import torch
from torch.utils.data import Dataset

class trainset(Dataset):
    '''
    训练集的预处理
    '''
    def __init__(self,name_list,coordinate_list,noise_img):
        self.name_list = name_list
        self.coordinate_list=coordinate_list
        self.noise_img = noise_img

    def __getitem__(self, index):
        #fn = self.images[index]
        cut_slices = self.coordinate_list[self.name_list[index]]
        cut_start_h = cut_slices['cut_start_h']
        cut_end_h = cut_slices['cut_end_h']
        cut_start_w = cut_slices['cut_start_w']
        cut_end_w = cut_slices['cut_end_w']
        cut_start_s = cut_slices['cut_start_s']
        cut_end_s = cut_slices['cut_end_s']
        input = self.noise_img[cut_start_s:cut_end_s:2, cut_start_h:cut_end_h, cut_start_w:cut_end_w]
        target = self.noise_img[cut_start_s + 1:cut_end_s:2, cut_start_h:cut_end_h, cut_start_w:cut_end_w]
        input=torch.from_numpy(np.expand_dims(input, 0))
        target=torch.from_numpy(np.expand_dims(target, 0))
        #target = self.target[index]
        return input, target

    def __len__(self):
        return len(self.name_list)

class testset(Dataset):
    '''
    测试集的预处理
    '''
    def __init__(self,name_list,coordinate_list,noise_img):
        self.name_list = name_list
        self.coordinate_list=coordinate_list
        self.noise_img = noise_img

    def __getitem__(self, index):
        #fn = self.images[index]
        cut_slices = self.coordinate_list[self.name_list[index]]
        cut_start_h = cut_slices['cut_start_h']
        cut_end_h = cut_slices['cut_end_h']
        cut_start_w = cut_slices['cut_start_w']
        cut_end_w = cut_slices['cut_end_w']
        cut_start_s = cut_slices['cut_start_s']
        cut_end_s = cut_slices['cut_end_s']
        noise_patch = self.noise_img[cut_start_s:cut_end_s, cut_start_h:cut_end_h, cut_start_w:cut_end_w]
        noise_patch=torch.from_numpy(np.expand_dims(noise_patch, 0))
        #target = self.target[index]
        return noise_patch,cut_slices

    def __len__(self):
        return len(self.name_list)

def singlebatch_test_save(cut_slices,output_image,raw_image):
    pile_start_w = int(cut_slices['pile_start_w'])
    pile_cut_end_w = int(cut_slices['pile_cut_end_w'])
    patch_start_w = int(cut_slices['patch_start_w'])
    patch_cut_end_w = int(cut_slices['patch_cut_end_w'])

    pile_start_h = int(cut_slices['pile_start_h'])
    pile_cut_end_h = int(cut_slices['pile_cut_end_h'])
    patch_start_h = int(cut_slices['patch_start_h'])
    patch_cut_end_h = int(cut_slices['patch_cut_end_h'])

    pile_start_s = int(cut_slices['pile_start_s'])
    pile_cut_end_s = int(cut_slices['pile_cut_end_s'])
    patch_start_s = int(cut_slices['patch_start_s'])
    patch_cut_end_s = int(cut_slices['patch_cut_end_s'])

    out_a = output_image[patch_start_s:patch_cut_end_s, patch_start_h:patch_cut_end_h, patch_start_w:patch_cut_end_w]
    out_b = raw_image[patch_start_s:patch_cut_end_s, patch_start_h:patch_cut_end_h, patch_start_w:patch_cut_end_w]
    return out_a,out_b,pile_start_w,pile_cut_end_w,pile_start_h,pile_cut_end_h,pile_start_s,pile_cut_end_s


def multibatch_test_save(cut_slices,id,output_image,raw_image):
    pile_start_w_id = cut_slices['pile_start_w'].numpy()
    pile_start_w = int(pile_start_w_id[id])
    pile_cut_end_w_id = cut_slices['pile_cut_end_w'].numpy()
    pile_cut_end_w=int(pile_cut_end_w_id[id])
    patch_start_w_id = cut_slices['patch_start_w'].numpy()
    patch_start_w=int(patch_start_w_id[id])
    patch_cut_end_w_id = cut_slices['patch_cut_end_w'].numpy()
    patch_cut_end_w=int(patch_cut_end_w_id[id])

    pile_start_h_id = cut_slices['pile_start_h'].numpy()
    pile_start_h = int(pile_start_h_id[id])
    pile_cut_end_h_id = cut_slices['pile_cut_end_h'].numpy()
    pile_cut_end_h = int(pile_cut_end_h_id[id])
    patch_start_h_id = cut_slices['patch_start_h'].numpy()
    patch_start_h = int(patch_start_h_id[id])
    patch_cut_end_h_id = cut_slices['patch_cut_end_h'].numpy()
    patch_cut_end_h = int(patch_cut_end_h_id[id])

    pile_start_s_id = cut_slices['pile_start_s'].numpy()
    pile_start_s = int(pile_start_s_id[id])
    pile_cut_end_s_id = cut_slices['pile_cut_end_s'].numpy()
    pile_cut_end_s = int(pile_cut_end_s_id[id])
    patch_start_s_id = cut_slices['patch_start_s'].numpy()
    patch_start_s = int(patch_start_s_id[id])
    patch_cut_end_s_id = cut_slices['patch_cut_end_s'].numpy()
    patch_cut_end_s = int(patch_cut_end_s_id[id])

    output_image_id=output_image[id]
    raw_image_id=raw_image[id]
    out_a = output_image_id[patch_start_s:patch_cut_end_s, patch_start_h:patch_cut_end_h, patch_start_w:patch_cut_end_w]
    out_b = raw_image_id[patch_start_s:patch_cut_end_s, patch_start_h:patch_cut_end_h, patch_start_w:patch_cut_end_w]

    return out_a,out_b,pile_start_w,pile_cut_end_w,pile_start_h,pile_cut_end_h,pile_start_s,pile_cut_end_s

def get_overlap_s(args, img, pile_num):
    overall_w = img.shape[2]
    overall_h = img.shape[1]
    overall_s = img.shape[0]
    # print('overall_w -----> ',overall_w)
    # print('overall_h -----> ',overall_h)
    # print('overall_s -----> ',overall_s)
    w_num = math.floor((overall_w-args.img_w)/args.overlap_w)+1
    h_num = math.floor((overall_h-args.img_h)/args.overlap_h)+1
    s_num = math.ceil(args.train_datasets_size/w_num/h_num/pile_num)
    # print('w_num -----> ',w_num)
    # print('h_num -----> ',h_num)
    # print('s_num -----> ',s_num)
    overlap_s = math.floor((overall_s-args.img_s*2)/(s_num-1))
    # print('overlap_s -----> ',overlap_s)
    return overlap_s

def train_preprocess(args):
    img_h = args.img_h
    img_w = args.img_w
    img_s2 = args.img_s*2
    overlap_h = args.overlap_h
    overlap_w = args.overlap_w
    overlap_s2 = args.overlap_s*2
    im_folder = args.datasets_path + '//' + args.datasets_folder

    name_list = []
    # train_raw = []
    coordinate_list={}

    print('\033[1;31mImage list for training -----> \033[0m')
    pile_num = len(list(os.walk(im_folder, topdown=False))[-1][-1])
    print('Total number -----> ', pile_num)
    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        print(im_name)
        im_dir = im_folder+ '//' + im_name
        noise_im = tiff.imread(im_dir)
        if noise_im.shape[0]>args.select_img_num:
            noise_im = noise_im[0:args.select_img_num,:,:]
        overlap_s2 = get_overlap_s(args, noise_im, pile_num)
        # print('noise_im shape -----> ',noise_im.shape)
        # print('noise_im max -----> ',noise_im.max())
        # print('noise_im min -----> ',noise_im.min())
        noise_im = (noise_im-noise_im.min()).astype(np.float32)/args.normalize_factor

        overall_w = noise_im.shape[2]
        overall_h = noise_im.shape[1]
        overall_s = noise_im.shape[0]
        # print('int((overall_h-img_h+overlap_h)/overlap_h) -----> ',int((overall_h-img_h+overlap_h)/overlap_h))
        # print('int((overall_w-img_w+overlap_w)/overlap_w) -----> ',int((overall_w-img_w+overlap_w)/overlap_w))
        # print('int((overall_s-img_s2+overlap_s2)/overlap_s2) -----> ',int((overall_s-img_s2+overlap_s2)/overlap_s2))
        for x in range(0,int((overall_h-img_h+overlap_h)/overlap_h)):
            for y in range(0,int((overall_w-img_w+overlap_w)/overlap_w)):
                for z in range(0,int((overall_s-img_s2+overlap_s2)/overlap_s2)):
                    cut_slices={'cut_start_h':0, 'cut_end_h':0, 'cut_start_w':0, 'cut_end_w':0, 'cut_start_s':0, 'cut_end_s':0}
                    cut_start_h = overlap_h*x
                    cut_end_h = overlap_h*x + img_h
                    cut_start_w = overlap_w*y
                    cut_end_w = overlap_w*y + img_w
                    cut_start_s = overlap_s2*z
                    cut_end_s = overlap_s2*z + img_s2
                    cut_slices['cut_start_h'] = cut_start_h
                    cut_slices['cut_end_h'] = cut_end_h
                    cut_slices['cut_start_w'] = cut_start_w
                    cut_slices['cut_end_w'] = cut_end_w
                    cut_slices['cut_start_s'] = cut_start_s
                    cut_slices['cut_end_s'] = cut_end_s
                    # noise_patch1 = noise_im[cut_start_s:cut_end_s,cut_start_h:cut_end_h,cut_start_w:cut_end_w]
                    patch_name = args.datasets_folder+'_'+im_name.replace('.tif','')+'_x'+str(x)+'_y'+str(y)+'_z'+str(z)
                    # train_raw.append(noise_patch1.transpose(1,2,0))
                    name_list.append(patch_name)
                    # print(' cut_slices -----> ',cut_slices)
                    coordinate_list[patch_name] = cut_slices
    return  name_list, noise_im, coordinate_list

def shuffle_datasets(name_list):
    index_list = list(range(0, len(name_list)))
    # print('index_list -----> ',index_list)
    random.shuffle(index_list)
    random_index_list = index_list
    # print('index_list -----> ',index_list)
    new_name_list = list(range(0, len(name_list)))
    for i in range(0,len(random_index_list)):
        new_name_list[i] = name_list[random_index_list[i]]
    return new_name_list

def test_preprocess (args, N):
    img_h = args.img_h
    img_w = args.img_w
    img_s2 = args.img_s
    overlap_h = args.overlap_h
    overlap_w = args.overlap_w
    overlap_s2 = args.overlap_s
    cut_w = (img_w - overlap_w)/2
    cut_h = (img_h - overlap_h)/2
    cut_s = (img_s2 - overlap_s2)/2
    im_folder = args.datasets_path+'//'+args.datasets_folder

    name_list = []
    # train_raw = []
    coordinate_list={}
    img_list = list(os.walk(im_folder, topdown=False))[-1][-1]
    img_list.sort()
    # print(img_list)

    im_name = img_list[N]

    im_dir = im_folder+'//'+im_name
    noise_im = tiff.imread(im_dir)
    # print('noise_im shape -----> ',noise_im.shape)
    # print('noise_im max -----> ',noise_im.max())
    # print('noise_im min -----> ',noise_im.min())
    if noise_im.shape[0]>args.test_datasize:
        noise_im = noise_im[0:args.test_datasize,:,:]
    noise_im = (noise_im-noise_im.min()).astype(np.float32)/args.normalize_factor

    overall_w = noise_im.shape[2]
    overall_h = noise_im.shape[1]
    overall_s = noise_im.shape[0]

    num_w = math.ceil((overall_w-img_w+overlap_w)/overlap_w)
    num_h = math.ceil((overall_h-img_h+overlap_h)/overlap_h)
    num_s = math.ceil((overall_s-img_s2+overlap_s2)/overlap_s2)
    # print('int((overall_h-img_h+overlap_h)/overlap_h) -----> ',int((overall_h-img_h+overlap_h)/overlap_h))
    # print('int((overall_w-img_w+overlap_w)/overlap_w) -----> ',int((overall_w-img_w+overlap_w)/overlap_w))
    # print('int((overall_s-img_s2+overlap_s2)/overlap_s2) -----> ',int((overall_s-img_s2+overlap_s2)/overlap_s2))
    for x in range(0,num_h):
        for y in range(0,num_w):
            for z in range(0,num_s):
                cut_slices={'cut_start_h':0, 'cut_end_h':0, 'cut_start_w':0, 'cut_end_w':0, 'cut_start_s':0, 'cut_end_s':0}
                if x != (num_h-1):
                    cut_start_h = overlap_h*x
                    cut_end_h = overlap_h*x + img_h
                elif x == (num_h-1):
                    cut_start_h = overall_h - img_h
                    cut_end_h = overall_h

                if y != (num_w-1):
                    cut_start_w = overlap_w*y
                    cut_end_w = overlap_w*y + img_w
                elif y == (num_w-1):
                    cut_start_w = overall_w - img_w
                    cut_end_w = overall_w

                if z != (num_s-1):
                    cut_start_s = overlap_s2*z
                    cut_end_s = overlap_s2*z + img_s2
                elif z == (num_s-1):
                    cut_start_s = overall_s - img_s2
                    cut_end_s = overall_s
                cut_slices['cut_start_h'] = cut_start_h
                cut_slices['cut_end_h'] = cut_end_h
                cut_slices['cut_start_w'] = cut_start_w
                cut_slices['cut_end_w'] = cut_end_w
                cut_slices['cut_start_s'] = cut_start_s
                cut_slices['cut_end_s'] = cut_end_s

                if y == 0:
                    cut_slices['pile_start_w'] = y*overlap_w
                    cut_slices['pile_cut_end_w'] = y*overlap_w+img_w-cut_w
                    cut_slices['patch_start_w'] = 0
                    cut_slices['patch_cut_end_w'] = img_w-cut_w
                elif y == num_w-1:
                    cut_slices['pile_start_w'] = overall_w-img_w+cut_w
                    cut_slices['pile_cut_end_w'] = overall_w
                    cut_slices['patch_start_w'] = cut_w
                    cut_slices['patch_cut_end_w'] = img_w
                else:
                    cut_slices['pile_start_w'] = y*overlap_w+cut_w
                    cut_slices['pile_cut_end_w'] = y*overlap_w+img_w-cut_w
                    cut_slices['patch_start_w'] = cut_w
                    cut_slices['patch_cut_end_w'] = img_w-cut_w

                if x == 0:
                    cut_slices['pile_start_h'] = x*overlap_h
                    cut_slices['pile_cut_end_h'] = x*overlap_h+img_h-cut_h
                    cut_slices['patch_start_h'] = 0
                    cut_slices['patch_cut_end_h'] = img_h-cut_h
                elif x == num_h-1:
                    cut_slices['pile_start_h'] = overall_h-img_h+cut_h
                    cut_slices['pile_cut_end_h'] = overall_h
                    cut_slices['patch_start_h'] = cut_h
                    cut_slices['patch_cut_end_h'] = img_h
                else:
                    cut_slices['pile_start_h'] = x*overlap_h+cut_h
                    cut_slices['pile_cut_end_h'] = x*overlap_h+img_h-cut_h
                    cut_slices['patch_start_h'] = cut_h
                    cut_slices['patch_cut_end_h'] = img_h-cut_h

                if z == 0:
                    cut_slices['pile_start_s'] = z*overlap_s2
                    cut_slices['pile_cut_end_s'] = z*overlap_s2+img_s2-cut_s
                    cut_slices['patch_start_s'] = 0
                    cut_slices['patch_cut_end_s'] = img_s2-cut_s
                elif z == num_s-1:
                    cut_slices['pile_start_s'] = overall_s-img_s2+cut_s
                    cut_slices['pile_cut_end_s'] = overall_s
                    cut_slices['patch_start_s'] = cut_s
                    cut_slices['patch_cut_end_s'] = img_s2
                else:
                    cut_slices['pile_start_s'] = z*overlap_s2+cut_s
                    cut_slices['pile_cut_end_s'] = z*overlap_s2+img_s2-cut_s
                    cut_slices['patch_start_s'] = cut_s
                    cut_slices['patch_cut_end_s'] = img_s2-cut_s

                # noise_patch1 = noise_im[cut_start_s:cut_end_s,cut_start_h:cut_end_h,cut_start_w:cut_end_w]
                patch_name = args.datasets_folder+'_x'+str(x)+'_y'+str(y)+'_z'+str(z)
                # train_raw.append(noise_patch1.transpose(1,2,0))
                name_list.append(patch_name)
                # print(' cut_slices -----> ',cut_slices)
                coordinate_list[patch_name] = cut_slices

    return  name_list, noise_im, coordinate_list

