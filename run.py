import os
import sys

flag = sys.argv[1]

###################################################################################################################################################################
if flag == 'train':
    os.system('python train.py --datasets_folder train \
                               --n_epochs 30 --GPU 0 --batch_size 1 \
                               --img_h 64 --img_w 64 --img_s 512 \
                               --train_datasets_size 500')  

if flag == 'test':
    os.system('python test.py --denoise_model attention_unet++_ --datasets_folder test\
                              --GPU 0 --batch_size 1 \
                              --test_datasize 256')
