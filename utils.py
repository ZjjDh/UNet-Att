import matplotlib.pyplot as plt
import yaml

plt.ioff()
plt.switch_backend('agg')

########################################################################################################################
def create_feature_maps(cut_start_channel_number, number_of_fmaps):
    return [cut_start_channel_number * 2 ** k for k in range(number_of_fmaps)]

def save_yaml(opt, yaml_name):
    para = {'n_epochs':0,
    'datasets_folder':0,
    'GPU':0,
    'output_dir':0,
    'batch_size':0,
    'img_s':0,
    'img_w':0,
    'img_h':0,
    'overlap_h':0,
    'overlap_w':0,
    'overlap_s':0,
    'lr':0,
    'b1':0,
    'b2':0,
    'normalize_factor':0}
    para["n_epochs"] = opt.n_epochs
    para["datasets_folder"] = opt.datasets_folder
    para["GPU"] = opt.GPU
    para["output_dir"] = opt.output_dir
    para["batch_size"] = opt.batch_size
    para["img_s"] = opt.img_s
    para["img_w"] = opt.img_w
    para["img_h"] = opt.img_h
    para["overlap_h"] = opt.overlap_h
    para["overlap_w"] = opt.overlap_w
    para["overlap_s"] = opt.overlap_s
    para["lr"] = opt.lr
    para["b1"] = opt.b1
    para["b2"] = opt.b2
    para["normalize_factor"] = opt.normalize_factor
    para["fmap"] = opt.fmap
    para["datasets_path"] = opt.datasets_path
    para["train_datasets_size"] = opt.train_datasets_size
    with open(yaml_name, 'w') as f:
        data = yaml.dump(para, f)


def read_yaml(opt, yaml_name):
    with open(yaml_name) as f:
        para = yaml.load(f, Loader=yaml.FullLoader)
        print(para)
        opt.n_epochspara = ["n_epochs"]
        # opt.datasets_folder = para["datasets_folder"]
        opt.output_dir = para["output_dir"]
        opt.batch_size = para["batch_size"]
        # opt.img_s = para["img_s"]
        # opt.img_w = para["img_w"]
        # opt.img_h = para["img_h"]
        # opt.overlap_h = para["overlap_h"]
        # opt.overlap_w = para["overlap_w"]
        # opt.overlap_s = para["overlap_s"]
        opt.lr = para["lr"]
        opt.fmap = para["fmap"]
        opt.b1 = para["b1"]
        para["b2"] = opt.b2
        para["normalize_factor"] = opt.normalize_factor


def name2index(opt, input_name, num_h, num_w, num_s):
    # print(input_name)
    name_list = input_name.split('_')
    # print(name_list)
    z_part = name_list[-1]
    # print(z_part)
    y_part = name_list[-2]
    # print(y_part)
    x_part = name_list[-3]
    # print(x_part)
    z_index = int(z_part.replace('z',''))
    y_index = int(y_part.replace('y',''))
    x_index = int(x_part.replace('x',''))
    # print("x_index ---> ",x_index,"y_index ---> ", y_index,"z_index ---> ", z_index)

    cut_w = (opt.img_w - opt.overlap_w)/2
    cut_h = (opt.img_h - opt.overlap_h)/2
    cut_s = (opt.img_s - opt.overlap_s)/2
    # print("z_index ---> ",cut_w, "cut_h ---> ",cut_h, "cut_s ---> ",cut_s)
    if x_index == 0:
        pile_start_w = x_index*opt.overlap_w
        pile_cut_end_w = x_index*opt.overlap_w+opt.img_w-cut_w
        patch_start_w = 0
        patch_cut_end_w = opt.img_w-cut_w
    elif x_index == num_w-1:
        pile_start_w = x_index*opt.overlap_w+cut_w
        pile_cut_end_w = x_index*opt.overlap_w+opt.img_w
        patch_start_w = cut_w
        patch_cut_end_w = opt.img_w
    else:
        pile_start_w = x_index*opt.overlap_w+cut_w
        pile_cut_end_w = x_index*opt.overlap_w+opt.img_w-cut_w
        patch_start_w = cut_w
        patch_cut_end_w = opt.img_w-cut_w

    if y_index == 0:
        pile_start_h = y_index*opt.overlap_h
        pile_cut_end_h = y_index*opt.overlap_h+opt.img_h-cut_h
        patch_start_h = 0
        patch_cut_end_h = opt.img_h-cut_h
    elif y_index == num_h-1:
        pile_start_h = y_index*opt.overlap_h+cut_h
        pile_cut_end_h = y_index*opt.overlap_h+opt.img_h
        patch_start_h = cut_h
        patch_cut_end_h = opt.img_h
    else:
        pile_start_h = y_index*opt.overlap_h+cut_h
        pile_cut_end_h = y_index*opt.overlap_h+opt.img_h-cut_h
        patch_start_h = cut_h
        patch_cut_end_h = opt.img_h-cut_h

    if z_index == 0:
        pile_start_s = z_index*opt.overlap_s
        pile_cut_end_s = z_index*opt.overlap_s+opt.img_s-cut_s
        patch_start_s = 0
        patch_cut_end_s = opt.img_s-cut_s
    elif z_index == num_s-1:
        pile_start_s = z_index*opt.overlap_s+cut_s
        pile_cut_end_s = z_index*opt.overlap_s+opt.img_s
        patch_start_s = cut_s
        patch_cut_end_s = opt.img_s
    else:
        pile_start_s = z_index*opt.overlap_s+cut_s
        pile_cut_end_s = z_index*opt.overlap_s+opt.img_s-cut_s
        patch_start_s = cut_s
        patch_cut_end_s = opt.img_s-cut_s
    return int(pile_start_w) ,int(pile_cut_end_w) ,int(patch_start_w) ,int(patch_cut_end_w) ,\
    int(pile_start_h) ,int(pile_cut_end_h) ,int(patch_start_h) ,int(patch_cut_end_h), \
    int(pile_start_s) ,int(pile_cut_end_s) ,int(patch_start_s) ,int(patch_cut_end_s)

