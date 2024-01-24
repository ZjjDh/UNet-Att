import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from torch.autograd import Variable

def conv3d(in_channels, out_channels, kernel_size, bias, padding=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=1):
    assert 'c' in order, "Conv layer MUST be present"   
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'  

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of gatchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)  
            modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding)))  
        elif char == 'g':
            is_before_conv = i < order.index('c')  
            assert not is_before_conv, 'GroupNorm MUST go after the Conv3d'  
            if out_channels < num_groups:
                num_groups = out_channels
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)))  
        elif char == 'b':
            is_before_conv = i < order.index('c')  
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules


class SingleConv(nn.Sequential):
    '''
    卷积层和激活层
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, order='cr', num_groups=8, padding=1):   
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=padding):
            self.add_module(name, module)  


class DoubleConv(nn.Sequential):
    '''
    卷积层 + 激活层 + 卷积层 + 激活层
    '''
    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='cr', num_groups=8):
        super(DoubleConv, self).__init__()
        if encoder:   
            # 编码器
            conv1_in_channels = in_channels    
            conv1_out_channels = out_channels // 2    
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:      
            # 解码器
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',  
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups))


class Encoder(nn.Module):
    '''
    编码器，池化-->卷积-->relu-->卷积-->relu
    '''
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=(2, 2, 2), pool_type='max', basic_module=DoubleConv, conv_layer_order='cr',
                 num_groups=8):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)  
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)

    def forward(self, x):  
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x    
        

class Decoder(nn.Module):
    '''
    解码器
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 scale_factor=(2, 2, 2), basic_module=DoubleConv, conv_layer_order='cr', num_groups=8):
        super(Decoder, self).__init__()
        if basic_module == DoubleConv:
            # 插值
            self.upsample = None   
        else:
            # 反卷积
            self.upsample = nn.ConvTranspose3d(in_channels,     
                                               out_channels,
                                               kernel_size=kernel_size,
                                               stride=scale_factor,
                                               padding=1,
                                               output_padding=1)
            in_channels = out_channels

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)

    def forward(self, encoder_features, x):
        if self.upsample is None:
            output_size = encoder_features.size()[2:] 
            # 临近值插值上采样
            x = F.interpolate(x, size=output_size, mode='nearest')  
            # 通道拼接
            x = torch.cat((encoder_features, x), dim=1)   
        else:
            # 使用反卷积进行上采样
            x = self.upsample(x)
            x += encoder_features

        x = self.basic_module(x)  
        return x

class Decoder2(nn.Module):
    '''
    UNet++的解码器
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 scale_factor=(2, 2, 2), basic_module=DoubleConv, conv_layer_order='cr', num_groups=8):
        super(Decoder2, self).__init__()

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)


    def forward(self, encoders_features):
        if len(encoders_features)==4:
            x1=encoders_features[0]
            # 三线性插值上采样
            x2=F.interpolate(encoders_features[1], scale_factor=2, mode='trilinear')
            x = torch.cat((x1,x2), dim=1)
        elif len(encoders_features)==5:
            x1=encoders_features[1]
            x2=F.interpolate(encoders_features[2], scale_factor=2, mode='trilinear')
            x = torch.cat((x1,x2), dim=1)
        elif len(encoders_features)==6:
            x1=encoders_features[2]
            x2=F.interpolate(encoders_features[3], scale_factor=2, mode='trilinear')
            x = torch.cat((x1,x2), dim=1)
        elif len(encoders_features)==7:
            x1=encoders_features[0]
            x2=encoders_features[4]
            x3=F.interpolate(encoders_features[5], scale_factor=2, mode='trilinear')
            x = torch.cat((x1,x2,x3), dim=1)
        elif len(encoders_features)==8:
            x1=encoders_features[1]
            x2=encoders_features[5]
            x3=F.interpolate(encoders_features[6], scale_factor=2, mode='trilinear')
            x = torch.cat((x1,x2,x3), dim=1)
        elif len(encoders_features)==9:
            x1=encoders_features[0]
            x2=encoders_features[4]
            x3=encoders_features[7]
            x4=F.interpolate(encoders_features[8], scale_factor=2, mode='trilinear')
            x = torch.cat((x1,x2,x3,x4), dim=1)
       
        x = self.basic_module(x)  
        return x


class Decoder2_attention_trilinear(nn.Module):
    '''
    attention_unet++的解码器
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 scale_factor=(2, 2, 2), basic_module=DoubleConv, conv_layer_order='cr', num_groups=8):
        super(Decoder2_attention_trilinear, self).__init__()
        
        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)

    def forward(self, encoders_features, attention_features):
        if len(encoders_features)==4:
            x1=attention_features[0]
            # 三线性插值上采样
            x2=F.interpolate(encoders_features[1], scale_factor=2, mode='trilinear', align_corners=True)
            x = torch.cat((x1,x2), dim=1)
        elif len(encoders_features)==5:
            x1=attention_features[1]
            x2=F.interpolate(encoders_features[2], scale_factor=2, mode='trilinear', align_corners=True)
            x = torch.cat((x1,x2), dim=1)
        elif len(encoders_features)==6:
            x1=attention_features[2]
            x2=F.interpolate(encoders_features[3], scale_factor=2, mode='trilinear', align_corners=True)
            x = torch.cat((x1,x2), dim=1)
        elif len(encoders_features)==7:
            x1=attention_features[0]
            x2=attention_features[3]
            x3=F.interpolate(encoders_features[5], scale_factor=2, mode='trilinear', align_corners=True)
            x = torch.cat((x1,x2,x3), dim=1)
        elif len(encoders_features)==8:
            x1=attention_features[1]
            x2=attention_features[4]
            x3=F.interpolate(encoders_features[6], scale_factor=2, mode='trilinear', align_corners=True)
            x = torch.cat((x1,x2,x3), dim=1)
        elif len(encoders_features)==9:
            x1=attention_features[0]
            x2=attention_features[3]
            x3=attention_features[5]
            x4=F.interpolate(encoders_features[8], scale_factor=2, mode='trilinear', align_corners=True)
            x = torch.cat((x1,x2,x3,x4), dim=1)
       
        x = self.basic_module(x)  
        return x


class Decoder2_attention_FA(nn.Module):
    '''
    attention_unet++_FA的解码器
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 scale_factor=(2, 2, 2), basic_module=DoubleConv, conv_layer_order='cr', num_groups=8):
        super(Decoder2_attention_FA, self).__init__()
        
        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)

    def forward(self, encoders_features, attention_features):
        if len(encoders_features)==4:
            x1=encoders_features[0]
            # 三线性插值上采样
            x2=F.interpolate(encoders_features[1], scale_factor=2, mode='trilinear', align_corners=True)
            x = torch.cat((x1,x2), dim=1)
        elif len(encoders_features)==5:
            x1=attention_features[0]
            x2=F.interpolate(encoders_features[2], scale_factor=2, mode='trilinear', align_corners=True)
            x = torch.cat((x1,x2), dim=1)
        elif len(encoders_features)==6:
            x1=attention_features[1]
            x2=F.interpolate(encoders_features[3], scale_factor=2, mode='trilinear', align_corners=True)
            x = torch.cat((x1,x2), dim=1)
        elif len(encoders_features)==7:
            x1=encoders_features[0]
            x2=encoders_features[4]
            x3=F.interpolate(encoders_features[5], scale_factor=2, mode='trilinear', align_corners=True)
            x = torch.cat((x1,x2,x3), dim=1)
        elif len(encoders_features)==8:
            x1=attention_features[0]
            x2=attention_features[2]
            x3=F.interpolate(encoders_features[6], scale_factor=2, mode='trilinear', align_corners=True)
            x = torch.cat((x1,x2,x3), dim=1)
        elif len(encoders_features)==9:
            x1=encoders_features[0]
            x2=encoders_features[4]
            x3=encoders_features[7]
            x4=F.interpolate(encoders_features[8], scale_factor=2, mode='trilinear', align_corners=True)
            x = torch.cat((x1,x2,x3,x4), dim=1)
       
        x = self.basic_module(x)  
        return x

class Decoder2_attention_DA(nn.Module):
    '''
    attention_unet++_DA的解码器
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 scale_factor=(2, 2, 2), basic_module=DoubleConv, conv_layer_order='cr', num_groups=8):
        super(Decoder2_attention_DA, self).__init__()
        
        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)

    def forward(self, encoders_features, attention_features):
        if len(encoders_features)==4:
            x1=attention_features[0]
            x2=F.interpolate(encoders_features[1], scale_factor=2, mode='nearest')
            x = torch.cat((x1,x2), dim=1)
        elif len(encoders_features)==5:
            x1=attention_features[1]
            x2=F.interpolate(encoders_features[2], scale_factor=2, mode='nearest')
            x = torch.cat((x1,x2), dim=1)
        elif len(encoders_features)==6:
            x1=attention_features[2]
            x2=F.interpolate(encoders_features[3], scale_factor=2, mode='nearest')
            x = torch.cat((x1,x2), dim=1)
        elif len(encoders_features)==7:
            x1=attention_features[3]
            x2=encoders_features[4]
            x3=F.interpolate(encoders_features[5], scale_factor=2, mode='nearest')
            x = torch.cat((x1,x2,x3), dim=1)
        elif len(encoders_features)==8:
            x1=attention_features[4]
            x2=encoders_features[5]
            x3=F.interpolate(encoders_features[6], scale_factor=2, mode='nearest')
            x = torch.cat((x1,x2,x3), dim=1)
        elif len(encoders_features)==9:
            x1=attention_features[5]
            x2=encoders_features[4]
            x3=encoders_features[7]
            x4=F.interpolate(encoders_features[8], scale_factor=2, mode='nearest')
            x = torch.cat((x1,x2,x3,x4), dim=1)
       
        x = self.basic_module(x)  
        return x

class Decoder3(nn.Module):
    """
    UNet3+的解码器
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 scale_factor=(2, 2, 2), basic_module=DoubleConv, conv_layer_order='cr', num_groups=8):
        super(Decoder3, self).__init__()
        
        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)


    def forward(self, encoders_features, x):
        if len(encoders_features)==4:
            x1=encoders_features[-2]
            # 临近值插值上采样
            x2=F.interpolate(encoders_features[1], scale_factor=0.5, mode='nearest')
            x3=F.interpolate(encoders_features[0], scale_factor=0.25, mode='nearest')
        elif len(encoders_features)==5:
            x1=F.interpolate(encoders_features[-2], scale_factor=4, mode='nearest')
            x2=encoders_features[1]
            x3=F.interpolate(encoders_features[0], scale_factor=0.5, mode='nearest')
        elif len(encoders_features)==6:
            x1=F.interpolate(encoders_features[-2], scale_factor=4, mode='nearest')
            x2=F.interpolate(encoders_features[-3], scale_factor=8, mode='nearest')
            x3=encoders_features[0]

        x = F.interpolate(x, scale_factor=2, mode='nearest')  
        x = torch.cat((x,x1,x2,x3), dim=1)   
       
        x = self.basic_module(x) 
        return x


class AttentionBlock(nn.Module):  
    '''
    注意力模块
    '''
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=3, mode='concat_attention',
                 sub_sample_factor=(2,2,2)):
        super(AttentionBlock, self).__init__()       

        # 断言数据格式
        assert dimension in [2, 3]  
        assert mode in ['concat_attention', 'concat_attention_add', 'concat_attention_add_2', 'concat_attention_residual']

        # 输入特征图的下采样
        if isinstance(sub_sample_factor, tuple): self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list): self.sub_sample_factor = tuple(sub_sample_factor)
        else: self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # 参数设置
        self.mode = mode
        self.dimension = dimension
        #self.sub_sample_kernel_size = self.sub_sample_factor
        self.sub_sample_kernel_size = (3,3,3)  

        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = 'trilinear'   
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplemented

        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn(self.in_channels),
        )

        self.conv_x = conv_nd(in_channels=self.in_channels, out_channels=self.gating_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0, bias=False)
        self.sig_conv = conv_nd(in_channels=self.gating_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        # 定义模式
        if mode == 'concat_attention':
            self.operation_function = self._concat_attention
        elif mode == 'concat_attention_add':
            self.operation_function = self._concat_attention_add
        elif mode == 'concat_attention_add_2':
            self.operation_function = self._concat_attention_add_2
        elif mode == 'concat_attention_residual':
            self.operation_function = self._concat_attention_residual
        else:
            raise NotImplementedError('Unknown operation function.')

    def forward(self, x, g):
        output = self.operation_function(x, g)
        return output
    
    def _concat_attention(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # conv_x => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        conv_x = self.conv_x(x)
        conv_x_size = conv_x.size()

        g = F.upsample(g, size=conv_x_size[2:], mode=self.upsample_mode)
        # print(conv_x.size(), g.size())
        f = F.relu(conv_x + g, inplace=True)

        # 为图像中的每个像素赋予权重
        sigmoid_f = F.sigmoid(self.sig_conv(f))

        sigmoid_f= F.upsample(sigmoid_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigmoid_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigmoid_f

    def _concat_attention_add(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # conv_x => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        conv_x = self.conv_x(x)
        conv_x_size = conv_x.size()

        g = F.upsample(g, size=conv_x_size[2:], mode=self.upsample_mode)
        f = F.relu(conv_x + g, inplace=True)

        # 为图像中的每个像素赋予权重
        sigmoid_f = torch.sigmoid(self.sig_conv(f))

        sigmoid_f = F.interpolate(sigmoid_f, size=input_size[2:], mode=self.upsample_mode, align_corners=True)
        y = sigmoid_f.expand_as(x) * x

        y = (y+x)/2

        W_y = self.W(y)

        return W_y, sigmoid_f

    def _concat_attention_add_2(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # conv_x => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        conv_x = self.conv_x(x)
        conv_x_size = conv_x.size()

        g = F.upsample(g, size=conv_x_size[2:], mode=self.upsample_mode)
        f = F.relu(conv_x + g, inplace=True)

        # 为图像中的每个像素赋予权重
        sigmoid_f = torch.sigmoid(self.sig_conv(f))

        sigmoid_f = F.interpolate(sigmoid_f, size=input_size[2:], mode=self.upsample_mode, align_corners=True)
        y = sigmoid_f.expand_as(x) * x

        y = y+x

        W_y = self.W(y)

        return W_y, sigmoid_f

    def _concat_attention_residual(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # conv_x => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        conv_x = self.conv_x(x)
        conv_x_size = conv_x.size()

        g = F.upsample(g, size=conv_x_size[2:], mode=self.upsample_mode)
        f = F.relu(conv_x + g, inplace=True)

        f = self.sig_conv(f).view(batch_size, 1, -1)
        sigmoid_f = torch.softmax(f, dim=2).view(batch_size, 1, *conv_x.size()[2:])

        sigmoid_f = F.interpolate(sigmoid_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigmoid_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigmoid_f


class FinalConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, order='cr', num_groups=8):
        super(FinalConv, self).__init__()

        self.add_module('SingleConv', SingleConv(in_channels, in_channels, kernel_size, order, num_groups))

        final_conv = nn.Conv3d(in_channels, out_channels, 1)
        self.add_module('final_conv', final_conv)
