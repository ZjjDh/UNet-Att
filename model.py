import importlib

import torch
import torch.nn as nn

from modules import Encoder, Decoder, Decoder2, Decoder2_attention_trilinear, Decoder2_attention_FA, Decoder2_attention_DA, Decoder3, FinalConv, DoubleConv, SingleConv, AttentionBlock
from utils import create_feature_maps

class UNet2_attention(nn.Module):
    '''
    在UNet++网络中加入注意力模块，注意力模块分布在所有的跳跃连接
    '''
    def __init__(self, in_channels, out_channels, final_sig, fea_maps=64, layer_order='cr', num_groups=8,
                 **kwargs):
        super(UNet2_attention, self).__init__()

        if isinstance(fea_maps, int):
            # 使用四层特征图
            fea_maps = create_feature_maps(fea_maps, number_of_fmaps=4)

        # 创建由编码器模块组成的解码器路径
        encoders = []
        for i, out_feature_num in enumerate(fea_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(fea_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # 创建由解码器模块组成的解码器路径，以及注意力模块组成的注意力模块路径
        decoders = []
        attentionblocks = []
        attentionblock_11 = AttentionBlock(in_channels=fea_maps[0], gating_channels=fea_maps[1],
                                                    inter_channels=fea_maps[0], sub_sample_factor=(2,2,2), mode='concat_attention')
        attentionblocks.append(attentionblock_11)
        d_11=Decoder2_attention_trilinear(fea_maps[0]+fea_maps[1], fea_maps[0], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_11)

        attentionblock_21 = AttentionBlock(in_channels=fea_maps[1], gating_channels=fea_maps[2],
                                                    inter_channels=fea_maps[1], sub_sample_factor=(2,2,2), mode='concat_attention')
        attentionblocks.append(attentionblock_21)
        d_21=Decoder2_attention_trilinear(fea_maps[1]+fea_maps[2], fea_maps[1], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_21)

        attentionblock_31 = AttentionBlock(in_channels=fea_maps[2], gating_channels=fea_maps[3],
                                                    inter_channels=fea_maps[2], sub_sample_factor=(2,2,2), mode='concat_attention')
        attentionblocks.append(attentionblock_31)
        d_31=Decoder2_attention_trilinear(fea_maps[2]+fea_maps[3], fea_maps[2], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_31)

        attentionblock_12 = AttentionBlock(in_channels=fea_maps[0], gating_channels=fea_maps[1],
                                                    inter_channels=fea_maps[0], sub_sample_factor=(2,2,2), mode='concat_attention')
        attentionblocks.append(attentionblock_12)
        d_12=Decoder2_attention_trilinear(fea_maps[1]+fea_maps[0]+fea_maps[0], fea_maps[0], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_12)

        attentionblock_22 = AttentionBlock(in_channels=fea_maps[1], gating_channels=fea_maps[2],
                                                    inter_channels=fea_maps[1], sub_sample_factor=(2,2,2), mode='concat_attention')
        attentionblocks.append(attentionblock_22)
        d_22=Decoder2_attention_trilinear(fea_maps[2]+fea_maps[1]+fea_maps[1], fea_maps[1], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_22)

        attentionblock_13 = AttentionBlock(in_channels=fea_maps[0], gating_channels=fea_maps[1],
                                                    inter_channels=fea_maps[0], sub_sample_factor=(2,2,2), mode='concat_attention')
        attentionblocks.append(attentionblock_13)
        d_13=Decoder2_attention_trilinear(fea_maps[1]+fea_maps[0]*3, fea_maps[0], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_13)

        self.decoders = nn.ModuleList(decoders)
        self.attentionblocks = nn.ModuleList(attentionblocks)

        # 最后一层使用1×1卷积层融合所有通道信息
        self.final_conv = nn.Conv3d(fea_maps[0], out_channels, 1)

        if final_sig:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # 编码器部分，构建所有网络层的路径
        encoders_decodes_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_decodes_features.append(x)

        # 解码器部分
        d_i=0
        final_=[]
        attention_features = []
        for decoder, attentionblock in zip(self.decoders,self.attentionblocks):
            if d_i < 3:
                attention_x, sig_out = attentionblock(encoders_decodes_features[d_i], encoders_decodes_features[d_i+1])
                attention_features.append(attention_x)
            elif 3 <= d_i <= 4:
                attention_x, sig_out = attentionblock(encoders_decodes_features[d_i+1], encoders_decodes_features[d_i+2])
                attention_features.append(attention_x)
            else:
                attention_x, sig_out = attentionblock(encoders_decodes_features[d_i+2], encoders_decodes_features[d_i+3])
                attention_features.append(attention_x)
            x = decoder(encoders_decodes_features, attention_features)
            encoders_decodes_features.append(x)   
            if d_i==0 or d_i==3 or d_i==5:
                final_i=self.final_conv(x)
                final_.append(final_i)
            d_i+=1

        x = (final_[0]+final_[1]+final_[2])/3
        return x


class UNet(nn.Module):
    '''
    UNet
    '''
    def __init__(self, in_channels, out_channels, final_sig, fea_maps=64, layer_order='cr', num_groups=8,
                 **kwargs):
        super(UNet, self).__init__()

        if isinstance(fea_maps, int):
            # 使用四层特征图
            fea_maps = create_feature_maps(fea_maps, number_of_fmaps=4)

        encoders = []
        for i, out_feature_num in enumerate(fea_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(fea_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        decoders = []
        reversed_fea_maps = list(reversed(fea_maps))
        for i in range(len(reversed_fea_maps) - 1):
            in_feature_num = reversed_fea_maps[i] + reversed_fea_maps[i + 1]
            out_feature_num = reversed_fea_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        self.final_conv = nn.Conv3d(fea_maps[0], out_channels, 1)

        if final_sig:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # 编码器部分
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # 反转编码器输出以与解码器对齐
            encoders_features.insert(0, x)
            #print(x.shape)

        encoders_features = encoders_features[1:]

        # 解码器部分
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)

        x = self.final_conv(x)
        return x


class UNet_attention(nn.Module):
    '''
    在UNet网络中加入注意力模块
    '''
    def __init__(self, in_channels, out_channels, final_sig, fea_maps=64, layer_order='cr', num_groups=8,
                 **kwargs):
        super(UNet_attention, self).__init__()

        if isinstance(fea_maps, int):
            # 使用四层特征图
            fea_maps = create_feature_maps(fea_maps, number_of_fmaps=4)

        # 创建由编码器模块组成的编码器路径
        encoders = []
        for i, out_feature_num in enumerate(fea_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(fea_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # 注意力模块 
        self.attentionblock2 = AttentionBlock(in_channels=fea_maps[0], gating_channels=fea_maps[3],
                                                    inter_channels=fea_maps[0], sub_sample_factor=(2,2,2), mode='concat_attention')
        self.attentionblock3 = AttentionBlock(in_channels=fea_maps[1], gating_channels=fea_maps[3],
                                                    inter_channels=fea_maps[1], sub_sample_factor=(2,2,2), mode='concat_attention')
        self.attentionblock4 = AttentionBlock(in_channels=fea_maps[2], gating_channels=fea_maps[3],
                                                    inter_channels=fea_maps[2], sub_sample_factor=(2,2,2), mode='concat_attention')
        attentionblocks = [self.attentionblock4, self.attentionblock3, self.attentionblock2]
        self.attentionblocks = nn.ModuleList(attentionblocks)

        # 创建由解码器模块组成的解码器路径
        decoders = []
        reversed_fea_maps = list(reversed(fea_maps))
        for i in range(len(reversed_fea_maps) - 1):
            in_feature_num = reversed_fea_maps[i] + reversed_fea_maps[i + 1]
            out_feature_num = reversed_fea_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # 最后一层使用1×1卷积层融合所有通道信息
        self.final_conv = nn.Conv3d(fea_maps[0], out_channels, 1)

        if final_sig:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # 编码器部分，构建所有网络层的路径
        encoders_features_all = []
        for encoder in self.encoders:
            x = encoder(x)
            # 反转编码器输出以与解码器对齐
            encoders_features_all.insert(0, x)

        encoders_features = encoders_features_all[1:]
        gating = encoders_features_all[0]

        # 解码器部分
        for decoder, encoder_features, attentionblock in zip(self.decoders, encoders_features, self.attentionblocks):
            attention_x, sig_out = attentionblock(encoder_features, gating)   
            x = decoder(attention_x, x)

        x = self.final_conv(x)
        return x


class UNet2(nn.Module):
    '''
    UNet++
    '''
    def __init__(self, in_channels, out_channels, final_sig, fea_maps=64, layer_order='cr', num_groups=8,
                 **kwargs):
        super(UNet2, self).__init__()

        if isinstance(fea_maps, int):
            # 使用四层特征图
            fea_maps = create_feature_maps(fea_maps, number_of_fmaps=4)

        # 创建由编码器模块组成的编码器路径
        encoders = []
        for i, out_feature_num in enumerate(fea_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(fea_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # 创建由解码器模块组成的解码器路径
        decoders = []
        d_11=Decoder2(fea_maps[0]+fea_maps[1], fea_maps[0], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_11)
        d_21=Decoder2(fea_maps[1]+fea_maps[2], fea_maps[1], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_21)
        d_31=Decoder2(fea_maps[2]+fea_maps[3], fea_maps[2], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_31)
        d_12=Decoder2(fea_maps[1]+fea_maps[0]+fea_maps[0], fea_maps[0], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_12)
        d_22=Decoder2(fea_maps[2]+fea_maps[1]+fea_maps[1], fea_maps[1], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_22)
        d_13=Decoder2(fea_maps[1]+fea_maps[0]*3, fea_maps[0], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_13)

        self.decoders = nn.ModuleList(decoders)

        # 最后一层使用1×1卷积层融合所有通道信息
        self.final_conv = nn.Conv3d(fea_maps[0], out_channels, 1)

        if final_sig:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # 编码器部分
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.append(x)

        # 解码器部分
        d_i=0
        final_=[]
        for decoder in self.decoders:
            x = decoder(encoders_features)
            encoders_features.append(x)
            if d_i==0 or d_i==3 or d_i==5:
                final_i=self.final_conv(x)
                final_.append(final_i)
            d_i+=1

        x = (final_[0]+final_[1]+final_[2])/3
        return x


class UNet2_attention_FA(nn.Module):
    '''
    在UNet++网络中加入注意力模块，注意力模块遍布在除了U型网络的第一层之外的其他层跳跃连接中
    '''
    def __init__(self, in_channels, out_channels, final_sig, fea_maps=64, layer_order='cr', num_groups=8,
                 **kwargs):
        super(UNet2_attention_FA, self).__init__()

        if isinstance(fea_maps, int):
            # 使用四层特征图
            fea_maps = create_feature_maps(fea_maps, number_of_fmaps=4)

        # 创建由编码器模块组成的编码器路径
        encoders = []
        for i, out_feature_num in enumerate(fea_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(fea_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # 创建由解码器模块组成的解码器路径
        decoders = []
        attentionblocks = []
        attentionblock_11 = AttentionBlock(in_channels=fea_maps[0], gating_channels=fea_maps[1],
                                                    inter_channels=fea_maps[0], sub_sample_factor=(2,2,2), mode='concat_attention')
        attentionblocks.append(attentionblock_11)
        d_11=Decoder2_attention_FA(fea_maps[0]+fea_maps[1], fea_maps[0], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_11)

        attentionblock_21 = AttentionBlock(in_channels=fea_maps[1], gating_channels=fea_maps[2],
                                                    inter_channels=fea_maps[1], sub_sample_factor=(2,2,2), mode='concat_attention')
        attentionblocks.append(attentionblock_21)
        d_21=Decoder2_attention_FA(fea_maps[1]+fea_maps[2], fea_maps[1], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_21)

        attentionblock_31 = AttentionBlock(in_channels=fea_maps[2], gating_channels=fea_maps[3],
                                                    inter_channels=fea_maps[2], sub_sample_factor=(2,2,2), mode='concat_attention')
        attentionblocks.append(attentionblock_31)
        d_31=Decoder2_attention_FA(fea_maps[2]+fea_maps[3], fea_maps[2], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_31)

        attentionblock_12 = AttentionBlock(in_channels=fea_maps[0], gating_channels=fea_maps[1],
                                                    inter_channels=fea_maps[0], sub_sample_factor=(2,2,2), mode='concat_attention')
        attentionblocks.append(attentionblock_12)
        d_12=Decoder2_attention_FA(fea_maps[1]+fea_maps[0]+fea_maps[0], fea_maps[0], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_12)

        attentionblock_22 = AttentionBlock(in_channels=fea_maps[1], gating_channels=fea_maps[2],
                                                    inter_channels=fea_maps[1], sub_sample_factor=(2,2,2), mode='concat_attention')
        attentionblocks.append(attentionblock_22)
        d_22=Decoder2_attention_FA(fea_maps[2]+fea_maps[1]+fea_maps[1], fea_maps[1], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_22)

        attentionblock_13 = AttentionBlock(in_channels=fea_maps[0], gating_channels=fea_maps[1],
                                                    inter_channels=fea_maps[0], sub_sample_factor=(2,2,2), mode='concat_attention')
        attentionblocks.append(attentionblock_13)
        d_13=Decoder2_attention_FA(fea_maps[1]+fea_maps[0]*3, fea_maps[0], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_13)

        self.decoders = nn.ModuleList(decoders)
        self.attentionblocks = nn.ModuleList(attentionblocks)

        # 最后一层使用1×1卷积层融合所有通道信息
        self.final_conv = nn.Conv3d(fea_maps[0], out_channels, 1)

        if final_sig:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # 编码器部分，构建所有网络层的路径
        encoders_decodes_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_decodes_features.append(x)

        # 解码器部分
        d_i=0
        final_=[]
        attention_features = []
        for decoder, attentionblock in zip(self.decoders,self.attentionblocks):
            if d_i == 1 or d_i == 2:
                attention_x, sig_out = attentionblock(encoders_decodes_features[d_i], encoders_decodes_features[d_i+1])
                attention_features.append(attention_x)
            elif d_i == 4:
                attention_x, sig_out = attentionblock(encoders_decodes_features[d_i+1], encoders_decodes_features[d_i+2])
                attention_features.append(attention_x)
            x = decoder(encoders_decodes_features, attention_features)
            encoders_decodes_features.append(x)
            if d_i==0 or d_i==3 or d_i==5:
                final_i=self.final_conv(x)
                final_.append(final_i)
            d_i+=1

        x = (final_[0]+final_[1]+final_[2])/3
        return x
    

class UNet2_attention_DA(nn.Module):
    '''
    在UNet++网络中加入注意力模块，注意力模块遍布编码层跳跃连接中
    '''
    def __init__(self, in_channels, out_channels, final_sig, fea_maps=64, layer_order='cr', num_groups=8,
                 **kwargs):
        super(UNet2_attention_DA, self).__init__()

        if isinstance(fea_maps, int):
            # 使用四层特征图
            fea_maps = create_feature_maps(fea_maps, number_of_fmaps=4)

        # 创建由编码器模块组成的解码器路径
        encoders = []
        for i, out_feature_num in enumerate(fea_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(fea_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # 创建由解码器模块组成的解码器路径，以及注意力模块组成的注意力模块路径
        decoders = []
        attentionblocks = []
        attentionblock_11 = AttentionBlock(in_channels=fea_maps[0], gating_channels=fea_maps[1],
                                                    inter_channels=fea_maps[0], sub_sample_factor=(2,2,2), mode='concat_attention')
        attentionblocks.append(attentionblock_11)
        d_11=Decoder2_attention_DA(fea_maps[0]+fea_maps[1], fea_maps[0], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_11)

        attentionblock_21 = AttentionBlock(in_channels=fea_maps[1], gating_channels=fea_maps[2],
                                                    inter_channels=fea_maps[1], sub_sample_factor=(2,2,2), mode='concat_attention')
        attentionblocks.append(attentionblock_21)
        d_21=Decoder2_attention_DA(fea_maps[1]+fea_maps[2], fea_maps[1], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_21)

        attentionblock_31 = AttentionBlock(in_channels=fea_maps[2], gating_channels=fea_maps[3],
                                                    inter_channels=fea_maps[2], sub_sample_factor=(2,2,2), mode='concat_attention')
        attentionblocks.append(attentionblock_31)
        d_31=Decoder2_attention_DA(fea_maps[2]+fea_maps[3], fea_maps[2], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_31)

        attentionblock_12 = AttentionBlock(in_channels=fea_maps[0], gating_channels=fea_maps[1],
                                                    inter_channels=fea_maps[0], sub_sample_factor=(2,2,2), mode='concat_attention')
        attentionblocks.append(attentionblock_12)
        d_12=Decoder2_attention_DA(fea_maps[1]+fea_maps[0]+fea_maps[0], fea_maps[0], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_12)

        attentionblock_22 = AttentionBlock(in_channels=fea_maps[1], gating_channels=fea_maps[2],
                                                    inter_channels=fea_maps[1], sub_sample_factor=(2,2,2), mode='concat_attention')
        attentionblocks.append(attentionblock_22)
        d_22=Decoder2_attention_DA(fea_maps[2]+fea_maps[1]+fea_maps[1], fea_maps[1], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_22)

        attentionblock_13 = AttentionBlock(in_channels=fea_maps[0], gating_channels=fea_maps[1],
                                                    inter_channels=fea_maps[0], sub_sample_factor=(2,2,2), mode='concat_attention')
        attentionblocks.append(attentionblock_13)
        d_13=Decoder2_attention_DA(fea_maps[1]+fea_maps[0]*3, fea_maps[0], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_13)

        self.decoders = nn.ModuleList(decoders)
        self.attentionblocks = nn.ModuleList(attentionblocks)

        # 最后一层使用1×1卷积层融合所有通道信息
        self.final_conv = nn.Conv3d(fea_maps[0], out_channels, 1)

        if final_sig:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # 编码器部分，构建所有网络层的路径
        encoders_decodes_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_decodes_features.append(x)

        # 解码器部分
        d_i=0
        final_=[]
        attention_features = []
        for decoder, attentionblock in zip(self.decoders,self.attentionblocks):
            if d_i < 3:
                attention_x, sig_out = attentionblock(encoders_decodes_features[d_i], encoders_decodes_features[d_i+1])
                attention_features.append(attention_x)
            elif d_i == 3:
                attention_x, sig_out = attentionblock(encoders_decodes_features[0], encoders_decodes_features[5])
                attention_features.append(attention_x)
            elif d_i == 4:
                attention_x, sig_out = attentionblock(encoders_decodes_features[1], encoders_decodes_features[6])
                attention_features.append(attention_x)
            elif d_i == 5:
                attention_x, sig_out = attentionblock(encoders_decodes_features[0], encoders_decodes_features[8])
                attention_features.append(attention_x)
            
            x = decoder(encoders_decodes_features, attention_features)
            encoders_decodes_features.append(x)  
            if d_i==0 or d_i==3 or d_i==5:
                final_i=self.final_conv(x)
                final_.append(final_i)
            d_i+=1

        x = (final_[0]+final_[1]+final_[2])/3
        return x


class UNet3(nn.Module):
    '''
    UNet3+
    '''
    def __init__(self, in_channels, out_channels, final_sig, fea_maps=64, layer_order='cr', num_groups=8,
                 **kwargs):
        super(UNet3, self).__init__()

        if isinstance(fea_maps, int):
            # 使用四层特征图
            fea_maps = create_feature_maps(fea_maps, number_of_fmaps=4)

        # 创建由编码器模块组成的解码器路径
        encoders = []
        for i, out_feature_num in enumerate(fea_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(fea_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # 创建由解码器模块组成的解码器路径
        decoders = []
        for i in range(len(fea_maps) - 1):
            in_feature_num = fea_maps[0] + fea_maps[1] + fea_maps[2] + fea_maps[3]
            out_feature_num = fea_maps[-i-2]
            decoder = Decoder3(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # 最后一层使用1×1卷积层融合所有通道信息
        self.final_conv = nn.Conv3d(fea_maps[0], out_channels, 1)

        if final_sig:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # 编码器部分
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.append(x)

        # 解码器部分
        for decoder in self.decoders:
            x = decoder(encoders_features, x)
            encoders_features.append(x)

        x = self.final_conv(x)
        return x


class Noise2Noise(nn.Module):
    '''
    Noise2Noise
    '''
    def __init__(self, in_channels, out_channels, fea_maps=16, num_groups=8, **kwargs):
        super(Noise2Noise, self).__init__()

        # 使用LeakyReLU激活函数，GroupNorm标准化
        conv_layer_order = 'clg'   

        if isinstance(fea_maps, int):
            fea_maps = create_feature_maps(fea_maps, number_of_fmaps=4) 

        encoders = []
        for i, out_feature_num in enumerate(fea_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=conv_layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(fea_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=conv_layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        decoders = []
        reversed_fea_maps = list(reversed(fea_maps))
        for i in range(len(reversed_fea_maps) - 1):
            in_feature_num = reversed_fea_maps[i] + reversed_fea_maps[i + 1]
            out_feature_num = reversed_fea_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=conv_layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # 最后一层使用1×1卷积层融合所有通道信息，并使用ReLU激活  
        self.final_conv = SingleConv(fea_maps[0], out_channels, kernel_size=1, order='cr', padding=0)

    def forward(self, x):
        # 编码器部分
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # 反转编码器输出以与解码器对齐
            encoders_features.insert(0, x)

        encoders_features = encoders_features[1:]

        # 解码器部分
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        return x


def get_model(config):
    def _model_class(class_name):
        m = importlib.import_module('attention_unet++.model')
        clazz = getattr(m, class_name) 
        return clazz

    assert 'model' in config, 'Could not find model configuration'
    model_config = config['model']
    model_class = _model_class(model_config['name'])
    return model_class(**model_config)
