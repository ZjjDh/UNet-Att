from model import UNet, Noise2Noise, UNet3, UNet2, UNet_attention, UNet2_attention, UNet2_attention_FA, UNet2_attention_DA

import torch.nn as nn

class Network(nn.Module):
    def __init__(self, type = 'UNet2_attention', in_channels=1, out_channels=1, fea_maps=64, final_sig = True):
        super(Network, self).__init__()

        self.type = type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_sig = final_sig

        if type == 'UNet':
            self.Generator = UNet( in_channels = in_channels,
                                     out_channels = out_channels,
                                     fea_maps = fea_maps,
                                     final_sig = final_sig) 
        elif type == 'UNet_attention':
            self.Generator = UNet_attention( in_channels = in_channels,
                                     out_channels = out_channels,
                                     fea_maps = fea_maps,
                                     final_sig = final_sig)  
        elif type == 'n2n':
            self.Generator = Noise2Noise( in_channels = in_channels,
                                                out_channels = out_channels,
                                                fea_maps = fea_maps,
                                                final_sig = final_sig)     
        elif type == 'UNet3':
            self.Generator = UNet3( in_channels = in_channels,
                                     out_channels = out_channels,
                                     fea_maps = fea_maps,
                                     final_sig = final_sig)
        elif type == 'UNet2':
            self.Generator = UNet2( in_channels = in_channels,
                                     out_channels = out_channels,
                                     fea_maps = fea_maps,
                                     final_sig = final_sig) 
        elif type == 'UNet2_attention':
            self.Generator = UNet2_attention( in_channels = in_channels,
                                     out_channels = out_channels,
                                     fea_maps = fea_maps,
                                     final_sig = final_sig)
        elif type == 'UNet2_attention_FA':
            self.Generator = UNet2_attention_FA( in_channels = in_channels,
                                     out_channels = out_channels,
                                     fea_maps = fea_maps,
                                     final_sig = final_sig) 
        elif type == 'UNet2_attention_DA':
            self.Generator = UNet2_attention_DA( in_channels = in_channels,
                                     out_channels = out_channels,
                                     fea_maps = fea_maps,
                                     final_sig = final_sig)                                              

    def forward(self, x):
        print(self.type)
        fake_x = self.Generator(x)
        return fake_x
