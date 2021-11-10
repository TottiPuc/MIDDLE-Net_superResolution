import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import * 

class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0: 
            idx = len(self) + idx

        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

class UNet_up(nn.Module):
    '''
        upsample_mode in ['deconv', 'nearest', 'bilinear']
        pad in ['zero', 'replication', 'none']
    '''
    def __init__(self, num_input_channels=3, num_output_channels=3, 
                       feature_scale=4, more_layers=0, concat_x=False,
                       upsample_mode='deconv', pad='zero', norm_layer=nn.InstanceNorm2d, need_sigmoid=True, need_bias=True):
        super(UNet_up, self).__init__()

        self.feature_scale = feature_scale
        self.more_layers = more_layers
        self.concat_x = concat_x


        #filters = [64, 128, 256, 512, 1024]
        #filters = [96, 96, 96, 96, 96] para pavia
        filters = [92, 92, 92, 92, 92]# para salinas
        #filters = [x // self.feature_scale for x in filters]

        self.start = unetConv2(num_input_channels, filters[3] , norm_layer, need_bias, pad)

        self.up4 = unetUp(filters[3], upsample_mode, need_bias, pad)
        self.up3 = unetUp(filters[2], upsample_mode, need_bias, pad)
        #self.up2 = unetUp(filters[1], upsample_mode, need_bias, pad)
        #self.up1 = unetUp(filters[0], upsample_mode, need_bias, pad)


        self.final = conv(filters[0], num_output_channels, 1, bias=need_bias, pad=pad)

        if need_sigmoid: 
            self.final = nn.Sequential(self.final, nn.Sigmoid())


    def forward(self, inputs):
        in64 = self.start(inputs)
        #print(inputs.shape)
        up4= self.up4(in64)
        #print(up4.shape)
        up3= self.up3(up4)
        #print('esta ',up3.shape)
        #up2= self.up2(up3)
        #print(up2.shape)
        #up1= self.up1(up2)
        #print(up1.shape)

        return self.final(up3)



class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
        super(unetConv2, self).__init__()

        
        if norm_layer is not None:
            self.conv1= nn.Sequential(conv(in_size, out_size, 3, bias=need_bias, pad=pad),
                                       norm_layer(out_size),
                                       nn.ReLU(),)
            self.conv2= nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                                       norm_layer(out_size),
                                       nn.ReLU(),)
            self.conv3= nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                                       nn.ReLU(),)
            self.conv4= nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                                       nn.ReLU(),) 
        else:
            self.conv1= nn.Sequential(conv(in_size, out_size, 3, bias=need_bias, pad=pad),
                                       nn.ReLU(),)
            self.conv2= nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                                       nn.ReLU(),)
            self.conv3= nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                                       nn.ReLU(),)
            self.conv4= nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                                       nn.ReLU(),) 
            #self.conv5= nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                #                       nn.ReLU(),)
            #self.conv6= nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
               #                        nn.ReLU(),)    
            #self.conv7= nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
              #                         nn.ReLU(),) 
            #self.conv8= nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
             #                          nn.ReLU(),)                          
    def forward(self, inputs):
        outputs1= self.conv1(inputs)
        outputs2= self.conv2(outputs1) # hasta aqui
        outputs3= self.conv3(outputs2) # esta es demas
        outputs4= self.conv4(outputs3) # esta de mas
        #outputs5= self.conv5(outputs4)
        #outputs6= self.conv5(outputs5)
        #outputs7= self.conv5(outputs6)
        #outputs= self.conv6(outputs7)
        return outputs4


class unetUp(nn.Module):
    def __init__(self, out_size, upsample_mode, need_bias, pad, same_num_filt=False):
        super(unetUp, self).__init__()

        num_filt = out_size if same_num_filt else out_size 
        if upsample_mode == 'deconv':
            self.up= nn.ConvTranspose2d(num_filt, out_size, 4, stride=2, padding=1)
            self.conv= unetConv2(out_size , out_size, None, need_bias, pad)
        elif upsample_mode=='bilinear' or upsample_mode=='nearest':
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode=upsample_mode),
                                   conv(num_filt, out_size, 3, bias=need_bias, pad=pad))
            self.conv= unetConv2(out_size , out_size, None, need_bias, pad)
        else:
            assert False

    def forward(self, inputs1):
        in1_up= self.up(inputs1)
        output= self.conv(in1_up)

        return output