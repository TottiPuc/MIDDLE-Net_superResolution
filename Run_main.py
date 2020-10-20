from __future__ import print_function
import matplotlib.pyplot as plt
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np

import torch
import torch.optim

import scipy.io
from skimage.measure import compare_psnr
from models.downsampler import Downsampler,Spectral
from models.resnet import ResNet
from models.unet import UNet
from models.skip import skip

from utils.sr_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor


file_name  = 'data/sr/sr2.mat'
imgs = scipy.io.loadmat(file_name)
#plot band
band = 24



HR = imgs['HR2'].transpose(2,0,1)
HR = torch.from_numpy(HR).type(dtype)
HR = HR[None, :].cuda()

# Low resolution Image
factor = 4
KERNEL_TYPE='lanczos2'
downsampler = Downsampler(n_planes=imgs['HR2'].shape[2], factor=factor, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True).type(dtype)
img_LR_var = torch.from_numpy(downsampler(HR).detach().cpu().numpy()).type(dtype)
mostr = img_LR_var.detach().cpu().squeeze().numpy().transpose(1, 2, 0)
plt.subplot(1,3,1),plt.imshow(imgs['HR2'][:,:,band]),plt.title('High')
plt.subplot(1,3,3),plt.imshow(mostr[:, :, band]),plt.title('Hyper')

dw_spectral = Spectral(Input_band=96,Oput_band=8)
img_hyper = dw_spectral(HR)
mostr = img_hyper.detach().cpu().squeeze().numpy().transpose(1, 2, 0)
plt.subplot(1,3,2),plt.imshow(mostr[:, :, 5]),plt.title('Multi')
plt.show()



input_depth = imgs['HR2'].shape[2]

method =    '2D'
pad   =     'reflection'
OPT_OVER =  'net'

show_every = 500
save_every = 2000

LR = 0.01
tv_weight = 0.0
OPTIMIZER = 'adam'
num_iter = 12001
reg_noise_std = 0.03  # try 0 0.03 0.05 0.08

net_input = get_noise(input_depth, method, (imgs['HR2'].shape[0], imgs['HR2'].shape[1])).type(dtype).detach()

net = skip(input_depth, imgs['HR2'].shape[2],
           num_channels_down = [128]*5,
           num_channels_up =   [128]*5,
           num_channels_skip =    [4]*5,
           filter_size_up = 3,filter_size_down = 3,  filter_skip_size=1,
           upsample_mode='bilinear', # downsample_mode='avg',
           need1x1_up=False,
           need_sigmoid=False, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

# Losses
mse = torch.nn.MSELoss().type(dtype)

#img_LR_var = imgs['LR'].transpose(2,0,1)
#img_LR_var = torch.from_numpy(img_LR_var).type(dtype)
#img_LR_var = img_LR_var[None, :].cuda()







def closure():
    global i, net_input

    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out_HR = net(net_input)
    out_HR = net(out_HR)

    out_LR = downsampler(out_HR)
    out_specteal =  dw_spectral(out_HR)

    total_loss = mse(out_LR, img_LR_var)

    if tv_weight > 0:
        total_loss += tv_weight * tv_loss(out_HR)

    total_loss.backward()

    # Log
    psnr_LR = compare_psnr(img_LR_var.detach().cpu().squeeze().numpy().transpose(1, 2, 0), out_LR.detach().cpu().squeeze().numpy().transpose(1, 2, 0))
    psnr_HR = compare_psnr(imgs['HR2'].astype(np.float32), out_HR.detach().cpu().squeeze().numpy().transpose(1, 2, 0))
    print('Iteration %05d    PSNR_LR %.3f   PSNR_HR %.3f' % (i, psnr_LR, psnr_HR), '\r', end='')

    # History
    psnr_history.append([psnr_LR, psnr_HR])
    if i % show_every == 0:
        out_HR_np = out_HR.detach().cpu().squeeze().numpy().transpose(1, 2, 0)
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 15))
        ax1.imshow(imgs['HR2'][:, :, band], cmap='gray', vmin=0.0, vmax=0.5)
        ax2.imshow(imgs['bicubic'][:, :, band], cmap='gray', vmin=0.0, vmax=0.5)
        ax3.imshow(np.clip(out_HR_np, 0, 1)[:, :, band], cmap='gray', vmin=0.0, vmax=0.5)
        plt.show()

    if i % save_every == 0:
        out_HR_np = out_HR.detach().cpu().squeeze().numpy().transpose(1, 2, 0)
        scipy.io.savemat("results/result_sr_2D_it%05d.mat" % (i), {'pred': np.clip(out_HR_np, 0, 1)})

    i += 1

    return total_loss

psnr_history = []
net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

i = 0
p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)
