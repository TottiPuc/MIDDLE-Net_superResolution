import math
import numbers
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
dtype = torch.cuda.FloatTensor
class gaussian(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(gaussian, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)






def Spatial_blurring(HR,Num_channels, kernel_size, sigma,factor):
  blur = gaussian(Num_channels, kernel_size, sigma)
  Hper = blur(F.pad(HR.detach().cpu(), (2, 2, 2, 2), mode='reflect'))
  Hyper = Hper.detach().cpu().squeeze().numpy().transpose(1, 2, 0)
  Hyper_in = Hyper[0:-1:factor, 0:-1:factor,:]
  img_Hyper = torch.from_numpy(np.expand_dims(Hyper_in.transpose(2,0,1),axis=0)).type(dtype)
  return img_Hyper, Hyper_in



class spectral_blurrin(torch.nn.Module):
    def __init__(self, tam, factor):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(spectral_blurrin, self).__init__()
        self.factor = factor
        self.I_HS = np.zeros(shape=(tam[3],tam[2],tam[1]//factor))


    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        for i in range(x.shape[1]//self.factor):
            self.I_HS[:,:,i] = np.mean(x[0,i*(self.factor):(i+1)*self.factor,:,:].detach().cpu().numpy().transpose(1, 2, 0),axis=2)
        img_Multi = torch.from_numpy(np.expand_dims(self.I_HS.transpose(2,0,1),axis=0)).type(dtype)
        return img_Multi