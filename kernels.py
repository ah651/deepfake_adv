import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np 
import scipy 
import scipy.ndimage 

import functools
import cyclegan

class GaussianLayer(nn.Module):
    def __init__(self, radius, sigma):
        super(GaussianLayer, self).__init__()
        assert (radius % 2) > 0, "Radius can not be even!"
        assert sigma > 0, "Sigma should be non-negative!"

        self.seq = nn.Sequential(
            nn.ReflectionPad2d(int((radius-1)/2)),
            nn.Conv2d(3,3,radius,stride=1,padding=0,bias=None,groups=3)
        )
        self.weights_init(radius, sigma)

    def forward(self, x):
        return self.seq(x)

    def weights_init(self, radius, sigma):
        n= np.zeros((radius,radius))
        n[int((radius-1)/2),int((radius-1)/2)] = 1
        k = scipy.ndimage.gaussian_filter(n, sigma=sigma)
        print('kernel in gaussian')
        print(k)
        for _, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))
            f.requires_grad = False

class PixelGaussian(nn.Module):
    def __init__(self, radius, image_size=256):
        super(PixelGaussian, self).__init__()
        assert (radius % 2) > 0, "Radius can not be even!"

        self.radius = radius 
        self.image_size = image_size
        self.pad = nn.ReflectionPad2d(int((self.radius-1)//2))
        # init 
        self.origin_weights = [[None for j in range(self.radius)] for i in range(self.radius)]
        a = radius // 2
        b = radius // 2 
        for i in range(self.radius):
            for j in range(self.radius):
                # print(-((i-a)*(i-a) + (j-b)*(j-b))/2)
                self.origin_weights[i][j] = torch.full((1, 3, image_size, image_size), -((i-a)*(i-a) + (j-b)*(j-b))/2, dtype=torch.float32).cuda()
                # print(self.origin_weights[i][j][0, 0, 0, 0])
        # self.weights_init(radius, sigma)

    def forward(self, x, sigma):
        # x.shape: b*s, 3, 224, 224
        # sigma: b*s, 1, 224, 224
        # self.tmp_weight[i][j].shape: b*s, 3, 224, 224
        self.tmp_weight = self.weights_init(sigma, x)
        
        result = torch.zeros_like(x)
        x = self.pad(x)
        # print(x.shape) 
        for i in range(self.radius):
            for j in range(self.radius):
                result += x[:, :, i:i+self.image_size, j:j+self.image_size] * self.tmp_weight[i][j]
        return result 

    def weights_init(self, sigma, x):
        tmp = [[None for j in range(self.radius)] for i in range(self.radius)]
        for i in range(self.radius):
            for j in range(self.radius):
                # tmp[i][j] = torch.exp(self.origin_weights[i][j].expand(x.shape[0], -1, -1, -1) * sigma * sigma)
                tmp[i][j] = torch.exp(self.origin_weights[i][j].expand(x.shape[0], -1, -1, -1) * sigma)
                if i == 0 and j == 0:
                    tmp_sum = tmp[i][j].clone()
                else:
                    tmp_sum += tmp[i][j]

        for i in range(self.radius):
            for j in range(self.radius):
                tmp[i][j] = tmp[i][j] / tmp_sum
        return tmp 

class PixelFilter(nn.Module):
    def __init__(self, radius=3):
        super(PixelFilter, self).__init__()
        assert (radius % 2) > 0, "Radius can not be even!"

        self.radius = radius 
        self.eps = 1e-9
        self.pad = nn.ReflectionPad2d(int((self.radius-1)//2))

    def forward(self, x, sigma):

        sum_tmp = sigma.sum(dim=1, keepdim=True) 
        sigma = sigma / (sum_tmp + self.eps)

        result = torch.zeros_like(x)
        x = self.pad(x)
        # print(x.shape) 
        for i in range(self.radius):
            for j in range(self.radius):
                result += x[:, :, i:i+256, j:j+256] * sigma[:, self.radius*i+j:self.radius*i+j+1, ...]
        return result 


class PixelKernelGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, pert_channels=0, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=0, padding_type='reflect', all=False, radius=3, use_bias=False, norm=False, learn_norm = False, no_clip=False, sigmoid=False):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks, it should be 6/9
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(PixelKernelGenerator, self).__init__()
        self.eps = 1e-9
        self.all = all 
        self.radius = radius
        self.use_bias = use_bias 
        self.norm = norm 
        self.learn_norm = learn_norm 
        self.output_nc = output_nc 
        self.pert_channels = pert_channels
        self.no_clip = no_clip 
        self.sigmoid = sigmoid 
        # [100, 500]
        if self.learn_norm:
            self.new_norm = nn.Parameter(torch.tensor(100.0, dtype=torch.float32)) 
        else:
            self.new_norm = 100.0 

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(16, 1, 224, 224)).cuda()
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [cyclegan.ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        if self.all:
            model += [nn.Conv2d(ngf, output_nc*self.radius*self.radius, kernel_size=7, padding=0)]
        else:
            model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        # model += [nn.Tanh()]

        self.model = nn.Sequential(*model)


    def forward(self, x):
        x = self.model(x)
        x = x * x 
        if self.all or self.no_clip:
            return x
        elif self.norm:
            # every sample a norm 
            norm_value = torch.norm(x, p=2., dim=(1,2,3), keepdim=True)
            return x / norm_value * self.new_norm
        else:
            return torch.clamp(x, min=1, max=1e12)


from cyclegan import init_weights
def pixelKernelGenerator(input_nc=3, output_nc=1, pert_channels=0, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=0, padding_type='reflect', all=False, radius=3, use_bias=False, norm=False, learn_norm=False, no_clip=False, sigmoid=False):
    model = PixelKernelGenerator(input_nc=input_nc, output_nc=output_nc, pert_channels=pert_channels, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks, padding_type=padding_type, all=all, radius=radius, use_bias=use_bias, norm=norm, learn_norm=learn_norm, no_clip=no_clip, sigmoid=sigmoid)
    return init_weights(model)

