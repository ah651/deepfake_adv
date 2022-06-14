import torch
import torchvision
import random
from PIL import Image
import numpy as np

class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        if random.random() < 0.5:
            return [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
        else:
            return img_group

class GroupCenterCrop(object):

    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class CutCenter(object):
    def __init__(self, size=96): 
        self.mask_size = size

    def __call__(self, img):
        '''
        img: uint8, ndarray
        '''
        h, w = img.shape[0], img.shape[1]
        ymin, xmin = (h - self.mask_size) // 2, (w - self.mask_size) // 2
        # mask = np.zeros((h,w,1),dtype=np.uint8) 
        # mask[ymin:ymin+self.mask_size, xmin:xmin+self.mask_size] = 127
        img[ymin:ymin+self.mask_size, xmin:xmin+self.mask_size] = 127
        # img = img + mask
        return np.clip(img, 0, 255) 


class GroupCutCenter(object):
    def __init__(self, size=96): 
        self.mask_size = size
        self.worker = CutCenter(size)

    def __call__(self, img_group):
        img_group = [self.worker(np.array(img)) for img in img_group]
        return [Image.fromarray(img) for img in img_group]


class GroupScale(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Scale(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class Stack(object):

    def __call__(self, img_group):
        return np.concatenate([np.expand_dims(x, 0) for x in img_group], axis=0)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        img = torch.from_numpy(pic).permute(0, 3, 1, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class GroupNormalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
        rep_std = self.std * (tensor.size()[0]//len(self.std))

        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor
