import os
import torch 
import torch.utils.data as data  
import torchvision
from transforms import *
from glob import glob
from tqdm import tqdm
import json 
import pickle
import multiprocessing as mp

import random 
import numpy as np
from numpy.random import randint

from PIL import Image
import random
import albumentations as A
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur

       
def create_train_transforms(size=320):
    return Compose([
      ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        GaussNoise(p=0.1),
        HorizontalFlip(),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
        ToGray(p=0.2),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    ]
    )


class VideoRecord(object):
    def __init__(self, item, labelx=-100, tp='jpg'):
        self._path = item[0]
        self._labelx = labelx 
        self.type = tp 
        self._labely = 1 if item[1] == "REAL" else 0
        self._frames = glob(r"%s/*.jpg"%self._path)
        self._frames += glob(r"%s/*.png"%self._path)
        self._index_construct()

    @property
    def path(self):
        return self._path

    @property
    def num_frames(self):
        return len(self._frames)

    @property
    def labelx(self):
        return self._labelx

    @property
    def labely(self):
        return self._labely

    @property
    def findex(self):
        return self._findex  

    def _index_construct(self):
        if self.type == 'jpg':
            tmpl = os.path.join(self._path, '{:d}.jpg') 
        else:
            tmpl = os.path.join(self._path, '{:d}.png') 
        skip, self._findex = 0, []
        for i in range(self.num_frames):
            fi = tmpl.format(i+skip)
            while (fi not in self._frames) and (skip < 999999):
                skip += 1
                fi = tmpl.format(i+skip)
            self._findex.append(i+skip)


class FF_video_dataset(data.Dataset): 
    '''
    Items: [video_path, label, img_list]
    '''
    def __init__(self,roots, annos, shuffle=True, phase = "TRAIN", multi = True, transform=None, img_size=256, label='all', args=None): 
        
        super(FF_video_dataset,self).__init__()  
     
        self.roots = roots 
        self.annos = annos 
        self.img_size = img_size
        self.phase = phase 
        self.multi = multi
        self.args = args 
        self.label = label 
        self.tr = None 

        if isinstance(roots, list):
            self.roots = roots[0] 
            self.annos = annos[0]

        if self.phase == 'TRAIN':
            self.tr = create_train_transforms(size=img_size)

        self.folder_labels = []
        new_list = []
        with open(self.annos, "rb") as fh:
            base_dir = self.roots
            fh = list(fh)
            for idx2 in tqdm(range(len(fh))):
                info = fh[idx2].decode("utf-8").rstrip().split()
                label = info[1]
                fold = os.path.join(base_dir, info[0])
                img_list = glob(r"%s/*_0.png"%fold)
                new_list.append([info[0], label, img_list])             
                
            self.folder_labels.extend(new_list) 


        print('Folder len: {}'.format(len(self.folder_labels)))          
        self.videos = []
        if self.label == 'REAL' or self.label == 'FAKE':
            for f in self.folder_labels:
                if f[1] == self.label:
                    self.videos.append(f) 
        else:
            self.videos = self.folder_labels 

        repeat_num = int(1000 * 32 / len(self.videos)) + 1
        self.videos = self.videos * repeat_num 
        
        print('Dataset len: {}'.format(len(self.videos)))

        if shuffle:
            random.shuffle(self.videos) 
        if args.debug:
            self.videos = self.videos[:150]

        self.transform = transform

    def __getitem__(self, index): 
        video, label, frame_lst = self.videos[index] 
        segment_indice = randint(len(frame_lst), size=1)
        segment_indice = segment_indice[0] 

        pth = frame_lst[segment_indice]
        img = Image.open(pth).convert('RGB')
        if self.phase == 'TRAIN' and self.tr:
            # add data aug ops
            img = np.asarray(img) 
            img = self.tr(image=img)['image']
            img = Image.fromarray(img) 

        if self.multi:
            if video.startswith('original'):
                label = 0 
            elif video.startswith('Deepfakes'):
                label = 1
            elif video.startswith('Face2Face'):
                label = 2
            elif video.startswith('FaceSwap'):
                label = 3
            else:
                label = 4
        else:
            label = 0 if label == 'FAKE' else 1

        if self.transform is not None:
            img = self.transform(img) 
        
        return img, label
 
    def __len__(self): 
        return len(self.videos)


class FF_lst_dataset(data.Dataset): 
    '''
    Items: [video_path, label, img_list]
    '''
    def __init__(self,roots, annos, shuffle=True, phase = "TRAIN", multi = True, transform=None, img_size=256, label='all', args=None): 
        
        super(FF_lst_dataset,self).__init__()  
     
        self.roots = roots 
        self.annos = annos 
        self.img_size = img_size
        self.phase = phase 
        self.multi = multi
        self.args = args 
        self.label = label 
        self.tr = None 

        if self.phase == 'TRAIN':
            self.tr = create_train_transforms(size=img_size)

        with open(annos) as f:
            frames = f.readlines() 
        frames = [f.strip() for f in frames]
        self.frames = []
        if self.label == 'REAL' or self.label == 'FAKE':
            for f in frames:
                if f.endswith(self.label):
                    self.frames.append(f) 
        else:
            self.frames = frames 
        
        print('Dataset len: {}'.format(len(self.frames)))          
        if shuffle:
            random.shuffle(self.frames) 
        if args.debug:
            self.frames = self.frames[:150]

        self.transform = transform

    def __getitem__(self, index): 
        pth = self.frames[index] 
        pth, label = pth.split()
        img = Image.open(os.path.join(self.roots, pth)).convert('RGB')
        if self.phase == 'TRAIN' and self.tr:
            # add data aug ops
            img = np.asarray(img) 
            img = self.tr(image=img)['image']
            img = Image.fromarray(img) 

        if self.multi:
            if pth.startswith('original'):
                label = 0 
            elif pth.startswith('Deepfakes'):
                label = 1
            elif pth.startswith('Face2Face'):
                label = 2
            elif pth.startswith('FaceSwap'):
                label = 3
            else:
                label = 4
        else:
            label = 0 if label == 'FAKE' else 1

        if self.transform is not None:
            img = self.transform(img) 
        
        return img, label
 
    def __len__(self): 
        return len(self.frames)
