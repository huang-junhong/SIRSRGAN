from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import random

class Train_Set(Dataset):
    def _load_img_path(self, path):
        files = []
        with open(path, 'r') as f:
            for line in f.readlines():
                files.append(line.strip('\n'))
        return files

    def __init__(self, LR_Folder, HR_Folder):
        print('Prepare LR')
        self.LR_PATH = self._load_img_path(LR_Folder+'/SRF4_PATH.txt')
        print('Prepare HR')
        self.HR_PATH = self._load_img_path(HR_Folder+'/HR_PATH.txt')

    def __len__(self):
        return len(self.LR_PATH)

    def __getitem__(self, index):
        lr = cv2.imread(self.LR_PATH[index])
        hr = cv2.imread(self.HR_PATH[index])
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        rot = random.randint(0, 3)
        lr = np.rot90(lr, rot, [0,1])
        hr = np.rot90(hr, rot, [0,1])
        filp = random.randint(-1, 2)
        if filp != 2:
            lr = cv2.flip(lr, filp)
            hr = cv2.flip(hr, filp)
        lr = lr.astype('float32') / 255.
        hr = hr.astype('float32') / 255.
        lr = np.transpose(lr, [2,0,1])
        hr = np.transpose(hr, [2,0,1])
        return lr, hr


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_lr, self.next_hr = next(self.loader)
        except StopIteration:
            self.next_lr = None
            self.next_hr = None
            return
        with torch.cuda.stream(self.stream):
            self.next_lr = self.next_lr.cuda(non_blocking=True)
            self.next_hr = self.next_hr.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        lr = self.next_lr
        hr = self.next_hr
        self.preload()
        return lr, hr
