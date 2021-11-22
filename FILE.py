import scipy.io as io
#import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
import scipy.signal
import copy
from skimage.measure import compare_psnr, compare_ssim
import scipy.misc

def load_file_path(PATH):
    filenames=[]
    for root,dir,files in os.walk(PATH):
        for file in files:
            if os.path.splitext(file)[1]=='.jpg' or os.path.splitext(file)[1]=='.png' or os.path.splitext(file)[1]=='.bmp' or os.path.splitext(file)[1]=='.tif':
                filenames.append(os.path.join(root,file))
    return filenames

def load_img(Paths,Normlize=False,as_array=False,Gray=False, Transpose=False):
    imgs=[]
    for i in range(len(Paths)):
        temp=None
        if Gray:
            temp=cv2.imread(Paths[i],0)
            temp=np.expand_dims(temp,2)
            temp=temp.astype('float32')
        else:
            temp=cv2.imread(Paths[i])
            temp=cv2.cvtColor(temp,cv2.COLOR_BGR2RGB)
            temp=temp.astype('float32')
        if Normlize:
            temp=temp/255.
        if Transpose:
            temp = np.transpose(temp, [2,0,1])
        imgs.append(temp)
    if as_array:
        imgs=np.array(imgs)
    return imgs

def mkdir(path):
  
  folder=os.path.exists(path)
  
  if not folder:
    
    os.makedirs(path)
    
    print(path,' Folder Created')
    
  else:
    
    print(path,' Already Exist')

def save_imgs(folder,imgs):

    mkdir(folder)
    
    for i in range(len(imgs)):
        cv2.imwrite(folder+'/'+str(i).zfill(3)+'_'+'.png', cv2.cvtColor(np.squeeze(imgs[i]), cv2.COLOR_RGB2BGR))

def load_test_date_res():
    INPUT='/sr/Huang/SAMPLE/SRF4'
    RESUL='/sr/Huang/SAMPLE/HR'

    INPUT=load_file_path(INPUT)
    RESUL=load_file_path(RESUL)

    INPUT=sorted(INPUT)
    RESUL=sorted(RESUL)

    INPUT=load_img(INPUT, Normlize=True, Transpose=True)
    RESUL=load_img(RESUL)

    return INPUT,RESUL
    
def calculate_psnr_ssim(HRS,LRS):
    psnr=0.
    ssim=0.
    for idx in range(len(HRS)):
        psnr+=compare_psnr(np.squeeze(HRS[idx].astype('uint8')),np.squeeze(LRS[idx].astype('uint8')))
        ssim+=compare_ssim(np.squeeze(HRS[idx].astype('uint8')),np.squeeze(LRS[idx].astype('uint8')),multichannel=True)

    return psnr/len(HRS),ssim/len(HRS)    
