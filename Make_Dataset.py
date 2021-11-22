import os
import cv2
import numpy as np
import Make_Dataset_config

def mkdir(path):
  
  folder=os.path.exists(path)
  
  if not folder:
    
    os.makedirs(path)
    
    print(path,' Folder Created')
    
  else:
    
    print(path,' Already Exist')

def load_file_path(PATH):
    filenames=[]
    for root,dir,files in os.walk(PATH):
        for file in files:
            if os.path.splitext(file)[1]=='.jpg' or os.path.splitext(file)[1]=='.png' or os.path.splitext(file)[1]=='.bmp':
                filenames.append(os.path.join(root,file))
    filenames = sorted(filenames)
    return filenames

def get_patch_img_rgb(img, patch_size, stride):
    patch = []                   
    temp = img.copy()
    h, w, _ = temp.shape
    h = h - h%patch_size
    w = w - w%patch_size
    for x in range(0, h, stride):
        for y in range(0, w, stride):
            if x+patch_size > h or y+patch_size > w:
                continue
            patch.append(temp[x:x+patch_size, y:y+patch_size,:])
    return np.array(patch, np.float32)

def rot180(input):
    temp=np.rot90(input)
    temp=np.rot90(temp)
    return temp

def rot270(input):
    temp=np.rot90(input)
    temp=np.rot90(temp)
    temp=np.rot90(temp)
    return temp

args = Make_Dataset_config.main()

HR_PATH = args.HR_PATH
LR_PATH = args.LR_PATH

HR_list = load_file_path(HR_PATH)
LR_list = load_file_path(LR_PATH)

HR_list = sorted(HR_list)
LR_list = sorted(LR_list)

mkdir(args.SAVE_PATH)
mkdir(args.SAVE_PATH+'/HR')
mkdir(args.SAVE_PATH+'/SRF4')

f_lr = args.SAVE_PATH + '/SRF4_PATH.txt'
f_hr = args.SAVE_PATH + '/HR_PATH.txt'

patch_count=0
for i in range(len(HR_list)):
    print(i)
        
    hr = cv2.imread(HR_list[i])
    lr = cv2.imread(LR_list[i])

    hr = get_patch_img_rgb(hr, 256, 128)
    lr = get_patch_img_rgb(lr, 64, 32)

    if len(hr) != len(lr):
        print('IMG ERROR')
        break

    for patch_i in range(len(hr)):
        patch_count +=1
            
        hri = hr[patch_i]
        lri = lr[patch_i]
 
        cv2.imwrite(args.SAVE_PATH+'/HR/'+str(patch_count)+'.png', hri)
        cv2.imwrite(args.SAVE_PATH+'/SRF4/'+str(patch_count)+'.png', lri)           
        with open(f_lr,'a') as file:
            file.write(args.SAVE_PATH+'/SRF4/'+str(patch_count)+'.png'+'\n')
        with open(f_hr,'a') as file:
            file.write(args.SAVE_PATH+'/HR/'+str(patch_count)+'.png'+'\n')


print(patch_count)
print('Complete')
