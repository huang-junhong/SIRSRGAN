import argparse

def main():
    parser = argparse.ArgumentParser(description='Parameter for make dataset')
    parser.add_argument('-hr_path', '--HR_PATH', default='D:\\SIR-SRGAN_OPEN\\Data/HR')
    parser.add_argument('-lr_path', '--LR_PATH', default='D:\\SIR-SRGAN_OPEN\\Data/SRF4')
    parser.add_argument('-save_path', '--SAVE_PATH', default='D:\\SIR-SRGAN_OPEN\\Train_Dataset')
    parser.add_argument('-hr_size', '--HR_SIZE', default=256)
    parser.add_argument('-patch_stride', '--PATCH_STRIDE', default=128)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()