import torch
import numpy as np
import FILE
import Test_config
import unit
import RRDB

def get_model(config):
    G = None
    if config.MODEL == 'SRRes':
        G = unit.SRRes()
        G.load_state_dict({k.replace('module.',''):v for k,v in torch.load(config.MODEL_PATH).items()})
    elif config.MODEL == 'RRDB':
        G = RRDB.RRDBNet(3,3,64,23)
        G.load_state_dict(torch.load(config.MODEL_PATH))
    G = G.eval().cuda()
    return G


def main():
    config = Test_config.get_test_config()
    G = get_model(config)
    test_imgs = FILE.load_img(FILE.load_file_path(config.TEST_PATH), Normlize=True, Transpose=True)
    sr_imgs = []
    for i in range(len(test_imgs)):
        print(i+1,'/',len(test_imgs))
        img = np.expand_dims(test_imgs[i], 0)
        img = torch.Tensor(img).cuda()
        sr  = G(img).squeeze().detach().cpu().numpy()
        sr  = np.transpose(sr, [1,2,0])
        sr  = np.clip(sr*255,0,255).astype('uint8')
        sr_imgs.append(sr)
    FILE.save_imgs(config.SAVE_PATH+'//'+config.FOLDER, sr_imgs)
    print('Complete')

if __name__ == '__main__':
    main()