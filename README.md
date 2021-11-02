# SIRSRGAN
[SIR-SRGAN]() has been accepted by bmvc2021.
Code will coming soon

## Quick Test
* Download our [pre-train model]().
- Change model_path in test_config.py for load model 
* Run test.py
- Get results in model folder

---

## Train
   - requirement: 

    cv2
  
    pywavelets == 1.0.3
  
    skimage == 0.16.2
  
### Prepare Dataset
1. Download [div2k]() & [fliker2k]() for train.
  
   note we use all fliker2k and div2k's train set for train.
     
2. Change trainset_path in train_config.py to save patches.
  
   You can change patch_size and patch_stride to control patch size and number of patches.
### Train PSNR-Orient model

    If you want train from zero
     
    1. Change model_type in train_config.py to chose model for generator.
    2. Change model_save_path in train_config.py for save models.
    3. Change val_set_lr, val_set_hr in train_config.py for test model during train.
    4. Run Train_PSNR.py
    
### Train GAN model
    1. Change model_load_path in train_config.py for load PSNR-Orient model to initial generator.
    2. Change model_save_path in train_config.py for save model.
    3. Run train.py
    
### Metrices
* PSNR & SSIM

  We use matlab for calculate PSNR and SSIM. Same with otheres, PSNR and SSIM only calculate y-channel.
  So you should use rgb2ycbcr convert to ycbcr then calculat PSNR and SSIM.
* PI

  PI also relay on matlab, donwload code from this [link]() and follow guide.
* LPIPS

  LPIPS's official code is [here]().In SIR-SRGAN we use VGG19 as backbone.
