# SIRSRGAN
[SIR-SRGAN]() has been accepted by bmvc2021.
Code will coming soon

## Quick Test
* Download our [pre-train model]().
- Change model_path in test_config.py for load model 
* Run test.py
- Get results in model folder


## Train
* Prepare Dataset:
  1. Download [div2k]() & [fliker2k]() for train.
  
     note we use all fliker2k and div2k's train set for train.
  2. Change trainset_path in train_config.py to save patches.
  
     You can change patch_size and patch_stride to control patch size and number of patches.
