import argparse

def get_test_config():
    parser = argparse.ArgumentParser(description='Parameter for test')
    parser.add_argument('-model', '--MODEL', default='SRRes')
    parser.add_argument('-model_path', '--MODEL_PATH', default='C:\\Users\\xx121\\Downloads/SRRes.pth')
    parser.add_argument('-test_images', '--TEST_PATH', default='D:\\SIR-SRGAN_OPEN\\Data\\SRF4')
    parser.add_argument('-foler', '--FOLDER', default='Test')
    parser.add_argument('-test_save', '--SAVE_PATH', default='D:\\SIR-SRGAN_OPEN\\Data')
    args = parser.parse_args()
    return args

