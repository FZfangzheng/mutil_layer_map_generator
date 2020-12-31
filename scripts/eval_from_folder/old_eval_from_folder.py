from fid.fid_score import fid_score
from MSE import MSE_from_floder
from SSIM import SSIM_from_floder
from PIL import Image
import numpy as np
import os
import skimage
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif','.npy'
]


def is_image_file(filename): # npy文件现在也算
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

if __name__=='__main__':
    # real_path=r'D:\map_translate\看看效果\0605p2p数据S2OMGAN默认参数，epoch200\real_result'
    # fake_path=r'D:\map_translate\看看效果\0605p2p数据S2OMGAN默认参数，epoch200\fake_result'

    real_path = r'D:\map_translate\看看效果\0627两个重绘数据集试验\color1-结合方法\real_result'
    fake_path = r'D:\map_translate\看看效果\0627两个重绘数据集试验\color1-结合方法\fake_result_tostand'
    real_files=sorted(make_dataset(real_path))
    fake_files = sorted(make_dataset(fake_path))
    assert len(real_files)==len(fake_files)

    # fid = fid_score(real_path=real_path, fake_path=fake_path, gpu='',batch_size=1)
    mse = MSE_from_floder(real_path, fake_path)
    from SSIM_detail import SSIM_from_floder
    ssim,ssim_mu,ssim_sigma = SSIM_from_floder(real_path, fake_path)
    # print(f'===> fid score:{fid:.4f},===> mse score:{mse:.4f},===> ssim score:{ssim:.4f}')
    print(f'ssim score:{ssim:.4f},{ssim_mu:.4f},{ssim_sigma:.4f}')