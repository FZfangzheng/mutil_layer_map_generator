import numpy as np
from PIL import Image
import sys
import json
import os
from random import shuffle
import shutil
import datetime
from tqdm import tqdm
from skimage import measure

''''''

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


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

if __name__=='__main__':

    real_path = r"D:\map_translate\数据集\WH16分析\二次筛选相关\6.2生成结果\real_result"
    fake_path = r"D:\map_translate\数据集\WH16分析\二次筛选相关\6.2生成结果\fake_result"

    score_filename = 'score.json'
    result_filename='result.json'

    mses=[]
    ssims=[]
    psnrs=[]
    # 评估real结果
    print("process REAL")
    print("folder processing...")
    imgsA = sorted(make_dataset(real_path))
    imgsB = sorted(make_dataset(fake_path))

    for i in tqdm(range(len(imgsA))):
        img1_PIL = Image.open(imgsA[i])
        img2_PIL = Image.open(imgsB[i])
        img1_np = np.array(img1_PIL)
        img2_np = np.array(img2_PIL)
        ssim_tmp = measure.compare_ssim(rgb2gray(img1_np), rgb2gray(img2_np))
        mse_tmp = measure.compare_mse(img1_np, img2_np)
        psnr_tmp = measure.compare_psnr(img1_np, img2_np)
        mses.append(mse_tmp)
        ssims.append(ssim_tmp)
        psnrs.append(psnr_tmp)

    text={'mse':mses,'ssim':ssims,'psnr':psnrs}
    with open(score_filename,'w') as f:
        json.dump(text,f)

    mses=np.array(mses)
    ssims=np.array(ssims)
    psnrs=np.array(psnrs)

    mses=np.argsort(mses)
    ssims = np.argsort(ssims)[::-1]
    psnrs = np.argsort(psnrs)[::-1]

    a=np.zeros(len(imgsA))
    for i in range(len(imgsA)):
        a[mses[i]]+=i
        a[ssims[i]] += i
        a[psnrs[i]] += i
    a=np.argsort(a)

    result=[]
    for i in range(3000):
        result.append(imgsA[a[i]])
    with open(result_filename,'w') as f:
        json.dump(result,f)