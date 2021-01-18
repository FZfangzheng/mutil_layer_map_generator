from MSE import MSE
from PIL import Image
import numpy as np
import os
import cv2

'''从一堆图片里寻找一张从某处看到的图'''
'''究极低配版识图'''

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
    target_img_path=r'C:\Users\xiaot\Desktop\1\1.jpg'
    find_folder=r'D:\datasets\maps\all_B\trainB'
    find_imgs=sorted(make_dataset(find_folder))

    picA=Image.open(target_img_path)
    picA=np.array(picA)
    mse_s=[]
    for find_img in find_imgs:
        picB=Image.open(find_img)
        picB=np.array(picB)
        if not picA.shape==picB.shape:
            picA= cv2.resize(picA, picB.shape[:2])
        mse=MSE(picA,picB)
        mse_s.append(mse)

    list=sorted(range(len(mse_s)), key=lambda k: mse_s[k])
    for i in range(20):
        print(find_imgs[list[i]])