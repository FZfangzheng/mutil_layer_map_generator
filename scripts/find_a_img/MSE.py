# from test import MSE
import numpy as np
import os
from PIL import Image

def MSE(pic1, pic2):
    return np.sum(np.square(pic1 - pic2)) / (pic1.shape[0] * pic1.shape[1])

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif',
]
def is_image_file(filename):
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
def get_inner_path(file_path,floder_path):
    assert file_path[:len(floder_path)]==floder_path,"传入的文件不在文件夹中！[%s][%s]"%(file_path,floder_path)
    file_path=file_path[len(floder_path)+1:]
    return file_path

def MSE_from_floder(path1,path2): # 假设所有图片尺寸相同
    mse=0
    num=0
    imgs1=make_dataset(path1)
    imgs2=make_dataset(path2)
    for img1 in imgs1:
        img_inner = get_inner_path(img1, path1)
        if os.path.join(path2,img_inner) in imgs2:
            img2=os.path.join(path2,img_inner)
            img1_PIL=Image.open(img1)
            img2_PIL=Image.open(img2)
            img1_np=np.array(img1_PIL)
            img2_np=np.array(img2_PIL)
            mse_tmp=MSE(img1_np,img2_np)
            mse+=mse_tmp
            num+=1
    assert num>0
    mse=mse/num
    return mse