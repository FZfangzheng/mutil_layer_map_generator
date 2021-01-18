import shutil
import os
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np

'''该脚本重命名图像'''


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif',
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

def rename_img(dir):
    print("Flod data floder creating!")
    num=0

    imgs_path=make_dataset(dir)
    for img_path in tqdm(imgs_path):
        os.rename(img_path,os.path.splitext(img_path)[0][:-2]+'.png')
        num+=1
    print("resize data floder created! %d img was processed"%num)

if __name__=="__main__":
    flag=1
    #首先解析文件路径
    path=r"D:\datasets\maps\all_seg_8label\repaint_color1"

    # # 然后获取list
    # imgs_old=make_dataset(path)

    rename_img(path)
    print("finish!")
