import shutil
import os
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np

'''该脚本缩放图像'''
'''该脚本暂不支持对中文路径文件夹进行操作'''


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

def resize_img(dir_old,dir_new):
    print("Flod data floder creating!")
    num=0

    imgs_old=make_dataset(dir_old)
    for img_old in tqdm(imgs_old):
        img_inner=get_inner_path(img_old,dir_old)
        photo_old=cv2.imdecode(np.fromfile(img_old,dtype=np.uint8),-1)
        if isinstance(photo_old,np.ndarray):
            photo_new=cv2.resize(photo_old,(256,256))
            img_new=os.path.join(dir_new,os.path.splitext(img_inner)[0]+'.png')
            if not os.path.isdir(os.path.split(img_new)[0]):
                os.makedirs(os.path.split(img_new)[0])
            cv2.imencode('.png',photo_new)[1].tofile(img_new)
            num+=1
    print("resize data floder created! %d img was processed"%num)

if __name__=="__main__":
    flag=1
    #首先解析文件路径
    path_old=r"D:\datasets\maps\all_A"
    path_new=r"D:\datasets\maps\all_A_resize256"

    #然后获取list
    imgs_old=make_dataset(path_old)

    resize_img(path_old,path_new)
    print("finish!")
