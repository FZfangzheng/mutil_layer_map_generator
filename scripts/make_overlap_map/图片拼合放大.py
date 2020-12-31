import shutil
import os
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np

'''该脚本制作拼接放大图'''
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

def make_flod_img(dir_A,dir_B,dir_AB):
    print("Flod data floder creating!")
    num=0

    imgs_A=make_dataset(dir_A)
    imgs_B=make_dataset(dir_B)
    for img_A in tqdm(imgs_A):
        img_inner=get_inner_path(img_A,dir_A)
        img_inner_B=os.path.splitext(img_inner)[0]+'.tif'
        if os.path.join(dir_B,img_inner_B) in imgs_B:
            photo_A=cv2.imdecode(np.fromfile(img_A,dtype=np.uint8),-1)
            photo_B=cv2.imdecode(np.fromfile(os.path.join(dir_B,img_inner_B),dtype=np.uint8),-1)
            if isinstance(photo_A,np.ndarray) and photo_A.shape==photo_B.shape:
                # photo_AB=np.uint8(photo_A*0.5+photo_B*0.5)
                photo_AB=np.concatenate((photo_A,photo_B),1)
                img_AB=os.path.join(dir_AB,img_inner)
                if not os.path.isdir(os.path.split(img_AB)[0]):
                    os.makedirs(os.path.split(img_AB)[0])
                photo_AB=cv2.resize(photo_AB,(0,0),fx=2,fy=2)
                cv2.imencode('.png',photo_AB)[1].tofile(img_AB)
                num+=1
    print("Flod data floder created! %d img was processed"%num)

if __name__=="__main__":
    flag=1
    #首先解析文件路径
    # path=r"D:\map_translate\数据集\WH16分析"
    path_a=r"D:\map_translate\数据集\BJ18分析\8k张\A_train"
    path_b = r"D:\map_translate\数据集\BJ18分析\8k张\B_train"
    path_new=r"D:\map_translate\数据集\BJ18分析\8k张\floder_train"

    #然后获取list
    imgs_a=make_dataset(path_a)
    imgs_b = make_dataset(path_b)

    make_flod_img(path_a,path_b,path_new)
    print("finish!")
