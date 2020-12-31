import shutil
import os
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np

'''该脚本将GAN生成的rgb图根据预设色值按照最近邻关系转换为标准色值'''

stand_rgb =[[239,238,236],[255,255,255],[170,218,255],[227,227,227]]


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


def get_inner_path(file_path, floder_path):
    assert file_path[:len(floder_path)] == floder_path, "传入的文件不在文件夹中！[%s][%s]" % (file_path, floder_path)
    file_path = file_path[len(floder_path) + 1:]
    return file_path


# def getmask(img3D, rgb):
#     img3D = img3D.transpose((2, 0, 1))  # 变为 3 h w 方便后续操作
#     img3D[0][img3D[0] != rgb[0]] = 0
#     img3D[0][img3D[0] == rgb[0]] = 255
#     img3D[1][img3D[1] != rgb[1]] = 0
#     img3D[1][img3D[1] == rgb[1]] = 255
#     img3D[2][img3D[2] != rgb[2]] = 0
#     img3D[2][img3D[2] == rgb[2]] = 255
#     img3D = img3D.min(axis=0)
#     zuobiao = np.nonzero(img3D)
#     mask = np.zeros(img3D.shape)
#     mask[zuobiao] = 1
#     return mask
#
# def distance(a,b):
#     return

def make_stand_img(dir_old, dir_new):
    if not os.path.isdir(dir_new):
        os.makedirs(dir_new)
        print("Flod data floder creating!")
    num = 0

    imgs_old = make_dataset(dir_old)
    for img in tqdm(imgs_old):
        img_inner = get_inner_path(img, dir_old)
        old = Image.open(img)
        old = np.array(old)

        stand_rgb_imgs=[]
        new=[]
        for i in range(len(stand_rgb)):
            tmp=np.expand_dims(np.array(stand_rgb[i]), axis=0).repeat(256,axis=0)
            tmp=np.expand_dims(tmp, axis=0).repeat(256,axis=0)
            stand_rgb_imgs.append(tmp)
            new.append(np.expand_dims(((old-tmp)**2).sum(axis=2),axis=0))
        new=np.concatenate(new,axis=0)
        new=np.argsort(new,axis=0)
        new=new[0,:,:]
        new_img=np.zeros((old.shape[0],old.shape[1],3),dtype=np.uint8)
        new_img[np.where(new == 0)] = stand_rgb[0]
        new_img[np.where(new == 1)] = stand_rgb[1]
        new_img[np.where(new == 2)] = stand_rgb[2]
        new_img[np.where(new == 3)] = stand_rgb[3]

        if False: # 双边滤波
            new_img=cv2.bilateralFilter(new_img,7,30,30)

        new_img = Image.fromarray(new_img)
        new_img.save(os.path.join(dir_new, img_inner))
        num += 1
    print("New data floder created! %d img was processed" % num)


if __name__ == "__main__":
    flag = 1
    # 首先解析文件路径
    path_old = r'D:\map_translate\看看效果\0627两个重绘数据集试验\color1-结合方法\fake_result'
    # path_new = r"D:\map_translate\看看效果\0426TW16_1708图_celvs,epoch200\real_seg_new"
    path_new = path_old + '_tostand'

    make_stand_img(path_old, path_new)
    print("finish!")
