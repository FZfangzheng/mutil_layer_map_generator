import shutil
import os
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np

'''该脚本根据做好的分割图重绘rgb网络地图'''
'''所用颜色各通道控制在12.75~242.25之间'''


# 背景、道路、水域、道路描边

# new_rgb = [[240,240,240],[240,240,175],[170,218,240],[240,210,106]]
new_rgb = [[239,238,236],[255,255,255],[170,218,255],[227,227,227]]

# new_rgb = [[128,0,0],[0,128,0],[128,128,0],[227,227,227]] # 语义分割上色

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

def make_floder_from_file_path(file_path):
    if not os.path.isdir(os.path.split(file_path)[0]):
        os.makedirs(os.path.split(file_path)[0])

def isedge(road,x,y):
    idx=[[y-1,x-1],[y-1,x],[y-1,x+1],[y,x-1],[y,x+1],[y+1,x-1],[y-1,x],[y+1,x+1]]
def sobel_edge(img):# 接收seg 也就是label图
    sobelx = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1)
    sobely = cv2.convertScaleAbs(sobely)
    # sobelxy = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=1)
    # sobelxy = cv2.convertScaleAbs(sobelxy)
    result = cv2.addWeighted(sobelx, 1, sobely, 1, 0)
    # result = cv2.addWeighted(result, 1, sobelxy, 1, 0)
    result=np.array(result)
    return result

def get_road_edge(seg): # seg: ndarray 256*256 label灰度图，0为背景，1为道路，2为水域
    notroad=seg!=1
    notroad=notroad.astype(np.uint8)
    y_idx,x_idx=np.nonzero(notroad)
    edge=sobel_edge(notroad) # edge 不是0-1图像，有其他值，需要统一一下
    edge[np.nonzero(edge)]=1
    road_edge=edge & notroad
    return road_edge
    # for i in range(len(x_idx)):
    #     x=x_idx[i]
    #     y=y_idx[i]
    #     if isedge(road,x,y):
    #         pass




def make_rgb_gt(dir_seg, dir_new): # seg 文件为256*256灰度label图
    if not os.path.isdir(dir_new):
        os.makedirs(dir_new)
        print("Flod data floder creating!")
    num = 0

    imgs_old = make_dataset(dir_seg)
    for img in tqdm(imgs_old):
        img_inner = get_inner_path(img, dir_seg)
        old = Image.open(img)
        old=np.array(old)
        new=np.zeros((old.shape[0],old.shape[1],3),dtype=np.uint8)
        new[np.where(old==0)]=new_rgb[0]
        new[np.where(old == 1)] = new_rgb[1]
        new[np.where(old == 2)] = new_rgb[2]
        new[np.where(old == 3)] = new_rgb[0]
        new[np.where(old == 4)] = new_rgb[1]
        new[np.where(old == 5)] = new_rgb[0]
        new[np.where(old == 6)] = new_rgb[0]
        new[np.where(old == 7)] = new_rgb[1]
        new[np.where(old == 8)] = new_rgb[1]

        # road_edge=get_road_edge(old)
        # new[np.where(road_edge == 1)]=new_rgb[3]
        new=Image.fromarray(new)
        make_floder_from_file_path(os.path.join(dir_new, img_inner))
        new.save(os.path.join(dir_new, img_inner))
        num += 1
    print("New data floder created! %d img was processed" % num)


if __name__ == "__main__":
    flag = 1
    # 首先解析文件路径
    path_seg = r"/data/fangzheng/map_project/18-SH/good_img_map"
    # path_new = r"D:\map_translate\看看效果\0426TW16_1708图_celvs,epoch200\real_seg_new"
    path_new = path_seg+'_repaint'

    make_rgb_gt(path_seg, path_new)
    print("finish!")
