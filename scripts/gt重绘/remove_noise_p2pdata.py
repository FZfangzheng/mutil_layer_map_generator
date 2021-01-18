import shutil
import os
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np
from skimage import morphology,measure

'''根据连通域大小去除噪点'''



# 背景、道路、水域、道路描边


new_rgb = [[239,238,236],[255,255,255],[170,218,255],[227,227,227]]


def save_max_objects(img):
    labels = measure.label(img)  # 返回打上标签的img数组
    jj = measure.regionprops(labels)  # 找出连通域的各种属性。  注意，这里jj找出的连通域不包括背景连通域
    # is_del = False
    if len(jj) == 1:
        out = img
        # is_del = False
    else:
        # 通过与质心之间的距离进行判断
        num = labels.max()  #连通域的个数
        del_array = np.array([0] * (num + 1))#生成一个与连通域个数相同的空数组来记录需要删除的区域（从0开始，所以个数要加1）
        for k in range(num):
            if k == 0:
                initial_area = jj[0].area
                save_index = 1  # 初始保留第一个连通域
            else:
                k_area = jj[k].area  # 将元组转换成array

                if initial_area < k_area:
                    initial_area = k_area
                    save_index = k + 1

        del_array[save_index] = 1
        del_mask = del_array[labels]
        out = img * del_mask
        # is_del = True
    return out

def make_floder_from_file_path(file_path):
    if not os.path.isdir(os.path.split(file_path)[0]):
        os.makedirs(os.path.split(file_path)[0])

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

def get_road_edge(seg): # seg: ndarray 256*256 label灰度图，0为背景，1为主干道路，2为水域
    notroad = seg != 1
    notroad = notroad.astype(np.uint8)
    y_idx, x_idx = np.nonzero(notroad)
    edge = sobel_edge(notroad)  # edge 不是0-1图像，有其他值，需要统一一下
    edge[np.nonzero(edge)] = 1
    road_edge = edge & notroad
    return road_edge
    # for i in range(len(x_idx)):
    #     x=x_idx[i]
    #     y=y_idx[i]
    #     if isedge(road,x,y):
    #         pass




def make_remove_noise_gt(dir_seg, dir_new): # seg 文件为256*256灰度label图
    if not os.path.isdir(dir_new):
        os.makedirs(dir_new)
        print("Flod data floder creating!")
    num = 0

    imgs_old = make_dataset(dir_seg)
    for img in tqdm(imgs_old):
        img_inner = get_inner_path(img, dir_seg)
        old = Image.open(img)
        old=np.array(old)

        # 类别融合
        old[old==0]=0 # 主要的背景
        old[old==1]=0 # 主要是建筑物
        old[old==2]=1
        old[old==3]=1
        old[old==4]=1
        old[old==5]=0
        old[old==6]=2
        old[old==7]=0

        # #去除噪声
        # old_mask0=np.array(old==0).astype(np.bool)
        # old_mask0_=morphology.remove_small_objects(old_mask0,min_size=25)
        # old_mask1=np.array(old==1).astype(np.bool)
        # old_mask1_=morphology.remove_small_objects(old_mask1,min_size=25)
        # old[np.logical_and(old_mask0 ,np.logical_not(old_mask0_))]=1
        # old[np.logical_and(old_mask1 ,np.logical_not(old_mask1_))]=0


        # new=np.zeros((old.shape[0],old.shape[1],3),dtype=np.uint8)
        # new[np.where(old==0)]=new_rgb[0]
        # new[np.where(old == 1)] = new_rgb[1]
        # new[np.where(old == 2)] = new_rgb[2]
        new = np.zeros((old.shape[0], old.shape[1]), dtype=np.uint8)
        new[np.where(old == 0)] = 0
        new[np.where(old == 1)] = 1
        new[np.where(old == 2)] = 2


        # road_edge=get_road_edge(old)
        # new[np.where(road_edge == 1)]=new_rgb[3]
        new=Image.fromarray(new)
        make_floder_from_file_path(os.path.join(dir_new, img_inner))
        new.save(os.path.join(dir_new, img_inner))

        # background=(old==0).astype(np.uint8)
        # background[background==1]=255
        # # background=np.expand_dims(background,2).repeat(3,axis=2)
        # background=Image.fromarray(background)
        # background.save(os.path.join(dir_new, os.path.splitext(img_inner)[0]+'background'+'.png'))
        num += 1
    print("New data floder created! %d img was processed" % num)


if __name__ == "__main__":
    flag = 1
    # 首先解析文件路径
    path_seg = r"D:\datasets\maps\all_seg_8label\seg"
    path_new = r"D:\datasets\maps\all_seg_8label\repaint_color1\seg"

    make_remove_noise_gt(path_seg, path_new)
    print("finish!")
