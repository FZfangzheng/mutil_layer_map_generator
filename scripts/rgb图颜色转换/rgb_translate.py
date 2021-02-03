import shutil
import os
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np

'''该脚本将rgb图的某个特定颜色转换为另一个特定颜色，目前主要是将黄色道路转为白色（黄色视觉效果太晃眼）'''


# old_rgb = np.array([255, 242, 175])
# new_rgb = np.array([255, 255, 255])
new_rgb = np.array([255, 242, 175])
old_rgb = np.array([255, 255, 255])

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


def make_change_img(dir_old, dir_new):
    if not os.path.isdir(dir_new):
        os.makedirs(dir_new)
        print("Flod data floder creating!")
    num = 0

    imgs_old = make_dataset(dir_old)
    for img in tqdm(imgs_old):
        img_inner = get_inner_path(img, dir_old)
        old = Image.open(img)
        old_ = np.array(old).transpose((2, 0, 1))  # 变为 3 h w 方便后续操作
        old = np.array(old)
        old_[0][old_[0] != old_rgb[0]] = 0
        old_[0][old_[0] == old_rgb[0]] = 255
        old_[1][old_[1] != old_rgb[1]] = 0
        old_[1][old_[1] == old_rgb[1]] = 255
        old_[2][old_[2] != old_rgb[2]] = 0
        old_[2][old_[2] == old_rgb[2]] = 255
        old_=old_.min(axis=0)
        zuobiao=np.nonzero(old_)
        old[zuobiao]=new_rgb
        new=old
        new=Image.fromarray(new)
        new.save(os.path.join(dir_new,img_inner))
        num+=1
    print("New data floder created! %d img was processed" % num)


if __name__ == "__main__":
    flag = 1
    # 首先解析文件路径
    path_old = r"/data/multilayer_map_project/new_data/map"
    # path_new = r"D:\map_translate\看看效果\0426TW16_1708图_celvs,epoch200\real_seg_new"
    path_new = path_old+'_new'
    make_change_img(path_old, path_new)
    print("finish!")
