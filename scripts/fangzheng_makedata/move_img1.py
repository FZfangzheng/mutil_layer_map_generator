import os
import shutil
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


root_new_map = "/data/fangzheng/map_project/18-SH/good_img_map_repaint"
root_seg = "/data/fangzheng/map_project/18-SH/18_seg_nojz/new_seg"
trainB="/data/fangzheng/map_project/data/trainB"
train_seg="/data/fangzheng/map_project/data/train_seg"
def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

imgs_map_old = make_dataset(root_new_map)
for i in imgs_map_old:
    print(i)
    i = i.rstrip('\n')
    dir_list = i.split("/")
    img_path_map = os.path.join(root_seg, dir_list[6], dir_list[7])
    new_path_map = os.path.join(train_seg, dir_list[6]+"_"+dir_list[7])
    shutil.copy(img_path_map, new_path_map)
