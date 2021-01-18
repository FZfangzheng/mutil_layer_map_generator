import os
import shutil
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


root_new_map = r"D:\project\北理项目\map_project\上海\18_seg2\log\log\15\good_img_map_repaint"
root_new_rs = r"D:\project\北理项目\map_project\上海\18_seg2\log\log\15\good_img_rs"
root_seg = r"D:\project\北理项目\map_project\上海\18_seg2\15\15_seg\seg"
trainA = r"D:\project\北理项目\map_project\上海\18_seg2\log\log\15\data\trainA"
trainB=r"D:\project\北理项目\map_project\上海\18_seg2\log\log\15\data\trainB"
train_seg=r"D:\project\北理项目\map_project\上海\18_seg2\log\log\15\data\train_seg"
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
imgs_rs_old = make_dataset(root_new_rs)
for i in imgs_map_old:
    print(i)
    i = i.rstrip('\n')
    dir_list = i.split("\\")
    img_path_map = os.path.join(root_new_map, dir_list[10], dir_list[11])
    new_path_map = os.path.join(trainB, dir_list[10]+"_"+dir_list[11])
    shutil.copy(img_path_map, new_path_map)
for i in imgs_rs_old:
    print(i)
    i = i.rstrip('\n')
    dir_list = i.split("\\")
    img_path_map = os.path.join(root_new_rs, dir_list[10], dir_list[11])
    new_path_map = os.path.join(trainA, dir_list[10]+"_"+dir_list[11])
    shutil.copy(img_path_map, new_path_map)
for i in imgs_rs_old:
    print(i)
    i = i.rstrip('\n')
    dir_list = i.split("\\")
    img_path_map = os.path.join(root_seg, dir_list[10], dir_list[11])
    new_path_map = os.path.join(train_seg, dir_list[10]+"_"+dir_list[11])
    shutil.copy(img_path_map, new_path_map)
# with open(root_file) as f:
#     imgs_list = f.readlines()
# for i in imgs_list:
#     print(i)
#     i = i.rstrip('\n')
#     dir_list = i.split("\\")
#     x_dir_rs = os.path.join(root_new_rs, dir_list[7])
#     x_dir_map = os.path.join(root_new_map, dir_list[7])
#     if not os.path.exists(x_dir_rs):
#         os.mkdir(x_dir_rs)
#     if not os.path.exists(x_dir_map):
#         os.mkdir(x_dir_map)
#     img_path_rs = os.path.join(root_rs,dir_list[7], dir_list[8])
#     img_path_map = os.path.join(root_map,dir_list[7], dir_list[8])
#     new_path_rs = os.path.join(root_new_rs, dir_list[7], dir_list[8])
#     new_path_map = os.path.join(root_new_map, dir_list[7], dir_list[8])
#     shutil.copy(img_path_rs, new_path_rs)
#     shutil.copy(img_path_map, new_path_map)
