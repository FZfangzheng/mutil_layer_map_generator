import os
import shutil
img_paths=[]
new_path=""
img_range=[[21398,109788],[10699,54894],[5350,27447],[2675,13723]]
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


if not os.path.exists(new_path):
    os.mkdir(new_path)
for i in range(len(img_paths)):
    img_list = make_dataset(img_paths[i])
    for img in img_list:
        paths = os.path.split(img)[0].split(os.sep)
        if paths[-1]=="18":
            id = 0
        if paths[-1]=="17":
            id = 1
        if paths[-1] == "16":
            id = 2
        if paths[-1] == "15":
            id = 3
        new_imgdir_path2 = os.path.join(new_path, paths[-2])
        if not os.path.exists(new_imgdir_path2):
            os.mkdir(new_imgdir_path2)
        new_imgdir_path = os.path.join(new_imgdir_path2, paths[-1])
        if not os.path.exists(new_imgdir_path):
            os.mkdir(new_imgdir_path)
        new_img_path=os.path.join(new_imgdir_path, os.path.split(img)[1])
        img_names = os.path.split(img)[1].split('.')[0].split('-')[0].split('_')
        int_x = int(img_names[-2])
        int_y = int(img_names[-1])
        if int_x<img_range[id][0] and int_y<img_range[id][1]:
            shutil.copy(img, new_img_path)
            print(new_img_path)
