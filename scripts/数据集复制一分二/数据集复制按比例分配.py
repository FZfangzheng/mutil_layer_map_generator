import os
import shutil
from tqdm import tqdm

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
def make_floder_from_file_path(file_path):
    if not os.path.isdir(os.path.split(file_path)[0]):
        os.makedirs(os.path.split(file_path)[0])

if __name__=='__main__':
    A_ratio=0.5
    source_img_floder=r'D:\map_translate\数据集\HN14分析\使用数据-repaint1-用于s2omgan-50%非对齐\trainB'
    new_img_floder=r'D:\map_translate\数据集\HN14分析\使用数据-repaint1-用于s2omgan-50%非对齐\trainB-分开'

    num = 0
    imgs_path = make_dataset(source_img_floder)
    total_num=len(imgs_path)
    A_num=int(total_num*A_ratio)
    for i in tqdm(range(total_num)):
        if i<A_num:
            path_new=os.path.join(new_img_floder+'_a',get_inner_path(imgs_path[i],source_img_floder))
        else:
            path_new=os.path.join(new_img_floder+'_b',get_inner_path(imgs_path[i],source_img_floder))
        make_floder_from_file_path(path_new)
        shutil.copy(imgs_path[i], path_new)
        num += 1

    print("data copy complete! %d img was processed, %d of A, %d of B" % (num,A_num,total_num-A_num))
