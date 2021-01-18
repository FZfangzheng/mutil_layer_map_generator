import numpy as np
from tqdm import tqdm
import os
from PIL import Image

def PSNRLossnp(y_true, y_pred):
    return 10 * np.log(255 * 2 / (np.mean(np.square(y_pred - y_true))))


def SSIMnp(y_true, y_pred):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01 * 255)
    c2 = np.square(0.03 * 255)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
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
def get_inner_path(file_path,floder_path):
    assert file_path[:len(floder_path)]==floder_path,"传入的文件不在文件夹中！[%s][%s]"%(file_path,floder_path)
    file_path=file_path[len(floder_path)+1:]
    return file_path

def SSIM_from_floder(path1,path2): # 假设所有图片尺寸相同
    ssim=0
    num=0
    imgs1=make_dataset(path1)
    imgs2=make_dataset(path2)
    for img1 in tqdm(imgs1):
        img_inner = get_inner_path(img1, path1)
        if os.path.join(path2,img_inner) in imgs2:
            img2=os.path.join(path2,img_inner)
            img1_PIL=Image.open(img1)
            img2_PIL=Image.open(img2)
            img1_np=np.array(img1_PIL)
            img2_np=np.array(img2_PIL)
            img1_np = rgb2gray(img1_np)
            img2_np = rgb2gray(img2_np)

            ssim_tmp=SSIMnp(img1_np,img2_np)
            ssim+=ssim_tmp
            num+=1
    assert num>0
    ssim=ssim/num
    return ssim

if __name__=='__main__':
    path1=r'D:\map_translate\看看效果\0704重绘方式1的补充试验\数据集对比试验\p2pHD模型-在TW16上训练&生成\f-test'
    path2=r'D:\map_translate\看看效果\0704重绘方式1的补充试验\数据集对比试验\p2pHD模型-在TW16上训练&生成\r-test'
    ssim=SSIM_from_floder(path1,path2)
    print(ssim)