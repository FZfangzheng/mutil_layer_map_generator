# from test import compute_ssim
import numpy as np
import os
from PIL import Image
from scipy.signal import convolve2d

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)
def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")
    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))
    ssim_mu_map = ((2 * mu1_mu2 + C1)) / ((mu1_sq + mu2_sq + C1) )
    ssim_sigma_map = ( (2 * sigmal2 + C2)) / ( (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map)),np.mean(np.mean(ssim_mu_map)),np.mean(np.mean(ssim_sigma_map))

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
    ssim_mu=0
    ssim_sigma=0
    num=0
    imgs1=make_dataset(path1)
    imgs2=make_dataset(path2)
    from tqdm import  tqdm
    for img1 in tqdm(imgs1):
        img_inner = get_inner_path(img1, path1)
        if os.path.join(path2,img_inner) in imgs2:
            img2=os.path.join(path2,img_inner)
            img1_PIL=Image.open(img1)
            img2_PIL=Image.open(img2)
            img1_np=np.array(img1_PIL)
            img2_np=np.array(img2_PIL)
            # RGB to 灰度
            if len(img1_np.shape)>2:
                img1_np=rgb2gray(img1_np)
            if len(img2_np.shape)>2:
                img2_np=rgb2gray(img2_np)
            ssim_tmp,ssim_mu_tmp,ssim_sigma_tmp=compute_ssim(img1_np,img2_np)
            ssim+=ssim_tmp
            ssim_mu+=ssim_mu_tmp
            ssim_sigma+=ssim_sigma_tmp
            num+=1
    assert num>0
    ssim=ssim/num
    ssim_mu/=num
    ssim_sigma/=num
    return ssim,ssim_mu,ssim_sigma