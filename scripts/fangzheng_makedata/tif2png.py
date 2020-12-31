from osgeo import gdal
import os

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
    
dir_tif = "D:\\project\\北理项目\\map_project\\18_seg2\\18"
dir_png = "D:\\project\\北理项目\\map_project\\18_seg2\\18_png"

if not os.path.isdir(dir_png):
    os.makedirs(dir_png)
    print("Flod data floder creating!")
dir_list = os.listdir(dir_tif)
for dir1 in dir_list:
    new_dir = os.path.join(dir_png, dir1)
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)

tif_dir = make_dataset(dir_tif)
for tif_img in tif_dir:
    img_inner=get_inner_path(tif_img,dir_tif).split('.')[0] + '.png'
    png_dir = os.path.join(dir_png,img_inner)
    print(tif_img)
    ds=gdal.Open(tif_img)
    driver=gdal.GetDriverByName('PNG')
    dst_ds = driver.CreateCopy(png_dir, ds)
    dst_ds = None
    src_ds = None