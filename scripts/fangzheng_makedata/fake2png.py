import os
import shutil
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

if __name__ == "__main__":
    dir_png3 = r"C:\Users\WhiteBl\Desktop\new_tile_png"
    dir_png = r"C:\Users\WhiteBl\Desktop\new_tile"

    if not os.path.isdir(dir_png3):
        os.makedirs(dir_png3)
        print("Flod data floder creating!")
    dir_list = os.listdir(dir_png)
    for dir1 in dir_list:
        new_dir = os.path.join(dir_png3, dir1)
        if not os.path.isdir(new_dir):
            os.mkdir(new_dir)
    png_dir = make_dataset(dir_png)
    for png_img in png_dir:
        img_inner=get_inner_path(png_img,dir_png).split('.')[0] + '.png'
        png3_dir = os.path.join(dir_png3,img_inner)
        # image = Image.open(png_img)
        # new_image = change_image_channels(image, png3_dir)
        shutil.copy(png_img, png3_dir)
        # print(image.mode)
        # print(new_image.mode)