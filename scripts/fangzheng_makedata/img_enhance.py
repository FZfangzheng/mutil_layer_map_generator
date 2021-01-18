import os
from PIL import Image

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
    root_path=r""
    all_img = make_dataset(root_path)
    for img_path in all_img:
        pwd_name = os.path.split(img_path)
        pwd = pwd_name[0]
        name = pwd_name[1]
        im = Image.open(img_path)
        im_rotate90 = im.rotate(90)
        im_rotate90.save(os.path.join(pwd, "rotate90_"+name))
        im_rotate180 = im.rotate(180)
        im_rotate180.save(os.path.join(pwd, "rotate180_" + name))
        im_rotate270 = im.rotate(270)
        im_rotate270.save(os.path.join(pwd, "rotate270_" + name))
        im_flip_left_right = im.transpose(Image.FLIP_LEFT_RIGHT)
        im_flip_left_right.save(os.path.join(pwd, "fliplr_" + name))
        im_flip_top_bottom = im.transpose(Image.FLIP_TOP_BOTTOM)
        im_flip_top_bottom.save(os.path.join(pwd, "fliptb_" + name))


