import PIL.Image as Image
import os
import shutil
IMAGES_PATH = 'D:\\PycharmDOC\\divide_test_photo\\s3\\'  # 图片集地址
IMAGES_FORMAT = ['.jpg', '.tif']  # 图片格式
IMAGE_SIZE = 256  # 每张小图片的大小
IMAGE_ROW = 8  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 8  # 图片间隔，也就是合并成一张图后，一共有几列
IMAGE_SAVE_PATH = 'D:\\PycharmDOC\\divide_test_photo\\pj.tif'  # 图片转换后的地址

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

# 定义图像拼接函数
def image_compose(img_path):
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(img_path[x-1][y-1])
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
    return to_image.save(IMAGE_SAVE_PATH)  # 保存新图


if __name__ == "__main__":
    save_path = r"C:\Users\WhiteBl\Desktop\SH-fake\merge17"
    map_l_path = r"C:\Users\WhiteBl\Desktop\SH-fake\new_tile_png\17"
    map_h_path = r"C:\Users\WhiteBl\Desktop\SH-fake\new_tile_png\14"
    image_list_low = make_dataset(map_l_path)
    image_list_high = make_dataset(map_h_path)
    min_x1 = 1000000
    min_y1 = 1000000
    max_x1 = 0
    max_y1 = 0
    for img_low in image_list_low:
        image_names = img_low.split("\\")
        x = image_names[7]
        y = image_names[8].split(".")[0]
        if int(x) > max_x1:
            max_x1 = int(x)
        if int(x) < min_x1:
            min_x1 = int(x)
        if int(y) > max_y1:
            max_y1 = int(y)
        if int(y) < min_y1:
            min_y1 = int(y)

    min_x2 = 1000000
    min_y2 = 1000000
    max_x2 = 0
    max_y2 = 0
    for img_high in image_list_high:
        image_names = img_high.split("\\")
        x = image_names[7]
        y = image_names[8].split(".")[0]
        if int(x) > max_x2:
            max_x2 = int(x)
        if int(x) < min_x2:
            min_x2 = int(x)
        if int(y) > max_y2:
            max_y2 = int(y)
        if int(y) < min_y2:
            min_y2 = int(y)

    len_x1 = max_x1 - min_x1 + 1
    len_y1 = max_y1 - min_y1 + 1
    len_x2 = max_x2 - min_x2 + 1
    len_y2 = max_y2 - min_y2 + 1

    print(len_x1)
    print(len_y1)
    print(len_x2)
    print(len_y2)

    ratio_x = len_x1 / len_x2
    ratio_y = len_y1 / len_y2
    img_count = 0
    for i in range(len_x2):
        x = min_x2 + i
        for j in range(len_y2):
            y = min_y2 + j
            low_i = i * ratio_x
            low_j = j * ratio_y
            low_x1 = round(min_x1 + low_i)
            low_y1 = round(min_y1 + low_j)
            low_x2 = round(low_x1 + ratio_x)
            low_y2 = round(low_y1 + ratio_y)
            IMAGE_ROW = low_y2-low_y1
            IMAGE_COLUMN = low_x2-low_x1
            img_path = []
            for x1 in range(low_x1, low_x2):
                y_img_path = []
                for y1 in range(low_y1, low_y2):
                    imgs = os.path.join(map_l_path, str(x1), str(y1)+".png")
                    y_img_path.append(imgs)
                img_path.append(y_img_path)
            new_img_name = str(img_count)+".png"
            img_new_14 = r"C:\Users\WhiteBl\Desktop\SH-fake\merge14"
            img_14_path = os.path.join(map_h_path, str(x), str(y)+".png")
            img_new_14_path = os.path.join(img_new_14, new_img_name)
            print(img_14_path)
            if not os.path.exists(img_new_14_path):
                shutil.copy(img_14_path, img_new_14_path)
            IMAGE_SAVE_PATH = os.path.join(save_path, new_img_name)
            image_compose(img_path)
            img_count += 1
    # 获取图片集地址下的所有图片名称
    # image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
    #                os.path.splitext(name)[1] == item]
    #
    # # 简单的对于参数的设定和实际图片集的大小进行数量判断
    # if len(image_names) != IMAGE_ROW * IMAGE_COLUMN:
    #     raise ValueError("合成图片的参数和要求的数量不能匹配！")