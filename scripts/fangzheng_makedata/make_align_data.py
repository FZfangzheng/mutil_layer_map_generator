from PIL import Image
import os
from PIL import ImageFile
import shutil
import math
import numpy as np
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

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


def make_inter_dataset():
    pt=[[224,39],[195,79],[138,157],[23,312]]
    h=256
    w=256
    image_dir_path = r"/data/multilayer_map_project/seg_repaint_for_show"
    #image_dir_path = r"/data/multilayer_map_project/out_mutil_layer_mix4_70_50epoch/newfake_result_for_show"
    id_layer=["14all","15all","16all"]
    # id_layer=["14all","15all","16all","17all"]
    for i in range(len(id_layer)):
        img_path = os.path.join(image_dir_path,id_layer[i],"all.png")
        img = Image.open(img_path)
        size=img.size
        print(size)
        box_list=[]
        len_x = math.floor((size[0]-pt[i][0])/w)
        len_y = math.floor((size[1]-pt[i][1])/h)
        print(len_x)
        print(len_y)
        for m in range(len_x):
            for n in range(len_y):
                box = (pt[i][0]+w*m, pt[i][1]+h*n, pt[i][0]+w*(m+1), pt[i][1]+h*(n+1))
                region = img.crop(box)
                region.save('/data/multilayer_map_project/align_data1_2/all_data/Seg/{}_{}_{}.png'.format(i+1,m+1,n+1))
                print('{},{}'.format(m,n))
def make_final_inter_dataset(path=""):
    path = r"/data/multilayer_map_project/align_data1_2/all_data/Seg"
    img_list = make_dataset(path)
    for img in img_list:
        img_name = os.path.split(img)[-1]
        img_name_list = img_name.split("_")
        path_layer = os.path.join(path,img_name_list[0])
        if not os.path.exists(path_layer):
            os.mkdir(path_layer)
        new_img_path = os.path.join(path_layer, img_name)
        shutil.copy(img,new_img_path)
        print(new_img_path)

def make_train_test():
    path=r"/data/multilayer_map_project/align_data1_2/all_data/Seg"
    path_train = r"/data/multilayer_map_project/align_data1_2/all_data/train_seg/Seg"
    path_test = r"/data/multilayer_map_project/align_data1_2/all_data/test_seg/Seg"

    layer="1"
    path1=os.path.join(path,layer)
    path_train1=os.path.join(path_train,layer)
    path_test1=os.path.join(path_test,layer)
    img_list = make_dataset(path1)
    for img in img_list:
        img_name = os.path.split(img)[1]
        x_line = int(img_name.split("_")[1])
        if x_line >=14:
            shutil.copy(img, os.path.join(path_test1,img_name))
        else:
            shutil.copy(img, os.path.join(path_train1, img_name))
    layer="2"
    path1=os.path.join(path,layer)
    path_train1=os.path.join(path_train,layer)
    path_test1=os.path.join(path_test,layer)
    img_list = make_dataset(path1)
    for img in img_list:
        img_name = os.path.split(img)[1]
        x_line = int(img_name.split("_")[1])
        if x_line >=27:
            shutil.copy(img, os.path.join(path_test1,img_name))
        else:
            shutil.copy(img, os.path.join(path_train1, img_name))
    layer="3"
    path1=os.path.join(path,layer)
    path_train1=os.path.join(path_train,layer)
    path_test1=os.path.join(path_test,layer)
    img_list = make_dataset(path1)
    for img in img_list:
        img_name = os.path.split(img)[1]
        x_line = int(img_name.split("_")[1])
        if x_line >=53:
            shutil.copy(img, os.path.join(path_test1,img_name))
        else:
            shutil.copy(img, os.path.join(path_train1, img_name))
    # layer="4"
    # path1=os.path.join(path,layer)
    # path_train1=os.path.join(path_train,layer)
    # path_test1=os.path.join(path_test,layer)
    # img_list = make_dataset(path1)
    # for img in img_list:
    #     img_name = os.path.split(img)[1]
    #     x_line = int(img_name.split("_")[1])
    #     if x_line >=105:
    #         shutil.copy(img, os.path.join(path_test1,img_name))
    #     else:
    #         shutil.copy(img, os.path.join(path_train1, img_name))

def test_merge():
    id_layer=3
    index_x=1
    index_y=1
    size=256
    dir_B=r"/data/multilayer_map_project/align_data/train/C/4"
    answer_dir=r"/data/multilayer_map_project/align_data/train/tmp_B"
    B1_path = os.path.join(dir_B,
                           str(id_layer + 1) + "_" + str(index_x * 2 - 1) + "_" + str(index_y * 2 - 1) + ".png")
    B2_path = os.path.join(dir_B,
                           str(id_layer + 1) + "_" + str(index_x * 2 - 1) + "_" + str(index_y * 2) + ".png")
    B3_path = os.path.join(dir_B,
                           str(id_layer + 1) + "_" + str(index_x * 2) + "_" + str(index_y * 2 - 1) + ".png")
    B4_path = os.path.join(dir_B,
                           str(id_layer + 1) + "_" + str(index_x * 2) + "_" + str(index_y * 2) + ".png")
    print(B1_path)
    B1 = Image.open(B1_path).convert('RGB')
    B1.save(os.path.join(answer_dir,"B1.png"))
    B2 = Image.open(B2_path).convert('RGB')
    B1.save(os.path.join(answer_dir, "B2.png"))
    B3 = Image.open(B3_path).convert('RGB')
    B1.save(os.path.join(answer_dir, "B3.png"))
    B4 = Image.open(B4_path).convert('RGB')
    B1.save(os.path.join(answer_dir, "B4.png"))

    B = Image.new('RGB', (2 * size, 2 * size))  # 创建一个新图
    B.paste(B1, (0 * size, 0 * size))
    B.paste(B2, (0 * size, 1 * size))
    B.paste(B3, (1 * size, 0 * size))
    B.paste(B4, (1 * size, 1 * size))
    B.save(os.path.join(answer_dir,
                        str(id_layer) + "_" + str(index_x) + "_" + str(index_y)) + "_x2.png")
    B = B.resize((size, size), Image.ANTIALIAS)
    B.save(os.path.join(answer_dir,
                        str(id_layer) + "_" + str(index_x) + "_" + str(index_y)) + ".png")
def test_seg(path):
    seg = Image.open(path)

    seg = np.asarray(seg)
    seg = torch.from_numpy(seg)  # H W
    print(seg.shape)
if __name__ == '__main__':
    # test_merge()
    # make_inter_dataset()
    # make_final_inter_dataset()
    # make_train_test()
    #make_final_inter_dataset("/data/multilayer_map_project/inter1_2/fake_result")
    test_seg("/data/multilayer_map_project/align_data1_6/train/Seg/1/1_1_1.png")
    # test_seg("/data/multilayer_map_project/seg_repaint_for_show/14/13712/2669.png")
