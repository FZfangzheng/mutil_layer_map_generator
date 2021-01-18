import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from hough_transform import one_seg2message, visual_theta_rho
from road_connect2 import road_connect
from skimage import morphology
from message_from_connection import line2theta,line2road,theta2wid

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


def get_inner_path(file_path, floder_path):
    assert file_path[:len(floder_path)] == floder_path, "传入的文件不在文件夹中！[%s][%s]" % (file_path, floder_path)
    file_path = file_path[len(floder_path) + 1:]
    return file_path


def segs2messages_hough(seg_path, save_path, save_path2):
    # 基于霍夫变换识别道路角度与宽度
    list_seg = make_dataset(seg_path)
    for seg_file in tqdm(list_seg):
        seg = Image.open(seg_file)
        seg_np_vis = np.array(seg)
        seg = np.array(seg).transpose((2, 0, 1))
        # 转换为只有道路与非道路的图
        # [255,255,255] 为道路
        seg[0][seg[0] != 255] = 0
        seg[0][seg[0] == 255] = 255
        seg[1][seg[1] != 255] = 0
        seg[1][seg[1] == 255] = 255
        seg[2][seg[2] != 255] = 0
        seg[2][seg[2] == 255] = 255
        seg = seg.min(axis=0)
        mask, thetas, rhos, roadwids = one_seg2message(seg, save_path)
        img_inner = get_inner_path(seg_file, seg_path)
        vis = visual_theta_rho(seg_np_vis, mask, thetas, rhos, roadwids)
        vis = Image.fromarray(vis)
        vis.save(os.path.join(save_path, img_inner))

        road_mask, road_line_old, road_endpoints, road_line = road_connect(seg_np_vis, mask, thetas, rhos, roadwids)
        mask_ = Image.fromarray(mask.astype(np.uint8) * 255).convert('1')
        mask_.save(os.path.join(save_path2, 'mask' + img_inner))
        road_line_old = Image.fromarray(road_line_old.astype(np.uint8) * 255).convert(mode='1')
        road_line_old.save(os.path.join(save_path2, 'old' + img_inner))
        road_endpoints = Image.fromarray(road_endpoints.astype(np.uint8) * 255).convert(mode='1')
        road_endpoints.save(os.path.join(save_path2, 'end' + img_inner))
        road_line = Image.fromarray(road_line.astype(np.uint8) * 255).convert(mode='1')
        road_line.save(os.path.join(save_path2, 'ret' + img_inner))
        # seg_connact_vis,seg_isend=road_connect(seg_np_vis,mask,thetas,rhos,roadwids)
        # seg_connact_vis=Image.fromarray(seg_connact_vis)
        # seg_connact_vis.save(os.path.join(save_path2,img_inner))
        # seg_isend[seg_isend!=0]=255
        # seg_isend=Image.fromarray(seg_isend).convert(mode='1')
        # seg_isend.save(os.path.join(save_path2,'end_'+img_inner))

def segs2messages_connect(seg_path, save_path, save_path2):
    # 焦点的判断可采用八邻域信息判断
    if not os.path.isdir(os.path.join(save_path,'theta')):
        os.makedirs(os.path.join(save_path,'theta'))
    if not os.path.isdir(os.path.join(save_path,'wid')):
        os.makedirs(os.path.join(save_path,'wid'))

    list_seg = make_dataset(seg_path)
    for seg_file in tqdm(list_seg):
        seg = Image.open(seg_file)
        seg=np.array(seg)
        seg_np_vis = np.array(seg)
        if len(seg.shape)==3:
            seg = seg.transpose((2, 0, 1))
            # 转换为只有道路与非道路的图
            # [255,255,255] 为道路
            seg[0][seg[0] != 255] = 0
            seg[0][seg[0] == 255] = 255
            seg[1][seg[1] != 255] = 0
            seg[1][seg[1] == 255] = 255
            seg[2][seg[2] != 255] = 0
            seg[2][seg[2] == 255] = 255
            seg = seg.min(axis=0)
        elif len(seg.shape)==2:
            # 灰度label图，1为道路
            seg[seg!=1]=0
        road_mask = (seg != 0)
        road_line = morphology.skeletonize(road_mask)
        road_line_theta,road_line_cross=line2theta(road_line)
        road_theta=line2road(road_mask,road_line,road_line_cross,road_line_theta)
        road_wid=theta2wid(road_mask,road_theta)


        img_inner = get_inner_path(seg_file, seg_path)
        np.save(os.path.join(save_path,'theta', os.path.splitext(img_inner)[0]),road_theta)
        np.save(os.path.join(save_path,'wid', os.path.splitext(img_inner)[0]),road_wid)
        vis = visual_theta_rho(seg_np_vis,road_mask, road_theta, None,road_wid)
        vis = Image.fromarray(vis)
        vis.save(os.path.join(save_path, img_inner))
        road_line = Image.fromarray(road_line.astype(np.uint8) * 255).convert(mode='1')
        road_line.save(os.path.join(save_path2, 'line' + img_inner))
        road_line_cross = Image.fromarray(road_line_cross.astype(np.uint8) * 255).convert(mode='1')
        road_line_cross.save(os.path.join(save_path2, 'cross' + img_inner))
        pass


if __name__ == '__main__':
    a = np.array([[1, 0], [0, 0]])

    b = a[0, 0]

    # seg_path = r"D:\map_translate\看看效果\0426TW16_1708图_celvs,epoch200\real_seg"
    seg_path = r"D:\map_translate\数据集\TW16分析\整体kmeans\按质量分类\3\train_seg"
    save_path = r"D:\map_translate\看看效果\道路信息tmp2"
    # seg_path = r"D:\map_translate\看看效果\0426TW16_1708图_celvs,epoch200\seg_result"
    # save_path = r"D:\map_translate\看看效果\道路信息2"
    save_path2 = r"D:\map_translate\看看效果\道路信息tmp连接图2"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.isdir(save_path2):
        os.makedirs(save_path2)
    # segs2messages_hough(seg_path,save_path,save_path2)
    segs2messages_connect(seg_path,save_path,save_path2)