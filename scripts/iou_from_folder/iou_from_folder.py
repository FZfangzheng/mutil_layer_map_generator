from eval_iou_source_code import label_accuracy_score
from PIL import Image
import numpy as np
import os
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif','.npy'
]


def is_image_file(filename): # npy文件现在也算
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

if __name__=='__main__':
    real_seg_path=r'D:\map_translate\看看效果\0627两个重绘数据集试验\color1-结合方法\seg_result_gray'
    fake_seg_path=r'D:\map_translate\看看效果\0627两个重绘数据集试验\color2-结合方法\seg_result_gray'
    real_files=sorted(make_dataset(real_seg_path))
    fake_files = sorted(make_dataset(fake_seg_path))
    assert len(real_files)==len(fake_files)
    label_preds = []
    label_targets = []
    for i in range(len(real_files)):
        pred=Image.open(fake_files[i])
        pred=np.expand_dims(np.array(pred),axis=0)
        target=Image.open(real_files[i])
        target=np.expand_dims(np.array(target),axis=0)
        label_preds.append(pred)
        label_targets.append(target)
    _, _, iou, _, _ = label_accuracy_score(label_targets, label_preds, 3)
    print(f'===> iou score:{iou:.4f}')