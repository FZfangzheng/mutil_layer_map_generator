import os
import shutil
from tqdm import tqdm
import json

def get_inner_path(file_path,floder_path):
    assert file_path[:len(floder_path)]==floder_path,"传入的文件不在文件夹中！[%s][%s]"%(file_path,floder_path)
    file_path=file_path[len(floder_path)+1:]
    return file_path
def make_floder_from_file_path(file_path):
    if not os.path.isdir(os.path.split(file_path)[0]):
        os.makedirs(os.path.split(file_path)[0])

if __name__=='__main__':
    train_ratio=0.7
    real_img_floder=r'D:\map_translate\数据集\WH16分析\二次筛选相关\所用数据集文件夹\test'
    json_paths = r'D:\map_translate\code\p2pHD_about\p2pHD-change-seg\GAN\scripts\根据结果自动筛选\result.json'
    json_base_floder=r'D:\map_translate\数据集\WH16分析\二次筛选相关\6.2生成结果\real_result'
    new_floders=r'D:\map_translate\数据集\WH16分析\二次筛选相关\二次筛选结果'

    # for floder in new_floders:
    #     if not os.path.isdir(floder):
    #         os.makedirs(floder)

    with open(json_paths) as f:
        imgs_paths=json.load(f)

    imgs_paths=imgs_paths[300:]
    imgs_paths=sorted(imgs_paths)
    for i in tqdm(range(len(imgs_paths))):
        path=imgs_paths[i]
        path=get_inner_path(path,json_base_floder)
        path=os.path.splitext(path)[0]
        row=path[:5]
        col=path[5:]
        old_path=real_img_floder+os.sep+row+os.sep+col+'.png'
        if i < len(imgs_paths)*train_ratio:
            new_path=new_floders+os.sep+'使用的2700图'+os.sep+'train'+os.sep+row+os.sep+col+'.tif'
        else:
            new_path = new_floders + os.sep + '使用的2700图' + os.sep + 'test' + os.sep + row + os.sep + col + '.tif'
        make_floder_from_file_path(new_path)
        shutil.copy(old_path, new_path)

