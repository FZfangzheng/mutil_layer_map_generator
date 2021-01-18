import os
import shutil
from tqdm import tqdm

def get_inner_path(file_path,floder_path):
    assert file_path[:len(floder_path)]==floder_path,"传入的文件不在文件夹中！[%s][%s]"%(file_path,floder_path)
    file_path=file_path[len(floder_path)+1:]
    return file_path
def make_floder_from_file_path(file_path):
    if not os.path.isdir(os.path.split(file_path)[0]):
        os.makedirs(os.path.split(file_path)[0])

if __name__=='__main__':
    train_ratio=0.7
    real_img_floder=r'D:\map_translate\数据集\SH分析\SH18\源数据\rs'
    txt_paths = [r'D:\map_translate\数据集\SH分析\SH18\pick03_SH.txt']
    txt_base_floder=r'D:\project\北理项目\map_project\18_seg2\overlap_image'
    new_floders=[r'D:\map_translate\数据集\SH分析\SH18\质量较好\rs']

    # txt_paths = [r'D:\map_translate\数据集\WH16分析\log\pick00.txt',r'D:\map_translate\数据集\WH16分析\log\pick01.txt',r'D:\map_translate\数据集\WH16分析\log\pick02.txt',r'D:\map_translate\数据集\WH16分析\log\pick03.txt',r'D:\map_translate\数据集\WH16分析\log\pick04.txt',r'D:\map_translate\数据集\WH16分析\log\pick05.txt']
    # new_floders = [r'D:\map_translate\数据集\WH16分析\按质量分类\0',r'D:\map_translate\数据集\WH16分析\按质量分类\1',r'D:\map_translate\数据集\WH16分析\按质量分类\2',r'D:\map_translate\数据集\WH16分析\按质量分类\3',r'D:\map_translate\数据集\WH16分析\按质量分类\4',r'D:\map_translate\数据集\WH16分析\按质量分类\5']

    assert len(txt_paths)==len(new_floders)
    # for floder in new_floders:
    #     if not os.path.isdir(floder):
    #         os.makedirs(floder)
    if not real_img_floder:
        pass
        # for i in range(len(txt_paths)):
        #     img_list=[]
        #     for row in open(txt_paths[i], 'r'):
        #         img_list.append(row[:len(row) - 1])
        #     print(len(img_list))
        #     for img in tqdm(img_list):
        #         shutil.copy(img,new_floders[i])
    else:
        for i in range(len(txt_paths)):
            img_list=[]
            for row in open(txt_paths[i], 'r'):
                img_list.append(row[:len(row) - 1])
                img_list[-1]=os.path.join(real_img_floder,get_inner_path(img_list[-1],txt_base_floder))
                img_list[-1]=os.path.splitext(img_list[-1])[0]+'.tif'
            print(len(img_list))
            for j in tqdm(range(len(img_list))):
                img = img_list[j]
                if j<(len(img_list)*train_ratio):
                    img_new=get_inner_path(img,real_img_floder)
                    img_new=os.path.join(new_floders[i]+'_train',img_new)
                    make_floder_from_file_path(img_new)
                    shutil.copy(img,img_new)
                else:
                    img_new = get_inner_path(img, real_img_floder)
                    img_new = os.path.join(new_floders[i] + '_test', img_new)
                    make_floder_from_file_path(img_new)
                    shutil.copy(img, img_new)