import shutil
import os
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np

'''拼合瓦片地图，暂定以5x5尺寸拼合；每张图右、下划黑以显示边界'''
'''遥感影像同样处理以作对比'''
'''该脚本忽略了不能整除的部分'''


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

class rowcol():
    def __init__(self):
        self.row_max = 0
        self.row_min = 999999999
        self.col_max = 0
        self.col_min = 999999999
    def update(self,row,col):
        if self.row_max<row:
            self.row_max=row
        if self.row_min>row:
            self.row_min=row
        if self.col_max<col:
            self.col_max=col
        if self.col_min>col:
            self.col_min=col

def make_big_img(dir_A,dir_B,dir_AB,n=5):
    print("Big data floder creating!")
    num=0

    imgs_A=make_dataset(dir_A)
    imgs_B=make_dataset(dir_B)
    assert len(imgs_A)==len(imgs_B)
    # 针对当前文件格式进行行列识别
    rc=rowcol()
    for img_A in tqdm(imgs_A):
        img_name=os.path.split(img_A)[1]
        img_name=os.path.splitext(img_name)[0]
        img_name=img_name.split('-')
        row=int(img_name[2])
        col=int(img_name[3])
        rc.update(row,col)
    base_img_name=os.path.split(imgs_A[0])[1]
    base_img_name,base_ext_name=os.path.splitext(base_img_name)
    base_img_name=base_img_name.split('-') #分四段，不包括最后的png

    for row in tqdm(range(rc.row_min,rc.row_max-n+1,n)): # 最边上不足一组的就不要了
        for col in tqdm(range(rc.col_min,rc.col_max-n+1,n)):
            photos_A=[]
            photos_B=[]
            for i in range(0,n):
                for j in range(0,n):
                    img_A=os.path.join(dir_A,base_img_name[0]+'-'+base_img_name[1]+'-'+str(row+i)+'-'+str(col+j)+base_ext_name)
                    photo_A =np.fromfile(img_A,dtype=np.uint8)
                    photo_A=cv2.imdecode(photo_A,-1)
                    photo_A[:,-1,:]=0
                    photo_A[-1,:,:]=0
                    photos_A.append(photo_A)
                    img_B = os.path.join(dir_B,base_img_name[0] + '-' + base_img_name[1] + '-' + str(row + i) + '-' + str(col + j) + base_ext_name)
                    photo_B = np.fromfile(img_B, dtype=np.uint8)
                    photo_B = cv2.imdecode(photo_B, -1)
                    photo_B[:, -1, :] = 0
                    photo_B[-1, :, :] = 0
                    photos_B.append(photo_B)
            # 以上得到的为列优先排序
            photo_final_A=None
            photo_final_B=None
            for i in range(0,n):
                photo_A=photos_A[0+i*n]
                photo_B=photos_B[0+i*n]
                for j in range(1,n):
                    photo_A=np.concatenate((photo_A,photos_A[j+i*n]),1)  # 列优先拼合
                    photo_B=np.concatenate((photo_B,photos_B[j+i*n]),1)  # 列优先拼合
                if photo_final_A is None:
                    photo_final_A=photo_A
                    photo_final_B=photo_B
                else:
                    photo_final_A=np.concatenate((photo_final_A,photo_A),0)
                    photo_final_B=np.concatenate((photo_final_B,photo_B),0)
            photo_final=np.concatenate((photo_final_A,photo_final_B),1)
            path_final=os.path.join(dir_AB,base_img_name[0]+'-'+base_img_name[1]+'-'+str(row)+'-'+str(col)+base_ext_name)
            if not os.path.isdir(os.path.split(path_final)[0]):
                os.makedirs(os.path.split(path_final)[0])
            cv2.imencode(base_ext_name, photo_final)[1].tofile(path_final)

    print("Big img floder created! %d img was processed"%num)

if __name__=="__main__":
    flag=1
    #首先解析文件路径
    path=r"D:\map_translate\看看效果\0613使用全集生成拼合\16"
    path_a=os.path.join(path,r'real_A')
    path_b = os.path.join(path, r'fake_B')
    path_new=os.path.join(path,'拼合示例')

    #然后获取list
    # imgs_a=make_dataset(path_a)
    # imgs_b = make_dataset(path_b)

    make_big_img(path_a,path_b,path_new)
    print("finish!")
