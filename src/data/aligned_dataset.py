import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
from util.my_util import get_inner_path
import cv2
import numpy as np
import math
from tqdm import tqdm


def make_align_img(dir_A,dir_B,dir_AB):
    print("Align data floder creating!")
    num=0
    imgs_A=sorted(make_dataset(dir_A))
    imgs_B=sorted(make_dataset(dir_B))
    imgs_A_=[]
    imgs_B_=[]
    for img_A in imgs_A:
        imgs_A_.append(os.path.splitext(img_A)[0])
    for img_B in imgs_B:
        imgs_B_.append(os.path.splitext(img_B)[0])
    for i in range(len(imgs_A)):
        img_A=imgs_A[i]
        img_inner = get_inner_path(img_A, dir_A)
        if get_inner_path(imgs_A_[i], dir_A) == get_inner_path(imgs_B_[i],dir_B):
            photo_A = cv2.imread(img_A)
            print(img_A)
            photo_B = cv2.imread(imgs_B[i])
            print(imgs_B[i])
            if photo_A.shape == photo_B.shape:
                photo_AB = np.concatenate([photo_A, photo_B], 1)
                img_AB = os.path.join(dir_AB, os.path.splitext(img_inner)[0]+'.png')
                if not os.path.isdir(os.path.split(img_AB)[0]):
                    os.makedirs(os.path.split(img_AB)[0])
                cv2.imwrite(img_AB, photo_AB)
                num += 1
    # for img_A in tqdm(imgs_A):
    #     img_inner=get_inner_path(img_A,dir_A)
    #     if os.path.join(dir_B,img_inner) in imgs_B:
    #         photo_A=cv2.imread(img_A)
    #         photo_B=cv2.imread(os.path.join(dir_B,img_inner))
    #         if photo_A.shape==photo_B.shape:
    #             photo_AB=np.concatenate([photo_A, photo_B], 1)
    #             img_AB=os.path.join(dir_AB,img_inner)
    #             if not os.path.isdir(os.path.split(img_AB)[0]):
    #                 os.makedirs(os.path.split(img_AB)[0])
    #             cv2.imwrite(img_AB, photo_AB)
    #             num+=1
    print("Align data floder created! %d img was processed"%num)


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        assert (os.path.isdir(self.dir_AB) or (os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'))),"dataset path not exsit! [%s]"%self.dir_AB
        if (not os.path.isdir(self.dir_AB)) and os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'):#to do:数据集AB自动合并功能尚不支持英文路径
            os.makedirs(self.dir_AB)
            dir_A=self.dir_AB+'A'
            dir_B=self.dir_AB+'B'
            make_align_img(dir_A,dir_B,self.dir_AB)

        self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        assert (len(self.AB_paths)!=0),"dataset floder is empty! [%s]"%self.dir_AB

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

        #deeplabv3+相关
        self.dir_segs=os.path.join(opt.dataroot, opt.phase+"_seg")
        from src.util.make_seg_img import labelpixels
        flag=0
        if os.path.isdir(self.dir_segs):
            segs = make_dataset(self.dir_segs)
            if len(self.AB_paths) == len(segs):
                flag=1
            else:
                import shutil
                shutil.rmtree(self.dir_segs)
        if flag==0:
            os.makedirs(self.dir_segs)
            for map in self.AB_paths:
                map_np_AB = np.array(Image.open(map).convert("RGB"))
                map_np=map_np_AB[:,map_np_AB.shape[1]//2:,:]
                seg_np = labelpixels(map_np).astype(np.uint8)
                seg_pil = Image.fromarray(seg_np)
                seg_path = os.path.join(self.dir_segs, get_inner_path(map, self.dir_AB))
                seg_pil.save(seg_path)
        self.seg_paths=sorted(make_dataset(self.dir_segs))

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        AB = self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]

        # deeplabv3相关，取label图，之后需将label图与A，B同步处理
        seg_path = self.seg_paths[index]
        seg = Image.open(seg_path)
        seg = np.asarray(seg)
        seg = torch.from_numpy(seg)  # H W
        seg = seg[h_offset:h_offset + self.opt.fineSize,
              w_offset:w_offset + self.opt.fineSize]

        if False: #self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        # self.opt.no_flip=False
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            seg=seg.index_select(1,idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)



        # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
        toseg_transform_list = [transforms.Normalize((0, 0, 0),(2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                transforms.Normalize((-0.5,-0.5,-0.5),(1,1,1)),
                                transforms.Normalize((0.485,0.456,0.406),((0.229,0.224,0.225)))] # 恢复标准化前的数值，并换一组数据标准化
        toseg_transform = transforms.Compose(toseg_transform_list)
        A_seg=toseg_transform(A)
        B_seg=toseg_transform(B)

        return {'A': A, 'B': B, 'seg':seg,'A_seg':A_seg,'B_seg':B_seg,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'


class AlignedDataset_Inter1(BaseDataset):
    def initialize(self, opt, id_layer):
        self.size = opt.loadSize
        self.opt = opt
        self.id_layer = id_layer
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase, "A", str(id_layer))
        self.dir_B = os.path.join(opt.dataroot, opt.phase, "B", str(id_layer+1))
        self.dir_C = os.path.join(opt.dataroot, opt.phase, "C", str(id_layer))
        self.dir_RS = os.path.join(opt.dataroot, opt.phase, "RS", str(id_layer))
        self.dir_Seg = os.path.join(opt.dataroot, opt.phase, "Seg", str(id_layer))


        self.A_paths = sorted(make_dataset(self.dir_A))
        self.C_paths = sorted(make_dataset(self.dir_C))
        if os.path.exists(self.dir_RS):
            self.RS_paths = sorted(make_dataset(self.dir_RS))
            self.Seg_paths = sorted(make_dataset(self.dir_Seg))
        else:
            self.RS_paths = []
            self.Seg_paths = []

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合


    def __getitem__(self, index):

        A_path = self.A_paths[index]
        C_path = self.C_paths[index]
        if self.RS_paths:
            RS_path = self.RS_paths[index]
            RS = Image.open(RS_path).convert('RGB')
            RS = self.transform(RS)

            seg_path = self.Seg_paths[index]
            seg = Image.open(seg_path)

            seg = np.asarray(seg)
            seg = torch.from_numpy(seg)  # H W
            # print(seg.shape)

            # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
            toseg_transform_list = [transforms.Normalize((0, 0, 0), (2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                    transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1)),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                         ((0.229, 0.224, 0.225)))]  # 恢复标准化前的数值，并换一组数据标准化
            toseg_transform = transforms.Compose(toseg_transform_list)
            RS_seg = toseg_transform(RS)  # 送往分割

        A = Image.open(A_path).convert('RGB')
        C = Image.open(C_path).convert('RGB')

        A = self.transform(A)
        C = self.transform(C)



        img_name = os.path.split(A_path)[1]
        img_name_split = img_name.split("_")
        index_x = int(img_name_split[1])
        index_y = int(img_name_split[2].split(".")[0])

        B1_path = os.path.join(self.dir_B,
                               str(self.id_layer + 1) + "_" + str(index_x * 2 - 1) + "_" + str(index_y * 2 - 1)+".png")
        B2_path = os.path.join(self.dir_B,
                               str(self.id_layer + 1) + "_" + str(index_x * 2 - 1) + "_" + str(index_y * 2)+".png")
        B3_path = os.path.join(self.dir_B,
                               str(self.id_layer + 1) + "_" + str(index_x * 2) + "_" + str(index_y * 2 - 1)+".png")
        B4_path = os.path.join(self.dir_B,
                               str(self.id_layer + 1) + "_" + str(index_x * 2) + "_" + str(index_y * 2)+".png")
        B1 = Image.open(B1_path).convert('RGB')
        B2 = Image.open(B2_path).convert('RGB')
        B3 = Image.open(B3_path).convert('RGB')
        B4 = Image.open(B4_path).convert('RGB')

        B = Image.new('RGB', (2 * self.size, 2 * self.size))  # 创建一个新图
        B.paste(B1, (0*self.size, 0*self.size))
        B.paste(B2, (0*self.size, 1*self.size))
        B.paste(B3, (1*self.size, 0*self.size))
        B.paste(B4, (1*self.size, 1*self.size))

        B = B.resize((self.size, self.size), Image.ANTIALIAS)
        B.save(os.path.join(self.root, self.opt.phase, "tmp_B",
                            str(self.id_layer) + "_" + str(index_x) + "_" + str(index_y)) + ".png")

        B = self.transform(B)
        if self.RS_paths:
            return {'A': A, 'B': B, 'C':C, 'A_path': A_path, 'C_path': C_path, 'RS':RS_seg, 'seg':seg}
        else:
            return {'A': A, 'B': B, 'C': C, 'A_path': A_path, 'C_path': C_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset_Inter1'

class AlignedDataset_Inter2(BaseDataset):
    def initialize(self, opt, id_layer):
        self.size = opt.loadSize
        self.opt = opt
        self.id_layer = id_layer
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase, "B", str(id_layer))
        self.dir_B = os.path.join(opt.dataroot, opt.phase, "B1", str(id_layer-1))
        self.dir_C = os.path.join(opt.dataroot, opt.phase, "C", str(id_layer))


        self.A_paths = sorted(make_dataset(self.dir_A))
        self.C_paths = sorted(make_dataset(self.dir_C))

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合


    def __getitem__(self, index):

        A_path = self.A_paths[index]
        C_path = self.C_paths[index]
        A = Image.open(A_path).convert('RGB')
        C = Image.open(C_path).convert('RGB')
        A = self.transform(A)
        C = self.transform(C)

        img_name = os.path.split(A_path)[1]
        img_name_split = img_name.split("_")
        index_x = int(img_name_split[1])
        index_y = int(img_name_split[2].split(".")[0])

        B_all_path = os.path.join(self.dir_B,
                               str(self.id_layer - 1) + "_" + str(math.ceil(index_x/2)) + "_" + str(math.ceil(index_y/2))+".png")
        if index_x%2==0:
            ix = 1
        else:
            ix = 0
        if index_y%2==0:
            iy = 1
        else:
            iy = 0

        B_all = Image.open(B_all_path).convert('RGB')
        B_all = B_all.resize((self.size*2, self.size*2), Image.ANTIALIAS)
        B = B_all.crop((ix*self.size, iy*self.size, ix*self.size+self.size, iy*self.size+self.size))

        # B = B.resize((self.size, self.size), Image.ANTIALIAS)
        B.save(os.path.join(self.root, self.opt.phase, "tmp_B",
                            str(self.id_layer) + "_" + str(index_x) + "_" + str(index_y)) + ".png")

        B = self.transform(B)

        return {'A': A, 'B': B, 'C':C, 'A_path': A_path, 'C_path': C_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset_Inter2'

class AlignedDataset_Inter3(BaseDataset):
    def initialize(self, opt, id_layer):
        self.size = opt.loadSize
        self.opt = opt
        self.id_layer = id_layer
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase, "A", str(id_layer))
        self.dir_B = os.path.join(opt.dataroot, opt.phase, "B", str(id_layer+1))
        self.dir_C = os.path.join(opt.dataroot, opt.phase, "C", str(id_layer))


        self.A_paths = sorted(make_dataset(self.dir_A))
        self.C_paths = sorted(make_dataset(self.dir_C))

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合


    def __getitem__(self, index):

        A_path = self.A_paths[index]
        C_path = self.C_paths[index]
        A = Image.open(A_path).convert('RGB')
        C = Image.open(C_path).convert('RGB')
        A = self.transform(A)
        C = self.transform(C)

        img_name = os.path.split(A_path)[1]
        img_name_split = img_name.split("_")
        index_x = int(img_name_split[1])
        index_y = int(img_name_split[2].split(".")[0])

        B1_path = os.path.join(self.dir_B,
                               str(self.id_layer + 1) + "_" + str(index_x * 2 - 1) + "_" + str(index_y * 2 - 1)+".png")
        B2_path = os.path.join(self.dir_B,
                               str(self.id_layer + 1) + "_" + str(index_x * 2 - 1) + "_" + str(index_y * 2)+".png")
        B3_path = os.path.join(self.dir_B,
                               str(self.id_layer + 1) + "_" + str(index_x * 2) + "_" + str(index_y * 2 - 1)+".png")
        B4_path = os.path.join(self.dir_B,
                               str(self.id_layer + 1) + "_" + str(index_x * 2) + "_" + str(index_y * 2)+".png")
        B1 = Image.open(B1_path).convert('RGB')
        B2 = Image.open(B2_path).convert('RGB')
        B3 = Image.open(B3_path).convert('RGB')
        B4 = Image.open(B4_path).convert('RGB')

        B = Image.new('RGB', (2 * self.size, 2 * self.size))  # 创建一个新图
        B.paste(B1, (0*self.size, 0*self.size))
        B.paste(B2, (0*self.size, 1*self.size))
        B.paste(B3, (1*self.size, 0*self.size))
        B.paste(B4, (1*self.size, 1*self.size))

        B = B.resize((self.size, self.size), Image.ANTIALIAS)
        B.save(os.path.join(self.root, self.opt.phase, "tmp_B",
                            str(self.id_layer) + "_" + str(index_x) + "_" + str(index_y)) + ".png")

        B = self.transform(B)

        return {'A': A, 'B': B, 'C':C, 'A_path': A_path, 'C_path': C_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset_Inter3'