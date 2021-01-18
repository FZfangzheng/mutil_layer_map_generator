import numpy as np
import math
import  random
from  PIL import  Image
import cv2
import os
import shutil
from tqdm import tqdm
from make_seg_img import make_dataset,labelpixels,gray2rgb,get_inner_path

stand_rgb=[[239,238,236],[255,242,175],[170,218,255],[208,236,208],[255,255,255]]
def pretreat(img3D, label_list=[(15,23),(24,26),(101,105),(58,62),(0,0)]):  # 接受参数为ndarray
    label_exist=np.zeros(len(label_list)) # 标记哪几个label是存在的
    Z0=[]
    img3D = cv2.cvtColor(img3D, cv2.COLOR_RGB2HSV)
    ret2D=np.zeros(img3D.shape[:-1])
    masks=[]
    for tmp in label_list:
        masks.append(cv2.inRange(img3D, np.array([tmp[0], 0, 0]), np.array([tmp[1], 255, 255])))
    for i,mask in enumerate(masks):
        mask=(mask==255)
        ret2D+=mask*(i+1)
        if mask.any():
            label_exist[i]=1
            Z0.append(stand_rgb[i])
    mask=(ret2D==0)
    ret2D+=mask*256
    ret2D=ret2D-1
    ret2D=ret2D.astype(np.uint8)
    return ret2D,len(Z0),Z0,label_exist

cnt=0
def calculate_zi(Gi,X):
#给定Gi,里面包含着属于这个类别的元素，然后计算这些元素的中心点
#在本实例中,Gi里面包含的是下标
    global  cnt
    sumi=np.zeros(len(X[0]))
    for each in Gi:
        cnt+=1
        sumi+=X[each]
    sumi/=(len(Gi)+0.000000001)
    zi=sumi
    return zi

def find_ci(xi,Z):
    #寻找离xi最近的中心元素ci,使得Z[ci]与xi之间的向量差的內积最小
    global  cnt
    dis_= np.inf
    len_=len(Z)
    rst_index = None
    for i in range(len_):
        cnt+=1
        tmp_dist=np.dot(xi-Z[i],np.transpose(xi-Z[i]))
        if tmp_dist<dis_:
            rst_index=i
            dis_=tmp_dist
    return  rst_index

def k_mean(X,k,Z0):
    G=[]
    Z=Z0 if len(Z0)==k else []
    N=len(X)
    c=[]
    tmpr=set()
    # 若
    if Z:
        for i in range(k):
            G.append(set())
    else:
        while len(Z)<k:
            r=random.randint(0,len(X)-1)
            if r not in tmpr:
                tmpr.add(r)
                Z.append(X[r])
                G.append(set())
    for i in range(N):
        c.append(-1)
    #随机生成K个中心元素
    while True:
        group_flag=np.zeros(k)
        for i in range(N):
            new_ci = find_ci(X[i],Z)
            if c[i] != new_ci:
                if c[i]!=-1:
                    #找到了更好的,把xi从原来的c[i]调到new_ci去，于是有两个组需要更新：new_ci,c[i]
                    if i in G[c[i]]:
                        G[c[i]].remove(i)
                    group_flag[c[i]]=1  #把i从原来所属的组中移出来

                G[new_ci].add(i)
                group_flag[new_ci]=1    #把i加入到新的所属组去

                c[i]=new_ci

        #上面已经更新好了各元素的所属
        if np.sum(group_flag)==0:
            #没有组被修改
            break
        for i in range(k):
            if group_flag[i]==0:
                #未修改,无须重新计算
                continue
            else:
                Z[i]=calculate_zi(list(G[i]),X)
    return Z,c,k


def test_rgb_img(filename):
    im = Image.open(filename)
    im=im.convert("RGB")
    img = im.load() # 返回一个用于读取和修改像素的像素访问对象
    #预处理图像以确定k
    im_np=np.array(im)
    im.close() #这句要放到赋值后面，具体的原理还没太弄清楚
    ret_old,k,Z0,label_exist=pretreat(im_np)

    height = im.size[0]
    width= im.size[1]
    X=[]
    for i in range(0,height):
        for j in range(0,width):
            X.append(np.array(img[i,j]))
    Z,c,k=k_mean(X,k,Z0)
    c=np.array(c) # 为了能使用数组判断
    j=k-1
    for i in range(len(label_exist)-1,-1,-1):
        if label_exist[i]==1:
            c[c==j]=i
            j=j-1
    #print(Z)
    # new_im = Image.new("RGB",(height,width))
    # for i in range(0,height):
    #     for j in range(0,width):
    #         index = i * width + j
    #         pix = list(Z[c[index]])
    #         for k in range(len(pix)):
    #             pix[k]=int(pix[k])
    #         new_im.putpixel((i,j),tuple(pix))
    # new_im.show()

    ret_new=np.array(c)
    ret_new=ret_new.reshape((height,width)).transpose(1,0)
    ret_new=ret_new.astype(np.uint8)
    return ret_old,ret_new

def make_seg_img_from_maps(maps_path,segs_old_path,segs_new_path):
    maps=make_dataset(maps_path)
    if not os.path.isdir(segs_old_path):
        os.makedirs(segs_old_path)
        os.makedirs(os.path.join(segs_old_path,'rgb'))
        os.makedirs(os.path.join(segs_old_path, 'seg'))
    if not os.path.isdir(segs_new_path):
        os.makedirs(segs_new_path)
        os.makedirs(os.path.join(segs_new_path, 'rgb'))
        os.makedirs(os.path.join(segs_new_path, 'seg'))

    for map in tqdm(maps):
        ret_old,ret_new=test_rgb_img(map)

        #此处执行label合并，将3（绿地）视为0（背景），将4（道路）视为1（主干道）。5分类变为3分类（背景、路、水）
        ret_old[(ret_old==3)]=0
        ret_old[(ret_old == 4)] = 1
        ret_new[(ret_new==3)]=0
        ret_new[(ret_new == 4)] = 1

        print(ret_old.max())
        print(ret_new.max())
        old_im=gray2rgb(ret_old)
        old_im=Image.fromarray(old_im)
        old_path_im=os.path.join(segs_old_path,'rgb',os.path.splitext(get_inner_path(map,maps_path))[0]+'.png')
        old_im.save(old_path_im)
        old_seg=Image.fromarray(ret_old)
        old_path_seg = os.path.join(segs_old_path, 'seg', os.path.splitext(get_inner_path(map, maps_path))[0]+'.png')
        old_seg.save(old_path_seg)

        new_im = gray2rgb(ret_new)
        new_im = Image.fromarray(new_im)
        new_path_im = os.path.join(segs_new_path,'rgb', os.path.splitext(get_inner_path(map, maps_path))[0]+".png")
        new_im.save(new_path_im)
        new_seg = Image.fromarray(ret_new)
        new_path_seg = os.path.join(segs_new_path, 'seg', os.path.splitext(get_inner_path(map, maps_path))[0]+'.png')
        new_seg.save(new_path_seg)
        # new_path=os.path.join(segs_new_path,get_inner_path(map,maps_path))
        # new_im.save(new_path)

    pass

if __name__ == '__main__':
    # filename = r"D:\map_translate\数据集\TW16分析\rs\tile-16-14003-27446.jpg"
    # test_rgb_img(filename)
    # print(cnt)
    maps_path = r"D:\map_translate\数据集\TW16分析\map"
    segs_old_path = r"D:\map_translate\数据集\TW16分析\label合并\old"
    segs_new_path = r"D:\map_translate\数据集\TW16分析\label合并\new"
    make_seg_img_from_maps(maps_path,segs_old_path,segs_new_path)
