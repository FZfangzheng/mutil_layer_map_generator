import numpy as np
import math
import  random
from  PIL import  Image
import cv2
import os
import shutil
from tqdm import tqdm
from make_seg_img import make_dataset,labelpixels,gray2rgb,get_inner_path
import time
import sys
import json
import gc # 需经常进行垃圾回收

'''该版本k-means对整个数据集进行k-means而非每张图分别进行'''
'''该版本代码会生成较大的缓存，且不会自动清理，建议用完手动清理（为了防止需要人工看一下）'''
'''这个版本用来在服务器上跑'''

cnt=0

def make_floder_from_file_path(file_path):
    if not os.path.isdir(os.path.split(file_path)[0]):
        os.makedirs(os.path.split(file_path)[0])
def default_G(G):
    # ret=[]
    # for i in range(len(G)):
    #     ret.append(list(G[i]))
    ret=list(G)
    return ret
# def hook_G(G):
#     # ret=[]
#     # for i in range(len(G)):
#     #     ret.append(set(G[i]))
#     # return ret
#     ret=set(G)
#     return ret

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

def calculate_zi_fromfile(cache_path,N_file,i):
#给定Gi,里面包含着属于这个类别的元素，然后计算这些元素的中心点
#在本实例中,Gi里面包含的是下标
    with open(os.path.join(cache_path, "allmaplist_%d.json" % 0)) as f:
        X=json.load(f)
    sumi=np.zeros(len(X[0]))
    total_num=0
    for a in range(N_file):
        with open(os.path.join(cache_path, "allmaplist_%d.json" % a)) as f:
            X=json.load(f)
        with open(os.path.join(cache_path, "G_%d.json" % a)) as f:
            G=json.load(f)
            Gi=set(G[i])
        for each in Gi:
            total_num+=1
            sumi+=np.array(X[each])
    sumi/=(total_num+0.000000001)
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
    # print(rst_index,end='\r')
    # assert not (rst_index is None)
    return  rst_index

# def k_mean(X,k,Z0):
#     G=[]
#     Z=Z0 if len(Z0)==k else []
#     N=len(X)
#     c=[]
#     tmpr=set()
#     # 若
#     if Z:
#         for i in range(k):
#             G.append(set())
#     else:
#         while len(Z)<k:
#             r=random.randint(0,len(X)-1)
#             if r not in tmpr:
#                 tmpr.add(r)
#                 Z.append(X[r])
#                 G.append(set())
#     for i in range(N):
#         c.append(-1)
#     #随机生成K个中心元素
#     while True:
#         group_flag=np.zeros(k)
#         for i in range(N):
#             new_ci = find_ci(X[i],Z)
#             if c[i] != new_ci:
#                 if c[i]!=-1:
#                     #找到了更好的,把xi从原来的c[i]调到new_ci去，于是有两个组需要更新：new_ci,c[i]
#                     if i in G[c[i]]:
#                         G[c[i]].remove(i)
#                     group_flag[c[i]]=1  #把i从原来所属的组中移出来
#
#                 G[new_ci].add(i)
#                 group_flag[new_ci]=1    #把i加入到新的所属组去
#
#                 c[i]=new_ci
#
#         #上面已经更新好了各元素的所属
#         if np.sum(group_flag)==0:
#             #没有组被修改
#             break
#         for i in range(k):
#             if group_flag[i]==0:
#                 #未修改,无须重新计算
#                 continue
#             else:
#                 Z[i]=calculate_zi(list(G[i]),X)
#     return Z,c,k

def k_mean_fromfile(cache_path,k,Z0,N_file , is_continue=0):
    if not is_continue:
        G=[]
        Z=Z0 if len(Z0)==k else []
        if (not Z) and len(Z0)<k:
            tmpr = set()
            Z=Z0
            with open(os.path.join(cache_path, "allmaplist_0.json")) as f:
                X=json.load(f)
            while len(Z)<k:
                r=random.randint(0,len(X)-1)
                if (r not in tmpr) and (X[r] not in Z):
                    tmpr.add(r)
                    Z.append(X[r])
        # N=len(X)
        c=[]
        tmpr=set()
        # 初始化
        for a in range(N_file):
            G = []
            c = []
            with open(os.path.join(cache_path, "allmaplist_%d.json" % a)) as f:
                X=json.load(f)
            for i in range(k):
                G.append(set())
            for i in range(len(X)):
                c.append(-1)
            with open(os.path.join(cache_path, "G_%d.json" % a), 'w') as f:
                json.dump(G, f,default=default_G)
            with open(os.path.join(cache_path, "c_%d.json" % a), 'w') as f:
                json.dump(c, f)
    else:
        print('compute Z from cache')
        Z=[]
        for i in tqdm(range(k)):
            Z.append(calculate_zi_fromfile(cache_path,N_file,i))


    num=0
    while True:
        num+=1
        print("第%d次循环开始"%num)
        gc.collect()
        group_flag = np.zeros(k)
        for a in tqdm(range(N_file)):
            with open(os.path.join(cache_path, "G_%d.json" % a)) as f:
                G=json.load(f)
                for i in range(len(G)):
                    G[i]=set(G[i])
            with open(os.path.join(cache_path, "c_%d.json" % a)) as f:
                c=json.load(f)
            with open(os.path.join(cache_path, "allmaplist_%d.json" % a)) as f:
                X=json.load(f)
            for i in range(len(X)):
                new_ci = find_ci(np.array(X[i]), Z)
                if c[i] != new_ci:
                    if c[i] != -1:
                        # 找到了更好的,把xi从原来的c[i]调到new_ci去，于是有两个组需要更新：new_ci,c[i]
                        if i in G[c[i]]:
                            G[c[i]].remove(i)
                        group_flag[c[i]] = 1  # 把i从原来所属的组中移出来
                    G[new_ci].add(i)
                    group_flag[new_ci] = 1  # 把i加入到新的所属组去
                    c[i] = new_ci
            with open(os.path.join(cache_path, "G_%d.json" % a), 'w') as f:
                json.dump(G, f,default=default_G)
            with open(os.path.join(cache_path, "c_%d.json" % a), 'w') as f:
                json.dump(c, f)
        if np.sum(group_flag)==0:
            #没有组被修改
            break
        for i in range(k):
            if group_flag[i]==0:
                #未修改,无须重新计算
                continue
            else:
                # # 此处占用内存多，尝试提前释放内存
                # del G
                # del c
                # del X
                gc.collect()
                Z[i]=calculate_zi_fromfile(cache_path,N_file,i)
    return Z, k


    #     for a in range(N_file):
    #         G = []
    #         c = []
    #         X = json.load(os.path.join(cache_path, "allmaplist_%d.json" % a))
    #         for i in range(k):
    #             G.append(set())
    #         for i in range(len(X)):
    #             c.append(-1)
    #         json.dump(G, os.path.join(cache_path, "G_%d.json" % a))
    #         json.dump(c, os.path.join(cache_path, "C_%d.json" % a))
    #
    # for i in range(k):
    #     G.append(set())
    #
    # for i in range(N):
    #     c.append(-1)
    # #随机生成K个中心元素
    # while True:
    #     group_flag=np.zeros(k)
    #     for i in range(N):
    #         new_ci = find_ci(X[i],Z)
    #         if c[i] != new_ci:
    #             if c[i]!=-1:
    #                 #找到了更好的,把xi从原来的c[i]调到new_ci去，于是有两个组需要更新：new_ci,c[i]
    #                 if i in G[c[i]]:
    #                     G[c[i]].remove(i)
    #                 group_flag[c[i]]=1  #把i从原来所属的组中移出来
    #
    #             G[new_ci].add(i)
    #             group_flag[new_ci]=1    #把i加入到新的所属组去
    #
    #             c[i]=new_ci
    #
    #     #上面已经更新好了各元素的所属
    #     if np.sum(group_flag)==0:
    #         #没有组被修改
    #         break
    #     for i in range(k):
    #         if group_flag[i]==0:
    #             #未修改,无须重新计算
    #             continue
    #         else:
    #             Z[i]=calculate_zi(list(G[i]),X)
    # return Z,c,k


# 在哪个维度进行整合？暂定第 0 维好了   [w, h, c]
def concat_map(allmap,mappath): # maps 的第1,2维须与amap保持一致，或为None;
    im = Image.open(mappath)
    im = im.convert("RGB")
    im_np = np.array(im)
    im.close()  # 这句要放到赋值后面，具体的原理还没太弄清楚
    if isinstance(allmap,np.ndarray):
        allmap=np.concatenate((allmap,im_np),axis=0)
    else:
        allmap=im_np
    return allmap

# 需要储存几种图像？因为没有pretreat过程，因此就不生成老版结果，直接生成k-means的rgb和gray图像
# 代码结构怎么设计？（不炸内存的理想状况下）遍历整个文件夹，将所有图像合并为一个ndarray→对其进行k-means→按原来的顺序切割存储图片
# 为解决内存问题和考虑扩展性，需传入tmp路径，所有大数据都在tmp里存为json格式，暂时以1000为单位
def make_seg_img_from_maps(maps_path,segs_new_path,cache_path,k_means_k=8,one_time_num=500,stand_rgb = [[233, 228, 222], [243, 240, 233], [251, 253, 252], [252, 157, 39], [252, 223, 147],[203,223,172],[177,208,254],[200,199,197]],is_continue=0):
    if is_continue:
        assert  os.path.isdir(cache_path)

    if not os.path.isdir(cache_path):
        os.makedirs(cache_path)
    maps=make_dataset(maps_path)
    if not os.path.isdir(segs_new_path):
        os.makedirs(segs_new_path)
        os.makedirs(os.path.join(segs_new_path, 'rgb'))
        os.makedirs(os.path.join(segs_new_path, 'seg'))
    N_map=len(maps)
    N_file=(N_map+one_time_num-1)//one_time_num # 向上取整
    print("切割为%d个块进行计算"%N_file)
    allmap=None

    if not is_continue:
        for a in tqdm(range(N_file)):
            allmap=None
            for j in range(one_time_num):
                if a*one_time_num+j>=N_map:
                    break
                allmap = concat_map(allmap, maps[a*one_time_num+j])
            h, w, chanl = allmap.shape
            allmap = allmap.reshape((h * w, chanl))
            print('allmap的内存占用' + str(sys.getsizeof(allmap)))
            allmaplist = []
            for k in range(h * w):
                allmaplist.append(allmap[k])
                # print(k, end='\r')
                # if (k % 100000 == 0):
                #     print(str(k) + '此时allmaplist的内存占用' + str(sys.getsizeof(allmaplist)))
            print(str(k) + '最终allmaplist的内存占用' + str(sys.getsizeof(allmaplist)))
            with open(os.path.join(cache_path,"allmaplist_%d.json"%a),'w') as f:
                for i in range(len(allmaplist)):
                    tmp=[]
                    for j in range(len(allmaplist[i])):
                        tmp.append(int(allmaplist[i][j]))
                    allmaplist[i]=tmp
                json.dump(allmaplist,f)
        del allmap
        del allmaplist
        gc.collect()

    print('start k-means')
    t1 = time.time()
    Z,  k = k_mean_fromfile(cache_path, k_means_k, stand_rgb,N_file=N_file,is_continue=is_continue)
    t2 = time.time()
    print('k-means 花费' + str(t2 - t1) + '秒')
    with open(os.path.join(cache_path,'Z.json'), 'w') as f:
        for i in range(len(Z)):
            tmp = []
            for j in range(len(Z[i])):
                tmp.append(int(Z[i][j]))
            Z[i] = tmp
        json.dump(Z,f)
    # 从cache文件夹中恢复图像
    for a in range(N_file):
        with open(os.path.join(cache_path, "c_%d.json" % a)) as f:
            c=json.load(f)
        rets_new = np.array(c)
        w = 256
        h = len(rets_new) // 256
        rets_new = rets_new.reshape((h, w))
        rets_new = rets_new.astype(np.uint8)
        # label 合并
        # rets_new[(rets_new == 3)] = 0
        # rets_new[(rets_new == 4)] = 1
        h1 = 256
        for i in range(one_time_num):
            if a*one_time_num+i>=N_map:
                break
            ret_new = rets_new[h1 * i:h1 * (i + 1), :]
            new_im = gray2rgb(ret_new,n_class=k_means_k, label_list=Z)
            new_im = Image.fromarray(new_im)
            new_path_im = os.path.join(segs_new_path, 'rgb',
                                       os.path.splitext(get_inner_path(maps[a*one_time_num+i], maps_path))[0] + ".png")
            make_floder_from_file_path(new_path_im)
            new_im.save(new_path_im)
            new_seg = Image.fromarray(ret_new)
            new_path_seg = os.path.join(segs_new_path, 'seg',
                                        os.path.splitext(get_inner_path(maps[a*one_time_num+i], maps_path))[0] + '.png')
            make_floder_from_file_path(new_path_seg)
            new_seg.save(new_path_seg)
    pass


if __name__ == '__main__':
    # filename = r"D:\map_translate\数据集\TW16分析\rs\tile-16-14003-27446.jpg"
    # test_rgb_img(filename)
    # print(cnt)
    if len(sys.argv) > 1:
        maps_path = sys.argv[1]
        segs_new_path = sys.argv[2]
        cache_path =  sys.argv[3]
        k=int(sys.argv[4])
        one_time_num=int(sys.argv[5])
        is_continue=int(sys.argv[6])
    else:
        maps_path = r"D:\datasets\maps\all_B_resize256"
        segs_new_path = r"D:\datasets\maps\all_seg_8label"
        cache_path = r"D:\map_translate\看看效果\0601TW16_1708图_分割结果做p2p输入，epoch200，重新训练\k_means_cache"
        k = 8
        one_time_num = 100
        is_continue=0
    print('is_continue : %d'%is_continue)
    make_seg_img_from_maps(maps_path,segs_new_path,cache_path,k_means_k=k,one_time_num=one_time_num,is_continue=is_continue)
