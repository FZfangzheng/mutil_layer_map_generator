import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import math

def get_neighbor(fullmap, x, y, l): # 上下左右各扩张l，实际长宽为2*l+1
    h,w=fullmap.shape
    left=max(x-l,0)
    right=min(x+l,w-1)
    top=max(y-l,0)
    bottom=min(y+l,h-1)
    x_bias=left # 小矩形和大矩形的坐标变换
    y_bias=top
    return fullmap[top:bottom + 1, left:right + 1], x_bias, y_bias

thetas = np.deg2rad(np.arange(-90.0, 90.0, 1))
cos_t = np.cos(thetas)
sin_t = np.sin(thetas)
def seg2theta(img,x_center,y_center, angle_step=1):
    # 从文件hough_transform复制过来并进行删改，用于计算某小邻域块的霍夫变换
    # Rho and Theta ranges
    global thetas
    width, height = img.shape
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)  # 在间隔内生成N个数字

    # Cache some resuable values
    global cos_t
    global sin_t
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
    road_mask = img
    y_idxs, x_idxs = np.nonzero(road_mask)  # 返回数组a中非零元素的索引值数组。

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))  # diag_len+是因为起始点其实表示负diag_len
            accumulator[rho, t_idx] += 1
    x = x_center
    y = y_center
    max=0
    this_theta=None
    for t_idx in range(num_thetas):
        # Calculate rho. diag_len is added for a positive index
        rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))  # diag_len+是因为起始点其实表示负diag_len
        if accumulator[rho, t_idx]>max:
            max = accumulator[rho, t_idx]
            this_theta=thetas[t_idx]
    return this_theta

def line2cross(road_line):# ndarray bool型
    road_line_cross=np.zeros(road_line.shape,dtype=np.bool)
    points=[[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]] # y,x的偏置值
    build=[[1,3,7],[1,3,8],[1,3,9],[1,6,7],[1,6,8],[1,7,9],[2,4,6],[2,4,8],[2,4,9],[2,6,7],[2,6,8],[2,7,9],[3,4,8],[3,4,9],[3,7,9],[4,6,8]]
    end=[[2,4],[2],[2,6],[4],[],[4,8],[],[],[],[],[],[8],[],[6],[6,8],[]]
    for i in build:
        for j in range(len(i)):
            i[j]=i[j]-1
    for i in end:
        for j in range(len(i)):
            i[j]=i[j]-1
    y_idx, x_idx = np.nonzero(road_line)
    for i in range(len(x_idx)):
        x = x_idx[i]
        y = y_idx[i]
        if x==0 or y==0 or x==road_line.shape[1]-1 or y==road_line.shape[0]-1 :
            continue
        iscross=None
        for j in range(len(build)):
            iscross=1
            for k in build[j]:
                if not road_line[y+points[k][0],x+points[k][1]]:
                    iscross = 0
            if iscross==0:
                continue
            else:
                for k in end[j]:
                    if road_line[y+points[k][0],x+points[k][1]]:
                        iscross = 0
                break
        road_line_cross[y,x]=iscross
    return road_line_cross
    pass


def line2theta(road_line): # ndarray bool型
    road_line_theta=np.zeros(road_line.shape,dtype=np.float)
    theta_cross=-2 # 正常theta取值为-π/2 ~ π/2
    road_line_cross=line2cross(road_line)
    y_idx,x_idx=np.nonzero(road_line)
    for i in range(len(x_idx)):
        x=x_idx[i]
        y=y_idx[i]
        if road_line_cross[y,x]:
            road_line_theta[y, x] = theta_cross
            continue
        neighbor,x_bias, y_bias=get_neighbor(road_line,x,y,5)
        x_center=x-x_bias
        y_center=y-y_bias
        theta=seg2theta(neighbor,x_center,y_center)
        road_line_theta[y,x]=theta
    return road_line_theta,road_line_cross
    pass

def line2road(road_mask,line_mask,cross_mask,line_message):
    road_message=np.zeros(line_message.shape,dtype=line_message.dtype)
    y_idx, x_idx = np.nonzero(road_mask)
    for i in range(len(x_idx)):
        x=x_idx[i]
        y=y_idx[i]
        if line_mask[y,x]:
            road_message[y,x]=line_message[y,x]
            continue
        else:
            l=2 # 从2x2开始依次扩大范围
            while l<50: #使用一个很大的值作为上限
                neighbor,x_bias,y_bias=get_neighbor(line_mask,x,y,l)
                x0=x-x_bias
                y0=y-y_bias
                y_idx_tmp,x_idx_tmp=np.nonzero(neighbor)
                min=l #仅有小于等于L的距离可以触发，以保证确实能取到最小值（因为neighbor是方的而不是圆的可能导致的误差）
                finded=0
                ret=None
                for j in range(len(x_idx_tmp)):
                    x1=x_idx_tmp[j]
                    y1=y_idx_tmp[j]
                    distance=math.sqrt((x1-x0)**2+(y1-y0)**2)
                    if distance<=min:
                        min=distance
                        finded=1
                        ret=line_message[y1+y_bias,x1+x_bias]
                if finded:
                    road_message[y, x] =ret
                    break
                else:
                    l=l+2
    # 最后单独处理一下交点，逻辑为（line交点附近某个圆形均为交点，半径按照能保证圆形内均为road的最长半径选取）
    y_idx, x_idx = np.nonzero(cross_mask)
    tmp_mask=np.ones(road_mask.shape,dtype=np.bool) # 做一个全1的遮罩
    for i in range(len(x_idx)):
        l_final = 0
        x=x_idx[i]
        y=y_idx[i]
        for l in range(1,100):
            neighbor,x_bias,y_bias=get_neighbor(tmp_mask,x,y,l)
            x0=x-x_bias
            y0=y-y_bias
            y_idx_tmp, x_idx_tmp = np.nonzero(neighbor)
            flag=1
            for j in range(len(x_idx_tmp)):
                x1=x_idx_tmp[j]
                y1=y_idx_tmp[j]
                if math.sqrt((x1-x0)**2+(y1-y0)**2)<=l:
                    if not road_mask[y1+y_bias,x1+x_bias]:
                        flag=0
                        break
            if flag==0:
                l_final=l-1
                break
        neighbor, x_bias, y_bias = get_neighbor(road_mask, x, y, l_final)
        x0 = x - x_bias
        y0 = y - y_bias
        y_idx_tmp, x_idx_tmp = np.nonzero(neighbor)
        for j in range(len(x_idx_tmp)):
            x1 = x_idx_tmp[j]
            y1 = y_idx_tmp[j]
            if math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) <= l_final:
                road_message[y1+y_bias,x1+x_bias]=line_message[y,x]
    return road_message

def x_y_ok(x,y,imgnp):
    h,w=imgnp.shape[0],imgnp.shape[1]
    return x>=0 and x<w and y>=0 and y<h

def median_filter(array,x,y,l=1):
    # 进行一个点的滤波操作
    assert len(array.shape)==2
    neighbor,x_bias,y_bias=get_neighbor(array,x,y,l) # 交点与非道路点的宽度均为0
    if (neighbor!=0).sum()==0:
        ave=0
    else:
        ave=neighbor.sum()//(neighbor!=0).sum()
    return ave

def theta2wid(road_mask,road_theta):
    y_idxs, x_idxs = np.nonzero(road_mask)  # 返回数组a中非零元素的索引值数组。
    global thetas
    global cos_t
    global sin_t
    thetas_dict = dict(zip(thetas, range(len(thetas))))
    road_wid = np.zeros(road_theta.shape, dtype=np.uint8)
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        if road_theta[y,x]==-2.0:
            continue
        this_t_idx =thetas_dict[road_theta[y,x]]
        this_roadwid_l = 0
        this_roadwid_r = 0
        for j in range(1, 400):  # 设置一个最大值为400，实际应永远达不到:
            # theta 是垂线角度，可以直接用
            x_ = x + int(cos_t[this_t_idx] * j)
            y_ = y + int(sin_t[this_t_idx] * j)
            if x_y_ok(x_, y_, road_mask) and road_mask[y_, x_] > 0:
                this_roadwid_l = j
            else:
                break
        for j in range(1, 400):  # 设置一个最大值为400，实际应永远达不到:
            # theta 翻转180度，cos 和 sin 都变成相反数
            x_ = x - int(cos_t[this_t_idx] * j)
            y_ = y - int(sin_t[this_t_idx] * j)
            if x_y_ok(x_, y_, road_mask) and road_mask[y_, x_] > 0:
                this_roadwid_r = j
            else:
                break
            pass
        this_roadwid = 1 + this_roadwid_l + this_roadwid_r
        road_wid[y,x]=this_roadwid
    # 此时交点宽度为0，做一个均值滤波
    road_wid[road_wid>20]=20 # 20 是一个足够大的值,这样的点还是挺多的
    # print((road_wid>20).sum())
    road_wid_=np.zeros(road_wid.shape,dtype=road_wid.dtype)
    zero_points=[]
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        road_wid_[y,x]=median_filter(road_wid,x,y,2)
        if road_wid_[y,x]==0:
            zero_points.append([x,y])
    road_wid=road_wid_
    while len(zero_points)>0:
        road_wid_ = np.zeros(road_wid.shape, dtype=road_wid.dtype)
        zero_points_ = []
        for i in range(len(zero_points)):
            x = zero_points[i][0]
            y = zero_points[i][1]
            road_wid_[y, x] = median_filter(road_wid, x, y, 2)
            if road_wid_[y, x] == 0:
                zero_points_.append([x, y])
        road_wid[road_wid_!=0] = road_wid_[road_wid_!=0]
        zero_points=zero_points_
    return road_wid