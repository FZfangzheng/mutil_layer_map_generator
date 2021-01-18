import numpy as np
from hough_transform import x_y_ok
from skimage import morphology

# 尝试迭代腐蚀的方式将道路细化

def get_neighbor(seg_isend,x,y,l_this):
    h,w=seg_isend.shape
    left=max(x-l_this,0)
    right=min(x+l_this,w)
    top=max(y-l_this,0)
    bottom=min(y+l_this,h)
    x_bias=left # 小矩形和大矩形的坐标变换
    y_bias=top
    return seg_isend[top:bottom,left:right],x_bias,y_bias

def get_rad(x0,y0,x1,y1): # 点1在点0的什么方向，以右为零度，逆时针为正
    a=np.array([x1-x0,y1-y0])
    b=np.array([1,0])
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    a_dot_b = a.dot(b)
    theta = np.arccos(a_dot_b / (a_norm * b_norm))
    if(y1>y0):
        theta*=-1
    return theta

def distance(x0,y0,x1,y1):
    return np.sqrt((x1-x0)**2+(y1-y0)**2)

# def label_point(x,y,road_mask):
#     # 对某个点进行标记：传入的mask为0-1二值图像，0表示背景，1表示前景（道路）
#     # 1.首先判断是否为骨架像素：

# def refine(road_mask):
#     # 将一个道路掩膜图像细化
#     road_state=np.array(road_mask).astype(np.uint8)
#     for y in road_state.shape[0]:
#         for x in road_state.shape[1]:

def endpoint(nbr): # 3x3 np.bool
    assert nbr[1,1]
    if nbr.sum()<=2:
        return True
    else:
        flag=0
        tmp=[[0,0],[0,1],[0,2],[1,2],[2,2],[2,1],[2,0],[1,0],[0,0]] # 左上角出发，顺时针一个循环
        for i in range(len(tmp)-1):
            if nbr[tmp[i][0],tmp[i][1]]==False and nbr[tmp[i+1][0],tmp[i+1][1]]==True: # 所谓0→1模式
                flag+=1
        if flag==1:
            return True
        else:
            return False


def find_endpoint(road_line):
    road_endpoints=np.zeros(road_line.shape,dtype=np.bool)
    y_idxs, x_idxs = np.nonzero(road_line)
    for i in range(len(x_idxs)):
        y=y_idxs[i]
        x=x_idxs[i]
        if x==0 or y==0 or x==road_line.shape[1]-1 or y==road_line.shape[0]-1:
            continue
        nbr=road_line[y-1:y+2,x-1:x+2]
        if endpoint(nbr):
            road_endpoints[y,x]=True
    return road_endpoints

def road_connect(seg_vis, road_mask, seg_theta, seg_rho, seg_roadwid):
    road_line=morphology.skeletonize(road_mask)
    road_line_old=np.array(road_line)
    road_endpoints=find_endpoint(road_line)
    # 在中心线的端点进行搜索与连接操作
    r=np.deg2rad(15) # 角度阈值-正负
    lmin=5
    l=50 # 怎么做这个自适应呢 思路1 越粗的道路距离越大， 思路2 附近道路越稀疏越大 先选择思路1（思路2不知道咋实现）
    l_stand_wid=10 # 达到50的最大宽度
    wid=3 # 允许的两个道路的宽度差距
    road_rgb=np.array([255,255,255])
    width, height = road_mask.shape
    angle_step = 1
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    thetas_dict = dict(zip(thetas, range(len(thetas))))
    y_idxs, x_idxs = np.nonzero(road_endpoints)
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        # neighbor=get_neighbor(road_endpoints,x,y,lmin)
        # y_idxs_tmp, x_idxs_tmp = np.nonzero(neighbor)
        # for j in range(len(x_idxs_tmp)):
        #     x_ = x_idxs_tmp[j]
        #     y_ = y_idxs_tmp[j]
        #     len_dots = int(distance(x, y, x_, y_))
        #     for k in range(1, len_dots):
        #         xtmp = int((x_ - x) * k / len_dots)
        #         ytmp = int((y_ - y) * k / len_dots)
        #         road_line[ytmp, xtmp] = True
        # if len(x_idxs_tmp)<=1: # 邻域内仅有该点自己，没有其他合适的点


        l_this=min(int(l*seg_roadwid[y,x]/l_stand_wid),l)
        neighbor,x_bias,y_bias=get_neighbor(road_endpoints,x,y,l_this)
        y_idxs_tmp, x_idxs_tmp = np.nonzero(neighbor)
        for j in range(len(x_idxs_tmp)):
            x_=x_idxs_tmp[j]+x_bias
            y_ = y_idxs_tmp[j]+y_bias
            theta_tmp=get_rad(x,y,x_,y_)
            theta_road=seg_theta[y,x]+np.deg2rad(90)
            # if road_endpoints[y,x]==1:
            #     theta_road = theta_road-np.deg2rad(180)
            if (abs(theta_road-theta_tmp)<r or abs(theta_road-np.deg2rad(180)-theta_tmp)<r) and distance(x,y,x_,y_)<l_this \
                    and abs(seg_theta[y,x]-seg_theta[y_,x_])<r \
                    and abs(seg_roadwid[y,x]-seg_roadwid[y_,x_])<wid:
                len_dots=int(distance(x,y,x_,y_))
                for k in range(1,len_dots):
                    xtmp=int((x_-x)*k/len_dots)+x
                    ytmp=int((y_-y)*k/len_dots)+y
                    # seg_vis[ytmp,xtmp]=road_rgb
                    road_line[ytmp,xtmp]=True
    return road_mask,road_line_old,road_endpoints,road_line


    # return seg_vis,seg_isend
