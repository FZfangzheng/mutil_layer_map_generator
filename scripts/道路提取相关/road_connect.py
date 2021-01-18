import numpy as np
from hough_transform import x_y_ok

def get_neighbor(seg_isend,x,y,l_this):
    h,w=seg_isend.shape
    left=max(x-l_this,0)
    right=min(x+l_this,w)
    top=max(y-l_this,0)
    bottom=min(y+l_this,h)
    return seg_isend[top:bottom,left:right]

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

def road_connect(seg_vis,are_edges,seg_theta,seg_rho,seg_roadwid):
    # 张量方法太复杂，放弃计算中心线的步骤，直接进行检测与连接操作
    r=np.deg2rad(15) # 角度阈值-正负
    l=50 # 怎么做这个自适应呢 思路1 越粗的道路距离越大， 思路2 附近道路越稀疏越大 先选择思路1（思路2不知道咋实现）
    l_stand_wid=5 # 达到50的最大宽度
    wid=3 # 允许的两个道路的宽度差距
    road_rgb=np.array([255,242,175])
    y_idxs, x_idxs = np.nonzero(are_edges)
    width, height = are_edges.shape
    angle_step = 1
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    thetas_dict = dict(zip(thetas, range(len(thetas))))
    seg_isend = np.zeros((width, height), dtype=np.uint8)
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        # is_end=0
        t_idx=thetas_dict[seg_theta[y, x]]
        is_end_l=1
        for j in range(1,5): # 保证取到不同的点
            x_=x+int(sin_t[t_idx]*j) # 沿着道路方向移动
            y_=y-int(cos_t[t_idx]*j)
            if x_==x and y_==y:
                continue
            if (not x_y_ok(x_,y_,are_edges)) or are_edges[y_,x_]==1:
                is_end_l=0 # 这是theta顺时针九十度的方向
        is_end_r = 2
        for j in range(1,5): # 保证取到不同的点
            x_=x-int(sin_t[t_idx]*j) # 沿着道路方向移动
            y_=y+int(cos_t[t_idx]*j)
            if x_==x and y_==y:
                continue
            if (not x_y_ok(x_,y_,are_edges)) or are_edges[y_,x_]==1:
                is_end_r=0 # 使用不同的标记区分两个方向 # 这是theta逆时针九十度的方向
        seg_isend[y,x]=max(is_end_l,is_end_r)
    y_idxs_end, x_idxs_end = np.nonzero(seg_isend)
    for i in range(len(x_idxs_end)):
        x = x_idxs_end[i]
        y = y_idxs_end[i]
        l_this=min(int(l*seg_roadwid[y,x]/l_stand_wid),l)
        neighbor=get_neighbor(seg_isend,x,y,l_this)
        y_idxs_tmp, x_idxs_tmp = np.nonzero(neighbor)
        for j in range(len(x_idxs_tmp)):
            x_=x_idxs_tmp[j]
            y_ = y_idxs_tmp[j]
            theta_tmp=get_rad(x,y,x_,y_)
            theta_road=seg_theta[y,x]+np.deg2rad(90)
            if seg_isend[y,x]==1:
                theta_road = theta_road-np.deg2rad(180)
            if abs(theta_road-theta_tmp)<r and distance(x,y,x_,y_)<l_this \
                    and abs(seg_theta[y,x]-seg_theta[y_,x_])<r \
                    and abs(seg_roadwid[y,x]-seg_roadwid[y_,x_])<wid:
                len_dots=int(distance(x,y,x_,y_))
                for k in range(1,len_dots):
                    xtmp=int((x_-x)*k/len_dots)
                    ytmp=int((y_-y)*k/len_dots)
                    seg_vis[ytmp,xtmp]=road_rgb
    return seg_vis,seg_isend
