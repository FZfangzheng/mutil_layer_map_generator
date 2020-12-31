import numpy as np
import imageio
import math
import os
from util import gray2rgb

iii=0
pi=3.141592653589793 #差不多够用了
def x_y_ok(x,y,imgnp):
    h,w=imgnp.shape[0],imgnp.shape[1]
    return x>=0 and x<w and y>=0 and y<h
def one_seg2message(img,save_path, angle_step=1, lines_are_white=True, value_threshold=5):

    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    width, height = img.shape
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)  # 在间隔内生成N个数字

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
    # (row, col) indexes to edges
    road_mask = img > value_threshold if lines_are_white else img < value_threshold  # 阈值
    y_idxs, x_idxs = np.nonzero(road_mask)  # 返回数组a中非零元素的索引值数组。

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))  # diag_len+是因为起始点其实表示负diag_len
            accumulator[rho, t_idx] += 1

    # 需要一个ρ矩阵，一个θ矩阵，配合mask矩阵食用
    seg_rho=np.zeros((width,height),dtype=np.int)
    seg_theta=np.zeros((width,height),dtype=np.float)
    seg_roadwid=np.zeros((width,height),dtype=np.uint8)
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        max=0
        second=0
        this_theta=None
        this_rho=None
        this_roadwid=None
        this_t_idx=None
        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))  # diag_len+是因为起始点其实表示负diag_len
            if accumulator[rho, t_idx]>max:
                max = accumulator[rho, t_idx]
                this_theta=thetas[t_idx]
                this_rho=rho
                this_t_idx=t_idx
        this_roadwid_l=0
        this_roadwid_r = 0
        for j in range(1,400): # 设置一个最大值为400，实际应永远达不到:
            # theta 是垂线角度，可以直接用
            x_=x+int(cos_t[this_t_idx]*j)
            y_=y+int(sin_t[this_t_idx]*j)
            if x_y_ok(x_,y_,road_mask) and road_mask[y_,x_]>0:
                this_roadwid_l = j
            else:
                break
        for j in range(1,400):  # 设置一个最大值为400，实际应永远达不到:
            # theta 翻转180度，cos 和 sin 都变成相反数
            x_ = x - int(cos_t[this_t_idx] * j)
            y_ = y - int(sin_t[this_t_idx] * j)
            if x_y_ok(x_, y_, road_mask) and road_mask[y_, x_] > 0:
                this_roadwid_r = j
            else:
                break
            pass
        this_roadwid = 1+this_roadwid_l+this_roadwid_r
        seg_theta[y,x]=this_theta
        seg_rho[y,x]=this_rho
        seg_roadwid[y,x]=this_roadwid

    # global iii
    # show_hough_line(img, accumulator, thetas, rhos, save_path=os.path.join(save_path,'hough/img%d.png'%iii))
    # iii+=1

    return road_mask,seg_theta,seg_rho,seg_roadwid

def hsv2rgb(h,s=1.0,v=1.0): #R, G, B是 [0, 255]. H 是[0, 360]. S, V 是 [0, 1].
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b

def visual_theta_rho(seg_vis, road_mask, seg_theta, seg_rho, seg_roadwid):
    if len(seg_vis.shape)==2:
        seg_vis=gray2rgb(seg_vis)
    # 做一个可视化，带图例的图，以256*256的图为例，rho取值范围为362*2，theta取值范围为-90~90度
    color_card_len=200
    thetas = np.deg2rad(np.arange(-90.0, 90.0, 1))
    thetas_dict=dict(zip(thetas,range(len(thetas))))
    thetas_card=[]
    rhos_card=[]
    roadwid_card = []
    if isinstance(seg_theta,np.ndarray):
        for i in range(len(thetas)):
            h=int(360*i/len(thetas))
            r,g,b=hsv2rgb(h,1.0,1.0)
            thetas_card.append([r,g,b])
    if isinstance(seg_rho, np.ndarray):
        for i in range(362*2):
            h = int(360 * i / (362*2))
            r, g, b = hsv2rgb(h, 1.0, 1.0)
            rhos_card.append([r,g,b])
    if isinstance(seg_roadwid, np.ndarray):
        for i in range(seg_roadwid.max()+1):
            h = int(240 * i / (seg_roadwid.max()+1))
            r, g, b = hsv2rgb(h, 1.0, 1.0)
            roadwid_card.append([r,g,b])
    y_idxs, x_idxs = np.nonzero(road_mask)
    thetas_vis=np.full(seg_vis.shape,255,np.uint8)  # 256*256*3
    rhos_vis=np.full(seg_vis.shape,255,np.uint8)
    roadwid_vis = np.full(seg_vis.shape, 255, np.uint8)
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        if isinstance(seg_theta, np.ndarray):
            if seg_theta[y,x]==-2.0:
                thetas_vis[y, x] = [0,0,0]
            else:
                thetas_vis[y,x]=thetas_card[thetas_dict[seg_theta[y,x]]]
        if isinstance(seg_rho, np.ndarray):
            rhos_vis[y,x]=rhos_card[seg_rho[y,x]]
        if isinstance(seg_roadwid, np.ndarray):
            roadwid_vis[y,x]=roadwid_card[seg_roadwid[y,x]]
    # 做两个色卡条子
    thetas_card_vis=np.zeros((50,200,3),dtype=np.uint8)
    rhos_card_vis=np.zeros((50,200,3),dtype=np.uint8)
    roadwid_card_vis = np.zeros((50, 200, 3), dtype=np.uint8)
    for i in range(200):
        for j in range(50):
            if isinstance(seg_theta, np.ndarray):
                thetas_card_vis[j][i]=thetas_card[int(len(thetas_card)*i/200)]
            if isinstance(seg_rho, np.ndarray):
                rhos_card_vis[j][i]=rhos_card[int(len(rhos_card)*i/200)]
            if isinstance(seg_roadwid, np.ndarray):
                roadwid_card_vis[j][i]=roadwid_card[int(len(roadwid_card)*i/200)]
    # ret=np.full((356,256*4,3),255,dtype=np.uint8)
    # ret[24:74,256:456,:]=thetas_card_vis
    # ret[24:74, 512:712, :] = rhos_card_vis
    # ret[24:74, 768:968, :] = roadwid_card_vis
    # ret[100:356,0:256,:]=seg_vis
    # ret[100:356, 256:512, :] = thetas_vis
    # ret[100:356, 512:768, :] = rhos_vis
    # ret[100:356, 768:1024, :] = roadwid_vis
    ret = np.full((356, 256 * 3, 3), 255, dtype=np.uint8)
    ret[24:74, 256:456, :] = thetas_card_vis
    ret[24:74, 512:712, :] = roadwid_card_vis
    ret[100:356, 0:256, :] = seg_vis
    ret[100:356, 256:512, :] = thetas_vis
    ret[100:356, 512:768, :] = roadwid_vis
    return ret

        






def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


def hough_line(img, angle_step=1, lines_are_white=True, value_threshold=5):
    """
    Hough transform for lines

    Input:
    img - 2D binary image with nonzeros representing edges
    angle_step - Spacing between angles to use every n-th angle
                 between -90 and 90 degrees. Default step is 1.
    lines_are_white - boolean indicating whether lines to be detected are white
    value_threshold - Pixel values above or below the value_threshold are edges

    Returns:
    accumulator - 2D array of the hough transform accumulator
    theta - array of angles used in computation, in radians.
    rhos - array of rho values. Max size is 2 times the diagonal
           distance of the input image.
    """
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    width, height = img.shape
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2) # 在间隔内生成N个数字

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
    # (row, col) indexes to edges
    are_edges = img > value_threshold if lines_are_white else img < value_threshold # 阈值
    y_idxs, x_idxs = np.nonzero(are_edges) #返回数组a中非零元素的索引值数组。

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx])) # diag_len+是因为起始点其实表示负diag_len
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos

   

def show_hough_line(img, accumulator, thetas, rhos, save_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Input image')
    ax[0].axis('image')

    ax[1].imshow(
        accumulator, cmap='jet',
        extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    # plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    imgpath = 'imgs/binary_crosses.png'
    img = imageio.imread(imgpath)
    if img.ndim == 3:
        img = rgb2gray(img)
    accumulator, thetas, rhos = hough_line(img)
    show_hough_line(img, accumulator, thetas, rhos, save_path='imgs/output.png')
