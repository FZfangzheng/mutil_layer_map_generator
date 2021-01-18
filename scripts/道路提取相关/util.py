import numpy as np

def gray2rgb(gray,n_class=3,label_list=[[239,238,236],[255,255,255],[170,218,255]]): # gray: np:h*w
    h,w=gray.shape
    mask=[]
    rgb=np.zeros((h,w,3))
    for i in range(n_class):
        tmp=(gray==i)
        mask.append(tmp)
        rgb+=np.expand_dims(tmp,2).repeat(3,axis=2)*label_list[i]
    rgb=rgb.astype(np.uint8)
    return rgb