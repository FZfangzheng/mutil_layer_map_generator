import os
import shutil

root_file = "D:\\project\\北理项目\\map_project\\上海\\18_seg2\\log\\log\\15_03_SH.txt"
root_new_map = "D:\\project\\北理项目\\map_project\\上海\\18_seg2\\log\\log\\15\\good_img_map"
root_new_rs = "D:\\project\\北理项目\\map_project\\上海\\18_seg2\\log\\log\\15\\good_img_rs"
root_rs="D:\\project\\北理项目\\map_project\\上海\\rs\\15_png3"
root_map="D:\\project\\北理项目\\map_project\\上海\\18_seg2\\15\\15_seg\\seg"
with open(root_file) as f:
    imgs_list = f.readlines()
for i in imgs_list:
    print(i)
    i = i.rstrip('\n')
    dir_list = i.split("\\")
    x_dir_rs = os.path.join(root_new_rs, dir_list[7])
    x_dir_map = os.path.join(root_new_map, dir_list[7])
    if not os.path.exists(x_dir_rs):
        os.mkdir(x_dir_rs)
    if not os.path.exists(x_dir_map):
        os.mkdir(x_dir_map)
    img_path_rs = os.path.join(root_rs,dir_list[7], dir_list[8])
    img_path_map = os.path.join(root_map,dir_list[7], dir_list[8])
    new_path_rs = os.path.join(root_new_rs, dir_list[7], dir_list[8])
    new_path_map = os.path.join(root_new_map, dir_list[7], dir_list[8])
    shutil.copy(img_path_rs, new_path_rs)
    shutil.copy(img_path_map, new_path_map)
