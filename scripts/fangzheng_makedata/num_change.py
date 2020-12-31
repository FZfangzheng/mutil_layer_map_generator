import os
from shutil import copy
root_path = r"D:\project\北理项目\map_project\多层联合\data_all\fake_mix\newfake_result"
new_path = r"D:\project\北理项目\map_project\多层联合\data_all\fake_mix\newfake_result_for_show"
list_layers = os.listdir(root_path)
if not os.path.exists(new_path):
    os.mkdir(new_path)
for str_num_layer in list_layers:
    num_layer = int(str_num_layer) - 1
    new_layers = os.path.join(new_path, str(num_layer))
    if not os.path.exists(new_layers):
        os.mkdir(new_layers)
    old_layers = os.path.join(root_path, str_num_layer)
    layers_x = os.listdir(old_layers)
    x_list = []
    for x in layers_x:
        img_dir = os.path.join(old_layers, x)
        layers_y = os.listdir(img_dir)
        for y in layers_y:
            new_img_dir_path = os.path.join(new_layers, y.split('.')[0])
            if not os.path.exists(new_img_dir_path):
                os.mkdir(new_img_dir_path)
            img_path = os.path.join(img_dir, y)
            new_img_path = os.path.join(new_img_dir_path, x + '.' + y.split('.')[1])
            copy(img_path, new_img_path)
            print(img_path)
