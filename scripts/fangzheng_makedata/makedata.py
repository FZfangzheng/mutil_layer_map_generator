import os
import shutil

root_map = "D:\\Document\\北理项目\\第四批数据\\map"
root_rs = "D:\\Document\\北理项目\\第四批数据\\rs"

map_list = os.listdir(root_map)
rs_list = os.listdir(root_rs)

for img in map_list:
    img_path = os.path.join(root_map, img)
    image_names = img.split("-")
    dir_names = [image_names[0], image_names[1], image_names[2]]
    dir1 = os.path.join(root_map, dir_names[0])
    dir2 = os.path.join(root_map, dir_names[0], dir_names[1])
    img_name = dir_names[2]
    if not os.path.isdir(dir1):
        os.makedirs(dir1)
        os.makedirs(dir2)
    elif not os.path.isdir(dir2):
        os.makedirs(dir2)
    new_path = os.path.join(dir2, img_name)
    shutil.copy(img_path, new_path)
    print(new_path)


for img in rs_list:
    img_path = os.path.join(root_rs, img)
    image_names = img.split("-")
    dir_names = [image_names[0], image_names[1], image_names[2]]
    dir1 = os.path.join(root_rs, dir_names[0])
    dir2 = os.path.join(root_rs, dir_names[0], dir_names[1])
    img_name = dir_names[2]
    if not os.path.isdir(dir1):
        os.makedirs(dir1)
        os.makedirs(dir2)
    elif not os.path.isdir(dir2):
        os.makedirs(dir2)
    new_path = os.path.join(dir2, img_name)
    shutil.copy(img_path, new_path)
    print(new_path)
