import os
import shutil
old_path = r"D:\project\北理项目\map_project\多层联合\data_all\fake_mix\fake_result"
new_path = r"D:\project\北理项目\map_project\多层联合\data_all\fake_mix\newfake_result"
all_imgs = os.listdir(old_path)
if not os.path.exists(new_path):
    os.makedirs(new_path)

for img in all_imgs:
    print(img)
    img_names = img.split("_")
    img_layer = os.path.join(new_path, img_names[0])
    img_x = os.path.join(new_path, img_names[0], img_names[1])
    img_y = os.path.join(new_path, img_names[0], img_names[1], img_names[2].split(".")[0]+".png")
    if not os.path.exists(img_layer):
        os.makedirs(img_layer)
    if not os.path.exists(img_x):
        os.makedirs(img_x)
    shutil.copy(os.path.join(old_path,img), img_y)
    print(img_y)