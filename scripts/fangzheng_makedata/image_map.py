import os

root_low = "D:\\project\\北理项目\\map_project\\map\\18"
root_high = "D:\\project\\北理项目\\map_project\\map\\17"
high_title = "tile-11-"
suffix = ".jpg"
path_txt = "11_03.txt"
good_ratio = 0.55

def write_txt(x, y):
    new_name = high_title + str(x) + "-" + str(y) + suffix
    new_path = os.path.join(root_high, new_name)
    with open(path_txt, "a") as f:
        f.write(new_path + "\n")


def find_image_right(low_x1, low_x2, low_y1, low_y2):
    num_image = 0
    good_num = 0
    for x in range(low_x1, low_x2):
        for y in range(low_y1, low_y2):
            num_image += 1
            low_x_and_y = str(x) + "-" + str(y)
            if low_x_and_y in good_image_list:
                good_num += 1

    if good_num / num_image > 0.5:
        print(good_num / num_image)
        return True
    else:
        print(good_num / num_image)
        return False

file = open("D:\project\北理项目\map_project\上海\18_seg2\log\log\pick03_SH_18_new.txt")
good_image_list = []
for line in file:
    x_and_y = line.split("-")[2] + "-" + line.split("-")[3].split(".")[0]
    good_image_list.append(x_and_y)
file.close()

image_list_low = os.listdir(root_low)
image_list_high = os.listdir(root_high)

min_x1 = 100000
min_y1 = 100000
max_x1 = 0
max_y1 = 0
for img_low in image_list_low:
    image_names = img_low.split("-")
    x = image_names[2]
    y = image_names[3].split(".")[0]
    if int(x) > max_x1:
        max_x1 = int(x)
    if int(x) < min_x1:
        min_x1 = int(x)
    if int(y) > max_y1:
        max_y1 = int(y)
    if int(y) < min_y1:
        min_y1 = int(y)

min_x2 = 100000
min_y2 = 100000
max_x2 = 0
max_y2 = 0
for img_high in image_list_high:
    image_names = img_high.split("-")
    x = image_names[2]
    y = image_names[3].split(".")[0]
    if int(x) > max_x2:
        max_x2 = int(x)
    if int(x) < min_x2:
        min_x2 = int(x)
    if int(y) > max_y2:
        max_y2 = int(y)
    if int(y) < min_y2:
        min_y2 = int(y)

len_x1 = max_x1 - min_x1 + 1
len_y1 = max_y1 - min_y1 + 1
len_x2 = max_x2 - min_x2 + 1
len_y2 = max_y2 - min_y2 + 1

print(len_x1)
print(len_y1)
print(len_x2)
print(len_y2)

ratio_x = len_x1 / len_x2
ratio_y = len_y1 / len_y2

for i in range(len_x2):
    x = min_x2 + i
    for j in range(len_y2):
        y = min_y2 + j
        low_i = i * ratio_x
        low_j = j * ratio_y
        low_x1 = round(min_x1 + low_i)
        low_y1 = round(min_y1 + low_j)
        low_x2 = round(low_x1 + ratio_x)
        low_y2 = round(low_y1 + ratio_y)
        if find_image_right(low_x1, low_x2, low_y1, low_y2):
            write_txt(x, y)


