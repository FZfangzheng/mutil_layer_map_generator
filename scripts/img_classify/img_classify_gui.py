from tkinter import *
import PIL
from PIL import Image
from PIL import ImageTk
import tkinter as tk
import os
import matplotlib.pyplot as plt
import tkinter.messagebox

root_path = r'D:\map_translate\数据集\WH16分析\list.txt'
log_path = r'D:\map_translate\数据集\WH16分析\log'
img_list = []
for i in open(root_path, 'r'):
    img_list.append(i[:len(i) - 1])
length = len(img_list)
str_v1 = StringVar
num = 0

# 断点续传
try:
    tmpfile = open(os.path.join(log_path,'log.txt'), 'r')
except:
    pass
else:
    num=int(tmpfile.read())+1



def last_image():
    global num, names, f_file_log
    f_file_log = open(os.path.join(log_path,'log.txt'), 'w')
    f_file_log.write(str(num))
    del names['path' + str(num)]
    del names['pilimage' + str(num)]
    del names['im_w' + str(num)]
    del names['im_h' + str(num)]
    del names['img' + str(num)]
    num = num - 1
    if (num < 0):
        tk.messagebox.showinfo(title='提示', message='已经到头了！')
    else:
        names['path' + str(num)] = img_list[num]
        names['pilimage' + str(num)] = Image.open(names['path' + str(num)])
        names['im_w' + str(num)] = names['pilimage' + str(num)].size[0]
        names['im_h' + str(num)] = names['pilimage' + str(num)].size[1]
        alpha = names['im_h' + str(num)] / names['im_w' + str(num)]
        names['pilimage' + str(num)] = names['pilimage' + str(num)].resize((700, int(700 * alpha)))
        names['img' + str(num)] = ImageTk.PhotoImage(names['pilimage' + str(num)])
        imLabel.configure(image=names['img' + str(num)])
        txLabel.configure(text='No: ' + str(num))
        nm_en.delete(0, END)
        nm_en.insert(0, img_list[num].split('/')[-1])
        nm_en1.delete(0, END)
        nm_en1.insert(0, img_list[num].split('\\')[-1])


def next_image():
    global num, names, f_file_log
    f_file_log = open(os.path.join(log_path,'log.txt'), 'w')
    f_file_log.write(str(num))
    del names['path' + str(num)]
    del names['pilimage' + str(num)]
    del names['im_w' + str(num)]
    del names['im_h' + str(num)]
    del names['img' + str(num)]
    num = num + 1
    if (num > length - 1):
        tk.messagebox.showinfo(title='提示', message='已经没有了！')
    else:
        names['path' + str(num)] = img_list[num]
        names['pilimage' + str(num)] = Image.open(names['path' + str(num)])
        names['im_w' + str(num)] = names['pilimage' + str(num)].size[0]
        names['im_h' + str(num)] = names['pilimage' + str(num)].size[1]
        alpha = names['im_h' + str(num)] / names['im_w' + str(num)]
        names['pilimage' + str(num)] = names['pilimage' + str(num)].resize((700, int(700 * alpha)))
        names['img' + str(num)] = ImageTk.PhotoImage(names['pilimage' + str(num)])
        imLabel.configure(image=names['img' + str(num)])
        txLabel.configure(text='No: ' + str(num))
        nm_en.delete(0, END)
        nm_en.insert(0, img_list[num].split('/')[-1])
        nm_en1.delete(0, END)
        nm_en1.insert(0, img_list[num].split('\\')[-1])


def pick_image00():
    global num, names, f_file_log, f_file00
    f_file_log = open(os.path.join(log_path,'log.txt'), 'w')
    f_file_log.write(str(num))
    f_file00=open(os.path.join(log_path,'pick00.txt'),'a')
    f_file00.write(img_list[num])
    f_file00.write('\n')
    f_file00.close()
    del names['path' + str(num)]
    del names['pilimage' + str(num)]
    del names['im_w' + str(num)]
    del names['im_h' + str(num)]
    del names['img' + str(num)]
    num = num + 1
    if (num > length - 1):
        tk.messagebox.showinfo(title='提示', message='已经没有了！')
    else:
        names['path' + str(num)] = img_list[num]
        names['pilimage' + str(num)] = Image.open(names['path' + str(num)])
        names['im_w' + str(num)] = names['pilimage' + str(num)].size[0]
        names['im_h' + str(num)] = names['pilimage' + str(num)].size[1]
        alpha = names['im_h' + str(num)] / names['im_w' + str(num)]
        names['pilimage' + str(num)] = names['pilimage' + str(num)]# .resize((700, int(700 * alpha)))
        names['img' + str(num)] = ImageTk.PhotoImage(names['pilimage' + str(num)])
        imLabel.configure(image=names['img' + str(num)])
        txLabel.configure(text='No: ' + str(num))
        nm_en.delete(0, END)
        nm_en.insert(0, img_list[num].split('/')[-1])
        nm_en1.delete(0, END)
        nm_en1.insert(0, img_list[num].split('\\')[-1])

def pick_image01():
    global num, names, f_file_log, f_file00
    f_file_log = open(os.path.join(log_path,'log.txt'), 'w')
    f_file_log.write(str(num))
    f_file00=open(os.path.join(log_path,'pick01.txt'),'a')
    f_file00.write(img_list[num])
    f_file00.write('\n')
    f_file00.close()
    del names['path' + str(num)]
    del names['pilimage' + str(num)]
    del names['im_w' + str(num)]
    del names['im_h' + str(num)]
    del names['img' + str(num)]
    num = num + 1
    if (num > length - 1):
        tk.messagebox.showinfo(title='提示', message='已经没有了！')
    else:
        names['path' + str(num)] = img_list[num]
        names['pilimage' + str(num)] = Image.open(names['path' + str(num)])
        names['im_w' + str(num)] = names['pilimage' + str(num)].size[0]
        names['im_h' + str(num)] = names['pilimage' + str(num)].size[1]
        alpha = names['im_h' + str(num)] / names['im_w' + str(num)]
        names['pilimage' + str(num)] = names['pilimage' + str(num)]# .resize((700, int(700 * alpha)))
        names['img' + str(num)] = ImageTk.PhotoImage(names['pilimage' + str(num)])
        imLabel.configure(image=names['img' + str(num)])
        txLabel.configure(text='No: ' + str(num))
        nm_en.delete(0, END)
        nm_en.insert(0, img_list[num].split('/')[-1])
        nm_en1.delete(0, END)
        nm_en1.insert(0, img_list[num].split('\\')[-1])

def pick_image02():
    global num, names, f_file_log, f_file00
    f_file_log = open(os.path.join(log_path,'log.txt'), 'w')
    f_file_log.write(str(num))
    f_file00=open(os.path.join(log_path,'pick02.txt'),'a')
    f_file00.write(img_list[num])
    f_file00.write('\n')
    f_file00.close()
    del names['path' + str(num)]
    del names['pilimage' + str(num)]
    del names['im_w' + str(num)]
    del names['im_h' + str(num)]
    del names['img' + str(num)]
    num = num + 1
    if (num > length - 1):
        tk.messagebox.showinfo(title='提示', message='已经没有了！')
    else:
        names['path' + str(num)] = img_list[num]
        names['pilimage' + str(num)] = Image.open(names['path' + str(num)])
        names['im_w' + str(num)] = names['pilimage' + str(num)].size[0]
        names['im_h' + str(num)] = names['pilimage' + str(num)].size[1]
        alpha = names['im_h' + str(num)] / names['im_w' + str(num)]
        names['pilimage' + str(num)] = names['pilimage' + str(num)]# .resize((700, int(700 * alpha)))
        names['img' + str(num)] = ImageTk.PhotoImage(names['pilimage' + str(num)])
        imLabel.configure(image=names['img' + str(num)])
        txLabel.configure(text='No: ' + str(num))
        nm_en.delete(0, END)
        nm_en.insert(0, img_list[num].split('/')[-1])
        nm_en1.delete(0, END)
        nm_en1.insert(0, img_list[num].split('\\')[-1])

def pick_image03():
    global num, names, f_file_log, f_file00
    f_file_log = open(os.path.join(log_path,'log.txt'), 'w')
    f_file_log.write(str(num))
    f_file00=open(os.path.join(log_path,'pick03.txt'),'a')
    f_file00.write(img_list[num])
    f_file00.write('\n')
    f_file00.close()
    del names['path' + str(num)]
    del names['pilimage' + str(num)]
    del names['im_w' + str(num)]
    del names['im_h' + str(num)]
    del names['img' + str(num)]
    num = num + 1
    if (num > length - 1):
        tk.messagebox.showinfo(title='提示', message='已经没有了！')
    else:
        names['path' + str(num)] = img_list[num]
        names['pilimage' + str(num)] = Image.open(names['path' + str(num)])
        names['im_w' + str(num)] = names['pilimage' + str(num)].size[0]
        names['im_h' + str(num)] = names['pilimage' + str(num)].size[1]
        alpha = names['im_h' + str(num)] / names['im_w' + str(num)]
        names['pilimage' + str(num)] = names['pilimage' + str(num)]# .resize((700, int(700 * alpha)))
        names['img' + str(num)] = ImageTk.PhotoImage(names['pilimage' + str(num)])
        imLabel.configure(image=names['img' + str(num)])
        txLabel.configure(text='No: ' + str(num))
        nm_en.delete(0, END)
        nm_en.insert(0, img_list[num].split('/')[-1])
        nm_en1.delete(0, END)
        nm_en1.insert(0, img_list[num].split('\\')[-1])

def pick_image04():
    global num, names, f_file_log, f_file00
    f_file_log = open(os.path.join(log_path,'log.txt'), 'w')
    f_file_log.write(str(num))
    f_file00=open(os.path.join(log_path,'pick04.txt'),'a')
    f_file00.write(img_list[num])
    f_file00.write('\n')
    f_file00.close()
    del names['path' + str(num)]
    del names['pilimage' + str(num)]
    del names['im_w' + str(num)]
    del names['im_h' + str(num)]
    del names['img' + str(num)]
    num = num + 1
    if (num > length - 1):
        tk.messagebox.showinfo(title='提示', message='已经没有了！')
    else:
        names['path' + str(num)] = img_list[num]
        names['pilimage' + str(num)] = Image.open(names['path' + str(num)])
        names['im_w' + str(num)] = names['pilimage' + str(num)].size[0]
        names['im_h' + str(num)] = names['pilimage' + str(num)].size[1]
        alpha = names['im_h' + str(num)] / names['im_w' + str(num)]
        names['pilimage' + str(num)] = names['pilimage' + str(num)]# .resize((700, int(700 * alpha)))
        names['img' + str(num)] = ImageTk.PhotoImage(names['pilimage' + str(num)])
        imLabel.configure(image=names['img' + str(num)])
        txLabel.configure(text='No: ' + str(num))
        nm_en.delete(0, END)
        nm_en.insert(0, img_list[num].split('/')[-1])
        nm_en1.delete(0, END)
        nm_en1.insert(0, img_list[num].split('\\')[-1])

def pick_image05():
    global num, names, f_file_log, f_file00
    f_file_log = open(os.path.join(log_path,'log.txt'), 'w')
    f_file_log.write(str(num))
    f_file00=open(os.path.join(log_path,'pick05.txt'),'a')
    f_file00.write(img_list[num])
    f_file00.write('\n')
    f_file00.close()
    del names['path' + str(num)]
    del names['pilimage' + str(num)]
    del names['im_w' + str(num)]
    del names['im_h' + str(num)]
    del names['img' + str(num)]
    num = num + 1
    if (num > length - 1):
        tk.messagebox.showinfo(title='提示', message='已经没有了！')
    else:
        names['path' + str(num)] = img_list[num]
        names['pilimage' + str(num)] = Image.open(names['path' + str(num)])
        names['im_w' + str(num)] = names['pilimage' + str(num)].size[0]
        names['im_h' + str(num)] = names['pilimage' + str(num)].size[1]
        alpha = names['im_h' + str(num)] / names['im_w' + str(num)]
        names['pilimage' + str(num)] = names['pilimage' + str(num)]# .resize((700, int(700 * alpha)))
        names['img' + str(num)] = ImageTk.PhotoImage(names['pilimage' + str(num)])
        imLabel.configure(image=names['img' + str(num)])
        txLabel.configure(text='No: ' + str(num))
        nm_en.delete(0, END)
        nm_en.insert(0, img_list[num].split('/')[-1])
        nm_en1.delete(0, END)
        nm_en1.insert(0, img_list[num].split('\\')[-1])

def pick_image06():
    global num, names, f_file_log, f_file00
    f_file_log = open(os.path.join(log_path,'log.txt'), 'w')
    f_file_log.write(str(num))
    f_file00=open(os.path.join(log_path,'pick06.txt'),'a')
    f_file00.write(img_list[num])
    f_file00.write('\n')
    f_file00.close()
    del names['path' + str(num)]
    del names['pilimage' + str(num)]
    del names['im_w' + str(num)]
    del names['im_h' + str(num)]
    del names['img' + str(num)]
    num = num + 1
    if (num > length - 1):
        tk.messagebox.showinfo(title='提示', message='已经没有了！')
    else:
        names['path' + str(num)] = img_list[num]
        names['pilimage' + str(num)] = Image.open(names['path' + str(num)])
        names['im_w' + str(num)] = names['pilimage' + str(num)].size[0]
        names['im_h' + str(num)] = names['pilimage' + str(num)].size[1]
        alpha = names['im_h' + str(num)] / names['im_w' + str(num)]
        names['pilimage' + str(num)] = names['pilimage' + str(num)]# .resize((700, int(700 * alpha)))
        names['img' + str(num)] = ImageTk.PhotoImage(names['pilimage' + str(num)])
        imLabel.configure(image=names['img' + str(num)])
        txLabel.configure(text='No: ' + str(num))
        nm_en.delete(0, END)
        nm_en.insert(0, img_list[num].split('/')[-1])
        nm_en1.delete(0, END)
        nm_en1.insert(0, img_list[num].split('\\')[-1])

def pick_image07():
    global num, names, f_file_log, f_file00
    f_file_log = open(os.path.join(log_path,'log.txt'), 'w')
    f_file_log.write(str(num))
    f_file00=open(os.path.join(log_path,'pick07.txt'),'a')
    f_file00.write(img_list[num])
    f_file00.write('\n')
    f_file00.close()
    del names['path' + str(num)]
    del names['pilimage' + str(num)]
    del names['im_w' + str(num)]
    del names['im_h' + str(num)]
    del names['img' + str(num)]
    num = num + 1
    if (num > length - 1):
        tk.messagebox.showinfo(title='提示', message='已经没有了！')
    else:
        names['path' + str(num)] = img_list[num]
        names['pilimage' + str(num)] = Image.open(names['path' + str(num)])
        names['im_w' + str(num)] = names['pilimage' + str(num)].size[0]
        names['im_h' + str(num)] = names['pilimage' + str(num)].size[1]
        alpha = names['im_h' + str(num)] / names['im_w' + str(num)]
        names['pilimage' + str(num)] = names['pilimage' + str(num)]# .resize((700, int(700 * alpha)))
        names['img' + str(num)] = ImageTk.PhotoImage(names['pilimage' + str(num)])
        imLabel.configure(image=names['img' + str(num)])
        txLabel.configure(text='No: ' + str(num))
        nm_en.delete(0, END)
        nm_en.insert(0, img_list[num].split('/')[-1])
        nm_en1.delete(0, END)
        nm_en1.insert(0, img_list[num].split('\\')[-1])


def key(event):
    if (event.char == '0'):
        pick_image00()
    if (event.char == '1'):
        pick_image01()
    if (event.char == '2'):
        pick_image02()
    if (event.char == '3'):
        pick_image03()
    if (event.char == '4'):
        pick_image04()
    if (event.char == '5'):
        pick_image05()
    if (event.char == '6'):
        pick_image06()
    if (event.char == '7'):
        pick_image07()



# if __name__=='__main__':
names = locals()
root = tk.Tk()
root.title('FYJ')
root['background'] = 'mintcream'
root.geometry('1800x800')
root.resizable(width=False, height=False)
btn1 = tk.Button(root, text='Next Image', command=next_image)
btn1['background'] = 'mintcream'
btn1.place(x=10, y=10, width=120, height=30)

btn2 = tk.Button(root, text='0.纯水域', command=pick_image00)
btn2['background'] = 'mintcream'
btn2.place(x=10, y=150, width=120, height=30)

btn3 = tk.Button(root, text='1.要素无法看到', command=pick_image01)
btn3['background'] = 'mintcream'
btn3.place(x=10, y=200, width=120, height=30)

btn4 = tk.Button(root, text='2.要素难以看到', command=pick_image02)
btn4['background'] = 'mintcream'
btn4.place(x=10, y=250, width=120, height=30)

btn5 = tk.Button(root, text='3.要素比较清晰', command=pick_image03)
btn5['background'] = 'mintcream'
btn5.place(x=10, y=300, width=120, height=30)

btn6 = tk.Button(root, text='4.要素对齐偏差', command=pick_image04)
btn6['background'] = 'mintcream'
btn6.place(x=10, y=350, width=120, height=30)

btn7 = tk.Button(root, text='5.待定1', command=pick_image05)
btn7['background'] = 'mintcream'
btn7.place(x=10, y=400, width=120, height=30)

btn8 = tk.Button(root, text='6.待定2', command=pick_image06)
btn8['background'] = 'mintcream'
btn8.place(x=10, y=450, width=120, height=30)

btn9 = tk.Button(root, text='7.待定3', command=pick_image07)
btn9['background'] = 'mintcream'
btn9.place(x=10, y=500, width=120, height=30)


txLabel = tk.Label(root, text='No: ' + str(num))
txLabel['background'] = 'mintcream'
txLabel.place(x=10, y=570, width=120, height=30)
nm_en = Entry(root, textvariable=str_v1)
nm_en['background'] = 'mintcream'
nm_en.insert(0, img_list[num])
nm_en.place(x=10, y=630, width=120, height=30)

nm_en1 = Entry(root, textvariable=str_v1)
nm_en1['background'] = 'mintcream'
nm_en1.insert(0, img_list[num].split('\\')[-1])
nm_en1.place(x=10, y=730, width=120, height=30)

frame = tk.Frame(root, width=100, height=100, background='pink')
frame.bind('<Key>', key)
frame.place(x=10, y=800, width=120, height=30)
frame.focus_set()

txLabel1 = tk.Label(root, text='Total: ' + str(length))
txLabel1['background'] = 'mintcream'
txLabel1.place(x=10, y=600, width=120, height=30)
names['path' + str(num)] = img_list[num]
names['pilimage' + str(num)] = Image.open(names['path' + str(num)])
names['im_w' + str(num)] = names['pilimage' + str(num)].size[0]
names['im_h' + str(num)] = names['pilimage' + str(num)].size[1]
alpha = names['im_h' + str(num)] / names['im_w' + str(num)]
names['pilimage' + str(num)] = names['pilimage' + str(num)]# .resize((1536, int(1536 * alpha)))
names['img' + str(num)] = ImageTk.PhotoImage(names['pilimage' + str(num)])
imLabel = tk.Label(root, image=names['img' + str(num)], width=1550, height=560)
imLabel['background'] = 'mintcream'
imLabel.pack()
root.mainloop()