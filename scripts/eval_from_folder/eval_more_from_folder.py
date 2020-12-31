import numpy as np
from PIL import Image
import sys
from eval.evaler import eval_memo
import json
import os
from random import shuffle
import shutil
import datetime

from eval.utils import make_dataset,get_inner_path

''''''

if __name__=='__main__':
    if len(sys.argv) > 1:
        real_path = sys.argv[1]
        fake_path = sys.argv[2]
    else:
        real_path = r"D:\map_translate\看看效果\0601TW16_1708图_分割结果做p2p输入，epoch200，重新训练\real_result"
        fake_path = r"D:\map_translate\看看效果\0601TW16_1708图_分割结果做p2p输入，epoch200，重新训练\fake_result"

    score_filename = 'log_p2p_' + str(0) + 'real.json'
    with open(os.path.join(os.getcwd(), score_filename), 'a') as f:
        json.dump(str(datetime.datetime.now()), f)
        f.write('\n')
        json.dump(real_path, f)
        f.write('\n')
        json.dump(fake_path, f)
        f.write('\n')

    # 评估real结果
    print("process REAL")
    print("folder processing...")
    imgsA = make_dataset(real_path)
    imgsB = make_dataset(fake_path)

    for s in [ 'inception_v3', 'resnet18']:#[ 'inception_v3','vgg13' ,'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        evaler=eval_memo(len(imgsA),conv_models=[s],gpu='0',needinception=True,needmode=True,needwasserstein=True)
        for i, f in enumerate(imgsA):
            img = np.array(Image.open(f))
            img=np.expand_dims(img,axis=0)
            evaler.add_imgA(img)
            print("\r%d/%d" % (i, len(imgsA)),end=" ")
        print("\n")
        for i, f in enumerate(imgsB):
            img = np.array(Image.open(f))
            img = np.expand_dims(img, axis=0)
            evaler.add_imgB(img)
            print("\r%d/%d" % (i, len(imgsB)),end=" ")
        print("\n")

        score=evaler.get_score()
        print(score)

        with open(os.path.join(os.getcwd(), score_filename), 'a') as f:
            json.dump(score, f)
            f.write('\n')


    print('另外的几个分数')
    from fid.fid_score import fid_score
    from MSE import MSE_from_floder
    from SSIM import SSIM_from_floder
    fid = fid_score(real_path=real_path, fake_path=fake_path, gpu='')
    mse = MSE_from_floder(real_path, fake_path)
    ssim = SSIM_from_floder(real_path, fake_path)
    print(f'===> fid score:{fid:.4f},===> mse score:{mse:.4f},===> ssim score:{ssim:.4f}')

    with open(os.path.join(os.getcwd(), score_filename), 'a') as f:
        json.dump(f'===> fid score:{fid:.4f},===> mse score:{mse:.4f},===> ssim score:{ssim:.4f}', f)
        f.write('\n')
        f.write('\n')
        f.write('\n')




