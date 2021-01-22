__author__ = "charles"
__email__ = "charleschen2013@163.com"
import os
from os import path as osp
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append(osp.join(sys.path[0], '../'))
sys.path.append(osp.join(sys.path[0], '../../'))
# sys.path.append(osp.join(sys.path[0], '../../../'))
import time
import torch
import shutil
import torch.nn as nn
from src.utils.train_utils import model_accelerate, get_device, mean, get_lr
from src.pix2pixHD.train_config import config
from src.pix2pixHD.networks import get_G, get_D, get_E
from torch.optim import Adam
from src.pix2pixHD.hinge_lr_scheduler import get_hinge_scheduler
from src.utils.logger import ModelSaver, Logger
from src.datasets import get_pix2pix_maps_dataloader
from src.datasets.pix2pix_maps import get_inter1_dataloader
from src.pix2pixHD.utils import get_edges, label_to_one_hot, get_encode_features
from src.utils.visualizer import Visualizer
from tqdm import tqdm
from torchvision import transforms
from src.pix2pixHD.criterion import get_GANLoss, get_VGGLoss, get_DFLoss, get_low_level_loss
from tensorboardX import SummaryWriter
from src.pix2pixHD.utils import from_std_tensor_save_image, create_dir
from src.data.image_folder import make_dataset
import shutil
import numpy as np
from PIL import Image

from src.pix2pixHD.deeplabv3plus.deeplabv3plus import Configuration
from src.pix2pixHD.deeplabv3plus.deeplabv3plus.my_deeplabv3plus_featuremap import deeplabv3plus
import torch.nn.functional as F

from src.pix2pixHD.deeplabv3plus.lovasz_losses import lovasz_softmax

from util.util import tensor2im  # 注意，该函数使用0.5与255恢复可视图像，所以如果是ImageNet标准化的可能会有色差？这里都显示试一下
from src.pix2pixHD.myutils import pred2gray, gray2rgb

from src.pix2pixHD.deeplabv3plus.focal_loss import FocalLoss
from evaluation.fid.fid_score import fid_score
import json
from src.eval.eval_every10epoch import eval_epoch

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

def eval_fidiou(args, sw, model_G, epoch=-1):
    fake_dir = osp.join(args.save, 'fake_result')
    real_dir = osp.join(args.save, 'real_result')
    eval_dir = osp.join(args.save, 'eval_result')
    create_dir(fake_dir)
    create_dir(real_dir)
    create_dir(eval_dir)
    index_layer=0
    for layer in range(args.layer_num):
        # start from layer_num
        id_layer = args.layer_num - layer
        if layer == 0:
            real_img_path = os.path.join(real_dir, str(id_layer))

            before_img_path = os.path.join(args.dataroot, "test", "A", str(id_layer))
            after_img_path = os.path.join(args.dataroot, "test", "B", str(id_layer))
            gt_img_path = os.path.join(args.dataroot, "test", "C", str(id_layer))
            create_dir(after_img_path)
            create_dir(real_img_path)

            img_list = make_dataset(before_img_path)
            for img_before in img_list:
                img_name = os.path.split(img_before)[1]
                img_after = os.path.join(after_img_path, img_name)
                img_real = os.path.join(real_img_path, img_name)

                img_gt = os.path.join(gt_img_path, img_name)
                shutil.copy(img_before, img_after)

                shutil.copy(img_gt, img_real)
        else:
            index_layer=index_layer+1
            data_loader = get_inter1_dataloader(args, id_layer, train=False)
            data_loader = tqdm(data_loader)
            model_G[index_layer-1].eval()
            device = get_device(args)
            model_G[index_layer-1] = model_G[index_layer-1].to(device)


            real_img_dir = os.path.join(real_dir, str(id_layer))
            test_B_dir = os.path.join(args.dataroot, "test", "B", str(id_layer))

            create_dir(real_img_dir)
            create_dir(test_B_dir)

            for i, sample in enumerate(data_loader):
                layer_imgs = sample['A'].to(device)
                final_before_img = sample['B'].to(device)
                gt_imgs = sample['C'].to(device)

                imgs_plus = torch.cat((layer_imgs.float(),final_before_img.float()),1)
                fakes = model_G[index_layer-1](imgs_plus).detach()

                batch_size = layer_imgs.size(0)
                im_name = sample['A_path']
                for b in range(batch_size):
                    file_name = osp.split(im_name[b])[-1].split('.')[0]
                    real_file = osp.join(real_img_dir, f'{file_name}.png')

                    test_B_file = osp.join(test_B_dir, f'{file_name}.png')

                    from_std_tensor_save_image(filename=test_B_file, data=fakes[b].cpu())
                    from_std_tensor_save_image(filename=real_file, data=gt_imgs[b].cpu())
    #   反向传递
    for layer in range(args.layer_num):
        # start from layer_num
        id_layer = layer + 1
        if layer == 0:
            fake_img_path = os.path.join(fake_dir, str(id_layer))
            before_img_path = os.path.join(args.dataroot, "test", "B", str(id_layer))
            after_img_path = os.path.join(args.dataroot, "test", "B1", str(id_layer))
            gt_img_path = os.path.join(args.dataroot, "test", "C", str(id_layer))
            create_dir(after_img_path)
            create_dir(fake_img_path)
            img_list = make_dataset(before_img_path)
            for img_before in img_list:
                img_name = os.path.split(img_before)[1]
                img_after = os.path.join(after_img_path, img_name)
                img_fake = os.path.join(fake_img_path, img_name)
                img_gt = os.path.join(gt_img_path, img_name)
                shutil.copy(img_before, img_after)
                shutil.copy(img_before, img_fake)
        else:
            index_layer=index_layer+1
            data_loader = get_inter1_dataloader(args, id_layer, train=False, flag=1)
            data_loader = tqdm(data_loader)
            model_G[index_layer-1].eval()
            device = get_device(args)
            model_G[index_layer-1] = model_G[index_layer-1].to(device)

            fake_img_dir = osp.join(fake_dir, str(id_layer))
            real_img_dir = os.path.join(real_dir, str(id_layer))
            test_B_dir = os.path.join(args.dataroot, "test", "B1", str(id_layer))
            create_dir(fake_img_dir)
            create_dir(real_img_dir)
            create_dir(test_B_dir)

            for i, sample in enumerate(data_loader):
                layer_imgs = sample['A'].to(device)
                final_before_img = sample['B'].to(device)

                imgs_plus = torch.cat((layer_imgs.float(),final_before_img.float()),1)
                fakes = model_G[index_layer-1](imgs_plus).detach()

                batch_size = layer_imgs.size(0)
                im_name = sample['A_path']
                for b in range(batch_size):
                    file_name = osp.split(im_name[b])[-1].split('.')[0]

                    fake_file = osp.join(fake_img_dir, f'{file_name}.png')
                    test_B_file = osp.join(test_B_dir, f'{file_name}.png')
                    from_std_tensor_save_image(filename=fake_file, data=fakes[b].cpu())
                    from_std_tensor_save_image(filename=test_B_file, data=fakes[b].cpu())

    real_img_list = make_dataset(real_dir)
    fake_img_list = make_dataset(fake_dir)
    new_fake_dir = osp.join(args.save, 'fake_result_without_layer')
    new_real_dir = osp.join(args.save, 'real_result_without_layer')
    create_dir(new_fake_dir)
    create_dir(new_real_dir)
    for real_img in real_img_list:
        img_name = osp.split(real_img)[-1]
        new_real_img_path = osp.join(new_real_dir, img_name)
        shutil.copy(real_img, new_real_img_path)
    for fake_img in fake_img_list:
        img_name = osp.split(fake_img)[-1]
        new_fake_img_path = osp.join(new_fake_dir, img_name)
        shutil.copy(fake_img, new_fake_img_path)
    real_paths = [osp.join(real_dir,"1"),osp.join(real_dir,"2"),osp.join(real_dir,"3"),osp.join(real_dir,"4")]
    fake_paths = [osp.join(fake_dir, "1"), osp.join(fake_dir, "2"), osp.join(fake_dir, "3"), osp.join(fake_dir, "4")]

    rets = eval_epoch(real_paths,fake_paths,epoch, eval_dir)
    if epoch != -1:
        sw.add_scalar('eval/kid_mean', rets[0].kid_mean, int(epoch/10))
        sw.add_scalar('eval/fid', rets[0].fid, int(epoch / 10))
        sw.add_scalar('eval/kNN', rets[0].kNN, int(epoch / 10))
        sw.add_scalar('eval/K_MMD', rets[0].K_MMD, int(epoch / 10))
        sw.add_scalar('eval/WD', rets[0].WD, int(epoch / 10))
        sw.add_scalar('eval/_IS', rets[0]._IS, int(epoch/10))
        sw.add_scalar('eval/_MS', rets[0]._MS, int(epoch / 10))
        sw.add_scalar('eval/_mse_skimage', rets[0]._mse_skimage, int(epoch / 10))
        sw.add_scalar('eval/_ssim_skimage', rets[0]._ssim_skimage, int(epoch / 10))
        sw.add_scalar('eval/_ssimrgb_skimage', rets[0]._ssimrgb_skimage, int(epoch / 10))
        sw.add_scalar('eval/_psnr_skimage', rets[0]._psnr_skimage, int(epoch / 10))
        sw.add_scalar('eval/_kid_std', rets[0]._kid_std, int(epoch / 10))
        sw.add_scalar('eval/_fid_inkid_mean', rets[0]._fid_inkid_mean, int(epoch / 10))
        sw.add_scalar('eval/_fid_inkid_std', rets[0]._fid_inkid_std, int(epoch / 10))
    model_G[0].train()
    model_G[1].train()
    model_G[2].train()





def train(args, get_dataloader_func=get_pix2pix_maps_dataloader):
    with open(os.path.join(args.save,'args.json'), 'w') as f:
        json.dump(vars(args), f)
    logger = Logger(save_path=args.save, json_name='img2map_seg')
    epoch_now = len(logger.get_data('D1_loss'))

    model_saver = ModelSaver(save_path=args.save,
                             name_list=['G1', 'D1', 'G1_optimizer', 'D1_optimizer',
                                        'G1_scheduler', 'D1_scheduler',
                                        'G2', 'D2', 'G2_optimizer', 'D2_optimizer',
                                        'G2_scheduler', 'D2_scheduler',
                                        'G3', 'D3', 'G3_optimizer', 'D3_optimizer',
                                        'G3_scheduler', 'D3_scheduler',
                                        'G4', 'D4', 'G4_optimizer', 'D4_optimizer',
                                        'G4_scheduler', 'D4_scheduler',
                                        'G5', 'D5', 'G5_optimizer', 'D5_optimizer',
                                        'G5_scheduler', 'D5_scheduler',
                                        'G6', 'D6', 'G6_optimizer', 'D6_optimizer',
                                        'G6_scheduler', 'D6_scheduler'])

    sw = SummaryWriter(args.tensorboard_path)


    G1 = get_G(args,input_nc=6) # 3+256+1，256为分割网络输出featuremap的通道数
    D1 = get_D(args,input_nc=6)
    model_saver.load('G1', G1)
    model_saver.load('D1', D1)
    G2 = get_G(args,input_nc=6) # 3+256+1，256为分割网络输出featuremap的通道数
    D2 = get_D(args,input_nc=6)
    model_saver.load('G2', G2)
    model_saver.load('D2', D2)
    G3 = get_G(args,input_nc=6) # 3+256+1，256为分割网络输出featuremap的通道数
    D3 = get_D(args,input_nc=6)
    model_saver.load('G3', G3)
    model_saver.load('D3', D3)
    G4 = get_G(args,input_nc=6) # 3+256+1，256为分割网络输出featuremap的通道数
    D4 = get_D(args,input_nc=6)
    model_saver.load('G1', G1)
    model_saver.load('D1', D1)
    G5 = get_G(args,input_nc=6) # 3+256+1，256为分割网络输出featuremap的通道数
    D5 = get_D(args,input_nc=6)
    model_saver.load('G2', G2)
    model_saver.load('D2', D2)
    G6 = get_G(args,input_nc=6) # 3+256+1，256为分割网络输出featuremap的通道数
    D6 = get_D(args,input_nc=6)
    model_saver.load('G3', G3)
    model_saver.load('D3', D3)
    G_all = [G1,G2,G3,G4,G5,G6]
    D_all = [D1,D2,D3,D4,D5,D6]

    cfg=Configuration()
    cfg.MODEL_NUM_CLASSES=args.label_nc


    G1_optimizer = Adam(G1.parameters(), lr=args.G_lr, betas=(args.beta1, 0.999))
    D1_optimizer = Adam(D1.parameters(), lr=args.D_lr, betas=(args.beta1, 0.999))
    G2_optimizer = Adam(G2.parameters(), lr=args.G_lr, betas=(args.beta1, 0.999))
    D2_optimizer = Adam(D2.parameters(), lr=args.D_lr, betas=(args.beta1, 0.999))
    G3_optimizer = Adam(G3.parameters(), lr=args.G_lr, betas=(args.beta1, 0.999))
    D3_optimizer = Adam(D3.parameters(), lr=args.D_lr, betas=(args.beta1, 0.999))
    G4_optimizer = Adam(G4.parameters(), lr=args.G_lr, betas=(args.beta1, 0.999))
    D4_optimizer = Adam(D4.parameters(), lr=args.D_lr, betas=(args.beta1, 0.999))
    G5_optimizer = Adam(G5.parameters(), lr=args.G_lr, betas=(args.beta1, 0.999))
    D5_optimizer = Adam(D5.parameters(), lr=args.D_lr, betas=(args.beta1, 0.999))
    G6_optimizer = Adam(G6.parameters(), lr=args.G_lr, betas=(args.beta1, 0.999))
    D6_optimizer = Adam(D6.parameters(), lr=args.D_lr, betas=(args.beta1, 0.999))

    model_saver.load('G1_optimizer', G1_optimizer)
    model_saver.load('D1_optimizer', D1_optimizer)
    model_saver.load('G2_optimizer', G2_optimizer)
    model_saver.load('D2_optimizer', D2_optimizer)
    model_saver.load('G3_optimizer', G3_optimizer)
    model_saver.load('D3_optimizer', D3_optimizer)
    model_saver.load('G4_optimizer', G1_optimizer)
    model_saver.load('D4_optimizer', D1_optimizer)
    model_saver.load('G5_optimizer', G2_optimizer)
    model_saver.load('D5_optimizer', D2_optimizer)
    model_saver.load('G6_optimizer', G3_optimizer)
    model_saver.load('D6_optimizer', D3_optimizer)

    G_optimizer_all = [G1_optimizer,G2_optimizer,G3_optimizer,G4_optimizer,G5_optimizer,G6_optimizer]
    D_optimizer_all = [D1_optimizer,D2_optimizer,D3_optimizer,D4_optimizer,D5_optimizer,D6_optimizer]

    G1_scheduler = get_hinge_scheduler(args, G1_optimizer)
    D1_scheduler = get_hinge_scheduler(args, D1_optimizer)
    G2_scheduler = get_hinge_scheduler(args, G2_optimizer)
    D2_scheduler = get_hinge_scheduler(args, D2_optimizer)
    G3_scheduler = get_hinge_scheduler(args, G3_optimizer)
    D3_scheduler = get_hinge_scheduler(args, D3_optimizer)
    G4_scheduler = get_hinge_scheduler(args, G4_optimizer)
    D4_scheduler = get_hinge_scheduler(args, D4_optimizer)
    G5_scheduler = get_hinge_scheduler(args, G5_optimizer)
    D5_scheduler = get_hinge_scheduler(args, D5_optimizer)
    G6_scheduler = get_hinge_scheduler(args, G6_optimizer)
    D6_scheduler = get_hinge_scheduler(args, D6_optimizer)

    model_saver.load('G1_scheduler', G1_scheduler)
    model_saver.load('D1_scheduler', D1_scheduler)
    model_saver.load('G2_scheduler', G2_scheduler)
    model_saver.load('D2_scheduler', D2_scheduler)
    model_saver.load('G3_scheduler', G3_scheduler)
    model_saver.load('D3_scheduler', D3_scheduler)
    model_saver.load('G4_scheduler', G4_scheduler)
    model_saver.load('D4_scheduler', D4_scheduler)
    model_saver.load('G5_scheduler', G5_scheduler)
    model_saver.load('D5_scheduler', D5_scheduler)
    model_saver.load('G6_scheduler', G6_scheduler)
    model_saver.load('D6_scheduler', D6_scheduler)
    G_scheduler_all = [G1_scheduler,G2_scheduler,G3_scheduler,G4_scheduler,G5_scheduler,G6_scheduler]
    D_scheduler_all = [D1_scheduler,D2_scheduler,D3_scheduler,G4_scheduler,G5_scheduler,G6_scheduler]


    device = get_device(args)

    GANLoss1 = get_GANLoss(args)
    GANLoss2 = get_GANLoss(args)
    GANLoss3 = get_GANLoss(args)
    GANLoss4 = get_GANLoss(args)
    GANLoss5 = get_GANLoss(args)
    GANLoss6 = get_GANLoss(args)
    GANLoss_all = [GANLoss1,GANLoss2,GANLoss3,GANLoss4,GANLoss5,GANLoss6]
    if args.use_ganFeat_loss:
        DFLoss1 = get_DFLoss(args)
        DFLoss2 = get_DFLoss(args)
        DFLoss3 = get_DFLoss(args)
        DFLoss4 = get_DFLoss(args)
        DFLoss5 = get_DFLoss(args)
        DFLoss6 = get_DFLoss(args)
        DFLoss_all = [DFLoss1,DFLoss2,DFLoss3,DFLoss4,DFLoss5,DFLoss6]
    if args.use_vgg_loss:
        VGGLoss1 = get_VGGLoss(args)
        VGGLoss2 = get_VGGLoss(args)
        VGGLoss3 = get_VGGLoss(args)
        VGGLoss4 = get_VGGLoss(args)
        VGGLoss5 = get_VGGLoss(args)
        VGGLoss6 = get_VGGLoss(args)
        VGGLoss_all = [VGGLoss1,VGGLoss2,VGGLoss3,VGGLoss4,VGGLoss5,VGGLoss6]
    if args.use_low_level_loss:
        LLLoss1 = get_low_level_loss(args)
        LLLoss2 = get_low_level_loss(args)
        LLLoss3 = get_low_level_loss(args)
        LLLoss4 = get_low_level_loss(args)
        LLLoss5 = get_low_level_loss(args)
        LLLoss6 = get_low_level_loss(args)
        LLLoss_all = [LLLoss1,LLLoss2,LLLoss3,LLLoss4,LLLoss5,LLLoss6]



    if epoch_now==args.epochs or args.epochs==-1:
        print('get final models')
        eval_fidiou(args, sw=sw, model_G=G_all)

    total_steps=0
    for epoch in range(epoch_now, args.epochs):
        G_loss_list = [[],[],[],[],[],[]]
        D_loss_list = [[],[],[],[],[],[]]

        index_layer=0
        for layer in range(args.layer_num):
            # start from layer_num
            id_layer = args.layer_num - layer
            if layer == 0:
                before_img_path = os.path.join(args.dataroot, "train", "A", str(id_layer))
                after_img_path = os.path.join(args.dataroot, "train", "B", str(id_layer))
                if not os.path.exists(after_img_path):
                    os.mkdir(after_img_path)
                img_list = make_dataset(before_img_path)
                for img_before in tqdm(img_list):
                    img_name = os.path.split(img_before)[1]
                    img_after = os.path.join(after_img_path,img_name)
                    shutil.copy(img_before,img_after)
            else:
                index_layer=index_layer+1
                data_loader = get_inter1_dataloader(args, id_layer, train=True)
                data_loader = tqdm(data_loader)

                for step, sample in enumerate(data_loader):
                    total_steps=total_steps+1
                    layer_imgs = sample['A'].to(device)
                    final_before_img = sample['B'].to(device)
                    target_layer_img = sample['C'].to(device)

                    # print(id_layer.shape)
                    imgs_plus = torch.cat((layer_imgs.float(),final_before_img.float()),1)
                    # train the Discriminator
                    D_optimizer_all[index_layer-1].zero_grad()
                    reals_maps = torch.cat([layer_imgs.float(), target_layer_img.float(), ], dim=1)
                    # reals_maps = torch.cat([imgs.float(), maps.float()], dim=1)

                    fakes = G_all[index_layer-1](imgs_plus).detach()

                    fake_dir = os.path.join(args.dataroot, "train", "B", str(id_layer))
                    if not os.path.exists(fake_dir):
                        os.mkdir(fake_dir)
                    batch_size = layer_imgs.size(0)
                    im_name = sample['A_path']
                    for b in range(batch_size):
                        file_name = osp.split(im_name[b])[-1].split('.')[0]
                        fake_file = osp.join(fake_dir, f'{file_name}.png')
                        from_std_tensor_save_image(filename=fake_file, data=fakes[b].cpu())

                    fakes_maps = torch.cat([layer_imgs.float(), fakes.float(), ], dim=1)
                    # fakes_maps = torch.cat([imgs.float(), fakes.float()], dim=1)

                    D_real_outs = D_all[index_layer-1](reals_maps)
                    D_real_loss = GANLoss_all[index_layer-1](D_real_outs, True)

                    D_fake_outs = D_all[index_layer-1](fakes_maps)
                    D_fake_loss = GANLoss_all[index_layer-1](D_fake_outs, False)

                    D_loss = 0.5 * (D_real_loss + D_fake_loss)
                    D_loss = D_loss.mean()
                    D_loss.backward()
                    D_loss = D_loss.item()
                    D_optimizer_all[index_layer-1].step()

                    # train generator and encoder
                    # G_optimizer.zero_grad()
                    fakes = G_all[index_layer-1](imgs_plus)


                    fakes_maps = torch.cat([layer_imgs.float(), fakes.float()], dim=1)
                    # fakes_maps = torch.cat([imgs.float(), fakes.float()], dim=1)
                    D_fake_outs = D_all[index_layer-1](fakes_maps)

                    gan_loss = GANLoss_all[index_layer-1](D_fake_outs, True)

                    G_loss = 0
                    G_loss += gan_loss
                    gan_loss = gan_loss.mean().item()

                    if args.use_vgg_loss:
                        vgg_loss = VGGLoss_all[index_layer-1](fakes, layer_imgs)
                        G_loss += args.lambda_feat * vgg_loss
                        vgg_loss = vgg_loss.mean().item()
                    else:
                        vgg_loss = 0.

                    if args.use_ganFeat_loss:
                        df_loss = DFLoss_all[index_layer-1](D_fake_outs, D_real_outs)
                        G_loss += args.lambda_feat * df_loss
                        df_loss = df_loss.mean().item()
                    else:
                        df_loss = 0.

                    if args.use_low_level_loss:
                        ll_loss = LLLoss_all[index_layer-1](fakes, target_layer_img)
                        G_loss += args.lambda_feat * ll_loss
                        ll_loss = ll_loss.mean().item()
                    else:
                        ll_loss = 0.

                    G_loss = G_loss.mean()
                    G_optimizer_all[index_layer-1].zero_grad()
                    G_loss.backward()

                    G_optimizer_all[index_layer-1].step()

                    G_loss = G_loss.item()

                    data_loader.write(f'Epochs:{epoch}  | Dloss:{D_loss:.6f} | Gloss:{G_loss:.6f}'
                                      f'| GANloss:{gan_loss:.6f} | VGGloss:{vgg_loss:.6f} | DFloss:{df_loss:.6f} '
                                      f'| LLloss:{ll_loss:.6f} | lr_gan:{get_lr(G_optimizer_all[index_layer-1]):.8f}')

                    G_loss_list[index_layer-1].append(G_loss)
                    D_loss_list[index_layer-1].append(D_loss)


                    # tensorboard log
                    if args.tensorboard_log and step % args.tensorboard_log == 0:  # defalut is 5
                        # total_steps = epoch * len(data_loader) + step
                        # sw.add_scalar('Loss1/G', G_loss, total_steps)
                        sw.add_scalar('Loss/G'+str(index_layer-1), G_loss, total_steps)
                        sw.add_scalar('Loss/D'+str(index_layer-1), D_loss, total_steps)
                        sw.add_scalar('Loss/gan'+str(index_layer-1), gan_loss, total_steps)
                        sw.add_scalar('Loss/vgg'+str(index_layer-1), vgg_loss, total_steps)
                        sw.add_scalar('Loss/df'+str(index_layer-1), df_loss, total_steps)
                        sw.add_scalar('Loss/ll'+str(index_layer-1), ll_loss, total_steps)

                        sw.add_scalar('LR/G'+str(index_layer-1), get_lr(G_optimizer_all[index_layer-1]), total_steps)
                        sw.add_scalar('LR/D'+str(index_layer-1), get_lr(D_optimizer_all[index_layer-1]), total_steps)


                        sw.add_image('img2/A', tensor2im(layer_imgs.data), total_steps, dataformats='HWC')
                        sw.add_image('img2/B', tensor2im(final_before_img.data), total_steps, dataformats='HWC')


                        sw.add_image('img2/C', tensor2im(target_layer_img.data), total_steps, dataformats='HWC')
                        sw.add_image('img2/fake', tensor2im(fakes.data), total_steps, dataformats='HWC')
            D_scheduler_all[index_layer-1].step(epoch)
            G_scheduler_all[index_layer-1].step(epoch)
        #     反向传递
        for layer in range(args.layer_num):
            # start from layer_num
            id_layer = layer + 1
            if layer == 0:
                before_img_path = os.path.join(args.dataroot, "train", "B", str(id_layer))
                after_img_path = os.path.join(args.dataroot, "train", "B1", str(id_layer))
                if not os.path.exists(after_img_path):
                    os.mkdir(after_img_path)
                img_list = make_dataset(before_img_path)
                for img_before in tqdm(img_list):
                    img_name = os.path.split(img_before)[1]
                    img_after = os.path.join(after_img_path, img_name)
                    shutil.copy(img_before, img_after)
            else:
                index_layer = index_layer + 1
                data_loader1 = get_inter1_dataloader(args, id_layer, train=True, flag=1)
                data_loader1 = tqdm(data_loader1)

                for step, sample in enumerate(data_loader1):
                    total_steps = total_steps + 1
                    layer_imgs = sample['A'].to(device)
                    final_before_img = sample['B'].to(device)
                    target_layer_img = sample['C'].to(device)

                    # print(id_layer.shape)
                    imgs_plus = torch.cat((layer_imgs.float(), final_before_img.float()), 1)
                    # train the Discriminator

                    D_optimizer_all[index_layer-1].zero_grad()
                    reals_maps = torch.cat([layer_imgs.float(), target_layer_img.float(), ], dim=1)
                    # reals_maps = torch.cat([imgs.float(), maps.float()], dim=1)

                    fakes = G_all[index_layer-1](imgs_plus).detach()

                    fake_dir = os.path.join(args.dataroot, "train", "B1", str(id_layer))
                    if not os.path.exists(fake_dir):
                        os.mkdir(fake_dir)
                    batch_size = layer_imgs.size(0)
                    im_name = sample['A_path']
                    for b in range(batch_size):
                        file_name = osp.split(im_name[b])[-1].split('.')[0]
                        fake_file = osp.join(fake_dir, f'{file_name}.png')
                        from_std_tensor_save_image(filename=fake_file, data=fakes[b].cpu())

                    fakes_maps = torch.cat([layer_imgs.float(), fakes.float(), ], dim=1)
                    # fakes_maps = torch.cat([imgs.float(), fakes.float()], dim=1)

                    D_real_outs = D_all[index_layer-1](reals_maps)
                    D_real_loss = GANLoss_all[index_layer-1](D_real_outs, True)

                    D_fake_outs = D_all[index_layer-1](fakes_maps)
                    D_fake_loss = GANLoss_all[index_layer-1](D_fake_outs, False)

                    D_loss = 0.5 * (D_real_loss + D_fake_loss)
                    D_loss = D_loss.mean()
                    D_loss.backward()
                    D_loss = D_loss.item()
                    D_optimizer_all[index_layer-1].step()

                    # train generator and encoder
                    # G_optimizer.zero_grad()
                    fakes = G_all[index_layer-1](imgs_plus)

                    fakes_maps = torch.cat([layer_imgs.float(), fakes.float()], dim=1)
                    # fakes_maps = torch.cat([imgs.float(), fakes.float()], dim=1)
                    D_fake_outs = D_all[index_layer-1](fakes_maps)

                    gan_loss = GANLoss_all[index_layer-1](D_fake_outs, True)

                    G_loss = 0
                    G_loss += gan_loss
                    gan_loss = gan_loss.mean().item()

                    if args.use_vgg_loss:
                        vgg_loss = VGGLoss_all[index_layer-1](fakes, layer_imgs)
                        G_loss += args.lambda_feat * vgg_loss
                        vgg_loss = vgg_loss.mean().item()
                    else:
                        vgg_loss = 0.

                    if args.use_ganFeat_loss:
                        df_loss = DFLoss_all[index_layer-1](D_fake_outs, D_real_outs)
                        G_loss += args.lambda_feat * df_loss
                        df_loss = df_loss.mean().item()
                    else:
                        df_loss = 0.

                    if args.use_low_level_loss:
                        ll_loss = LLLoss_all[index_layer-1](fakes, target_layer_img)
                        G_loss += args.lambda_feat * ll_loss
                        ll_loss = ll_loss.mean().item()
                    else:
                        ll_loss = 0.

                    G_loss = G_loss.mean()
                    G_optimizer_all[index_layer-1].zero_grad()
                    G_loss.backward()

                    G_optimizer_all[index_layer-1].step()

                    G_loss = G_loss.item()

                    data_loader.write(f'Epochs:{epoch}  | Dloss:{D_loss:.6f} | Gloss:{G_loss:.6f}'
                                      f'| GANloss:{gan_loss:.6f} | VGGloss:{vgg_loss:.6f} | DFloss:{df_loss:.6f} '
                                      f'| LLloss:{ll_loss:.6f} | lr_gan:{get_lr(G_optimizer_all[index_layer-1]):.8f}')

                    G_loss_list[index_layer-1].append(G_loss)
                    D_loss_list[index_layer-1].append(D_loss)

                    # tensorboard log
                    if args.tensorboard_log and step % args.tensorboard_log == 0:  # defalut is 5
                        # total_steps = epoch * len(data_loader) + step
                        # sw.add_scalar('Loss1/G', G_loss, total_steps)
                        sw.add_scalar('Loss/G' + str(index_layer-1), G_loss, total_steps)
                        sw.add_scalar('Loss/D' + str(index_layer-1), D_loss, total_steps)
                        sw.add_scalar('Loss/gan' + str(index_layer-1), gan_loss, total_steps)
                        sw.add_scalar('Loss/vgg' + str(index_layer-1), vgg_loss, total_steps)
                        sw.add_scalar('Loss/df' + str(index_layer-1), df_loss, total_steps)
                        sw.add_scalar('Loss/ll' + str(index_layer-1), ll_loss, total_steps)

                        sw.add_scalar('LR/G' + str(index_layer-1), get_lr(G_optimizer_all[index_layer-1]), total_steps)
                        sw.add_scalar('LR/D' + str(index_layer-1), get_lr(D_optimizer_all[index_layer-1]), total_steps)

                        sw.add_image('img2/A', tensor2im(layer_imgs.data), total_steps, dataformats='HWC')
                        sw.add_image('img2/B', tensor2im(final_before_img.data), total_steps, dataformats='HWC')

                        sw.add_image('img2/C', tensor2im(target_layer_img.data), total_steps, dataformats='HWC')
                        sw.add_image('img2/fake', tensor2im(fakes.data), total_steps, dataformats='HWC')
            D_scheduler_all[index_layer-1].step(epoch)
            G_scheduler_all[index_layer-1].step(epoch)
        # D_scheduler.step(epoch)
        # G_scheduler.step(epoch)




        logger.log(key='D1_loss', data=sum(D_loss_list[0]) / float(len(D_loss_list[0])))
        logger.log(key='G1_loss', data=sum(G_loss_list[0]) / float(len(G_loss_list[0])))
        logger.log(key='D2_loss', data=sum(D_loss_list[1]) / float(len(D_loss_list[1])))
        logger.log(key='G2_loss', data=sum(G_loss_list[1]) / float(len(G_loss_list[1])))
        logger.log(key='D3_loss', data=sum(D_loss_list[2]) / float(len(D_loss_list[2])))
        logger.log(key='G3_loss', data=sum(G_loss_list[2]) / float(len(G_loss_list[2])))
        logger.log(key='D4_loss', data=sum(D_loss_list[3]) / float(len(D_loss_list[3])))
        logger.log(key='G4_loss', data=sum(G_loss_list[3]) / float(len(G_loss_list[3])))
        logger.log(key='D5_loss', data=sum(D_loss_list[4]) / float(len(D_loss_list[4])))
        logger.log(key='G5_loss', data=sum(G_loss_list[4]) / float(len(G_loss_list[4])))
        logger.log(key='D6_loss', data=sum(D_loss_list[5]) / float(len(D_loss_list[5])))
        logger.log(key='G6_loss', data=sum(G_loss_list[5]) / float(len(G_loss_list[5])))

        logger.save_log()
        # logger.visualize()

        model_saver.save('G1', G_all[0])
        model_saver.save('D1', D_all[0])
        model_saver.save('G2', G_all[1])
        model_saver.save('D2', D_all[1])
        model_saver.save('G3', G_all[2])
        model_saver.save('D3', D_all[2])
        model_saver.save('G4', G_all[3])
        model_saver.save('D4', D_all[3])
        model_saver.save('G5', G_all[4])
        model_saver.save('D5', D_all[4])
        model_saver.save('G6', G_all[5])
        model_saver.save('D6', D_all[5])
        # model_saver.save('DLV3P', DLV3P)

        model_saver.save('G1_optimizer', G_optimizer_all[0])
        model_saver.save('D1_optimizer', D_optimizer_all[0])
        model_saver.save('G2_optimizer', G_optimizer_all[1])
        model_saver.save('D2_optimizer', D_optimizer_all[1])
        model_saver.save('G3_optimizer', G_optimizer_all[2])
        model_saver.save('D3_optimizer', D_optimizer_all[2])
        model_saver.save('G4_optimizer', G_optimizer_all[3])
        model_saver.save('D4_optimizer', D_optimizer_all[3])
        model_saver.save('G5_optimizer', G_optimizer_all[4])
        model_saver.save('D5_optimizer', D_optimizer_all[4])
        model_saver.save('G6_optimizer', G_optimizer_all[5])
        model_saver.save('D6_optimizer', D_optimizer_all[5])


        model_saver.save('G1_scheduler', G_scheduler_all[0])
        model_saver.save('D1_scheduler', D_scheduler_all[0])
        model_saver.save('G2_scheduler', G_scheduler_all[1])
        model_saver.save('D2_scheduler', D_scheduler_all[1])
        model_saver.save('G3_scheduler', G_scheduler_all[2])
        model_saver.save('D3_scheduler', D_scheduler_all[2])
        model_saver.save('G4_scheduler', G_scheduler_all[3])
        model_saver.save('D4_scheduler', D_scheduler_all[3])
        model_saver.save('G5_scheduler', G_scheduler_all[4])
        model_saver.save('D5_scheduler', D_scheduler_all[4])
        model_saver.save('G6_scheduler', G_scheduler_all[5])
        model_saver.save('D6_scheduler', D_scheduler_all[5])



        if epoch == (args.epochs - 1) or (epoch % 10 == 0):
            import copy
            args2=copy.deepcopy(args)
            args2.batch_size=args.batch_size_eval
            eval_fidiou(args, sw=sw, model_G=G_all, epoch=epoch)


if __name__ == '__main__':
    args = config()

    # args.label_nc = 5

    from src.pix2pixHD.myutils import seed_torch
    print(f'\nset seed as {args.seed}!\n')
    seed_torch(args.seed)

    train(args, get_dataloader_func=get_pix2pix_maps_dataloader)

pass