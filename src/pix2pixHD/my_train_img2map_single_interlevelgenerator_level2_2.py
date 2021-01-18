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

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif',
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


def eval_fidiou(args, model_G):
    for layer in range(args.layer_num):
        # start from layer_num
        id_layer = args.layer_num - layer
        if layer == 0:
            before_img_path = os.path.join(args.dataroot, "test", "A", str(id_layer))
            after_img_path = os.path.join(args.dataroot, "test", "B", str(id_layer))
            if not os.path.exists(after_img_path):
                os.mkdir(after_img_path)
            img_list = make_dataset(before_img_path)
            for img_before in img_list:
                img_name = os.path.split(img_before)[1]
                img_after = os.path.join(after_img_path, img_name)
                shutil.copy(img_before, img_after)
        else:
            data_loader = get_inter1_dataloader(args, id_layer, train=False)
            data_loader = tqdm(data_loader)
            model_G.eval()
            device = get_device(args)
            model_G = model_G.to(device)

            fake_dir = osp.join(args.save, 'fake_result')
            test_B_dir = os.path.join(args.dataroot, "test", "B", str(id_layer))
            create_dir(fake_dir)
            create_dir(test_B_dir)

            for i, sample in enumerate(data_loader):
                layer_imgs = sample['A'].to(device)
                final_before_img = sample['B'].to(device)

                imgs_plus = torch.cat((layer_imgs.float(), final_before_img.float()), 1)
                fakes = model_G(imgs_plus).detach()

                batch_size = layer_imgs.size(0)
                im_name = sample['A_path']
                for b in range(batch_size):
                    file_name = osp.split(im_name[b])[-1].split('.')[0]
                    fake_file = osp.join(fake_dir, f'{file_name}.png')
                    test_B_file = osp.join(test_B_dir, f'{file_name}.png')
                    from_std_tensor_save_image(filename=fake_file, data=fakes[b].cpu())
                    from_std_tensor_save_image(filename=test_B_file, data=fakes[b].cpu())
    for layer in range(args.layer_num):
        # start from layer_num
        id_layer = layer + 1
        if layer == 0:
            before_img_path = os.path.join(args.dataroot, "test", "B", str(id_layer))
            after_img_path = os.path.join(args.dataroot, "test", "B1", str(id_layer))
            if not os.path.exists(after_img_path):
                os.mkdir(after_img_path)
            img_list = make_dataset(before_img_path)
            for img_before in tqdm(img_list):
                img_name = os.path.split(img_before)[1]
                img_after = os.path.join(after_img_path, img_name)
                shutil.copy(img_before, img_after)
        else:
            data_loader = get_inter1_dataloader(args, id_layer, train=False, flag=1)
            data_loader = tqdm(data_loader)
            # model_G.eval()
            # device = get_device(args)
            # model_G = model_G.to(device)

            fake_dir = osp.join(args.save, 'fake_result')
            test_B_dir = os.path.join(args.dataroot, "test", "B1", str(id_layer))
            create_dir(fake_dir)
            create_dir(test_B_dir)

            for i, sample in enumerate(data_loader):
                layer_imgs = sample['A'].to(device)
                final_before_img = sample['B'].to(device)

                imgs_plus = torch.cat((layer_imgs.float(), final_before_img.float()), 1)
                fakes = model_G(imgs_plus).detach()

                batch_size = layer_imgs.size(0)
                im_name = sample['A_path']
                for b in range(batch_size):
                    file_name = osp.split(im_name[b])[-1].split('.')[0]
                    fake_file = osp.join(fake_dir, f'{file_name}.png')
                    test_B_file = osp.join(test_B_dir, f'{file_name}.png')
                    from_std_tensor_save_image(filename=fake_file, data=fakes[b].cpu())
                    from_std_tensor_save_image(filename=test_B_file, data=fakes[b].cpu())
    model_G.train()


def train(args, get_dataloader_func=get_pix2pix_maps_dataloader):
    with open(os.path.join(args.save, 'args.json'), 'w') as f:
        json.dump(vars(args), f)
    logger = Logger(save_path=args.save, json_name='img2map_seg')
    epoch_now = len(logger.get_data('D_loss'))

    model_saver = ModelSaver(save_path=args.save,
                             name_list=['G', 'D', 'G_optimizer', 'D_optimizer',
                                        'G_scheduler', 'D_scheduler'])

    sw = SummaryWriter(args.tensorboard_path)

    G = get_G(args, input_nc=6)  # 3+256+1，256为分割网络输出featuremap的通道数
    D = get_D(args, input_nc=6)

    model_saver.load('G', G)
    model_saver.load('D', D)

    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = args.label_nc

    G_optimizer = Adam(G.parameters(), lr=args.G_lr, betas=(args.beta1, 0.999))
    D_optimizer = Adam(D.parameters(), lr=args.D_lr, betas=(args.beta1, 0.999))

    model_saver.load('G_optimizer', G_optimizer)
    model_saver.load('D_optimizer', D_optimizer)

    G_scheduler = get_hinge_scheduler(args, G_optimizer)
    D_scheduler = get_hinge_scheduler(args, D_optimizer)

    model_saver.load('G_scheduler', G_scheduler)
    model_saver.load('D_scheduler', D_scheduler)

    device = get_device(args)

    GANLoss = get_GANLoss(args)
    if args.use_ganFeat_loss:
        DFLoss = get_DFLoss(args)
    if args.use_vgg_loss:
        VGGLoss = get_VGGLoss(args)
    if args.use_low_level_loss:
        LLLoss = get_low_level_loss(args)


    if epoch_now == args.epochs or args.epochs == -1:
        print('get final models')
        eval_fidiou(args, model_G=G)

    total_steps = 0
    for epoch in range(epoch_now, args.epochs):
        G_loss_list = []
        D_loss_list = []


        # 前向传递

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
                    img_after = os.path.join(after_img_path, img_name)
                    shutil.copy(img_before, img_after)
            else:

                data_loader = get_inter1_dataloader(args, id_layer, train=True)
                data_loader = tqdm(data_loader)

                for step, sample in enumerate(data_loader):
                    total_steps = total_steps + 1
                    layer_imgs = sample['A'].to(device)
                    final_before_img = sample['B'].to(device)
                    target_layer_img = sample['C'].to(device)

                    # print(id_layer.shape)
                    imgs_plus = torch.cat((layer_imgs.float(), final_before_img.float()), 1)
                    # train the Discriminator
                    D_optimizer.zero_grad()
                    reals_maps = torch.cat([layer_imgs.float(), target_layer_img.float(), ], dim=1)
                    # reals_maps = torch.cat([imgs.float(), maps.float()], dim=1)

                    fakes = G(imgs_plus).detach()

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

                    D_real_outs = D(reals_maps)
                    D_real_loss = GANLoss(D_real_outs, True)

                    D_fake_outs = D(fakes_maps)
                    D_fake_loss = GANLoss(D_fake_outs, False)

                    D_loss = 0.5 * (D_real_loss + D_fake_loss)
                    D_loss = D_loss.mean()
                    D_loss.backward()
                    D_loss = D_loss.item()
                    D_optimizer.step()

                    # train generator and encoder
                    # G_optimizer.zero_grad()
                    fakes = G(imgs_plus)

                    fakes_maps = torch.cat([layer_imgs.float(), fakes.float()], dim=1)
                    # fakes_maps = torch.cat([imgs.float(), fakes.float()], dim=1)
                    D_fake_outs = D(fakes_maps)

                    gan_loss = GANLoss(D_fake_outs, True)

                    G_loss = 0
                    G_loss += gan_loss
                    gan_loss = gan_loss.mean().item()

                    if args.use_vgg_loss:
                        vgg_loss = VGGLoss(fakes, layer_imgs)
                        G_loss += args.lambda_feat * vgg_loss
                        vgg_loss = vgg_loss.mean().item()
                    else:
                        vgg_loss = 0.

                    if args.use_ganFeat_loss:
                        df_loss = DFLoss(D_fake_outs, D_real_outs)
                        G_loss += args.lambda_feat * df_loss
                        df_loss = df_loss.mean().item()
                    else:
                        df_loss = 0.

                    if args.use_low_level_loss:
                        ll_loss = LLLoss(fakes, target_layer_img)
                        G_loss += args.lambda_feat * ll_loss
                        ll_loss = ll_loss.mean().item()
                    else:
                        ll_loss = 0.

                    G_loss = G_loss.mean()
                    G_optimizer.zero_grad()
                    G_loss.backward()

                    G_optimizer.step()

                    G_loss = G_loss.item()

                    data_loader.write(f'Epochs:{epoch}  | Dloss:{D_loss:.6f} | Gloss:{G_loss:.6f}'
                                      f'| GANloss:{gan_loss:.6f} | VGGloss:{vgg_loss:.6f} | DFloss:{df_loss:.6f} '
                                      f'| LLloss:{ll_loss:.6f} | lr_gan:{get_lr(G_optimizer):.8f}')

                    G_loss_list.append(G_loss)
                    D_loss_list.append(D_loss)

                    # tensorboard log
                    if args.tensorboard_log and step % args.tensorboard_log == 0:  # defalut is 5
                        # total_steps = epoch * len(data_loader) + step
                        # sw.add_scalar('Loss1/G', G_loss, total_steps)
                        sw.add_scalar('Loss/G', G_loss, total_steps)
                        sw.add_scalar('Loss/D', D_loss, total_steps)
                        sw.add_scalar('Loss/gan', gan_loss, total_steps)
                        sw.add_scalar('Loss/vgg', vgg_loss, total_steps)
                        sw.add_scalar('Loss/df', df_loss, total_steps)
                        sw.add_scalar('Loss/ll', ll_loss, total_steps)

                        sw.add_scalar('LR/G', get_lr(G_optimizer), total_steps)
                        sw.add_scalar('LR/D', get_lr(D_optimizer), total_steps)

                        sw.add_image('img2/A', tensor2im(layer_imgs.data), total_steps, dataformats='HWC')
                        sw.add_image('img2/B', tensor2im(final_before_img.data), total_steps, dataformats='HWC')

                        sw.add_image('img2/C', tensor2im(target_layer_img.data), total_steps, dataformats='HWC')
                        sw.add_image('img2/fake', tensor2im(fakes.data), total_steps, dataformats='HWC')
            # 反向传递

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
                    D_optimizer.zero_grad()
                    reals_maps = torch.cat([layer_imgs.float(), target_layer_img.float(), ], dim=1)
                    # reals_maps = torch.cat([imgs.float(), maps.float()], dim=1)

                    fakes = G(imgs_plus).detach()

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

                    D_real_outs = D(reals_maps)
                    D_real_loss = GANLoss(D_real_outs, True)

                    D_fake_outs = D(fakes_maps)
                    D_fake_loss = GANLoss(D_fake_outs, False)

                    D_loss = 0.5 * (D_real_loss + D_fake_loss)
                    D_loss = D_loss.mean()
                    D_loss.backward()
                    D_loss = D_loss.item()
                    D_optimizer.step()

                    # train generator and encoder
                    # G_optimizer.zero_grad()
                    fakes = G(imgs_plus)

                    fakes_maps = torch.cat([layer_imgs.float(), fakes.float()], dim=1)
                    # fakes_maps = torch.cat([imgs.float(), fakes.float()], dim=1)
                    D_fake_outs = D(fakes_maps)

                    gan_loss = GANLoss(D_fake_outs, True)

                    G_loss = 0
                    G_loss += gan_loss
                    gan_loss = gan_loss.mean().item()

                    if args.use_vgg_loss:
                        vgg_loss = VGGLoss(fakes, layer_imgs)
                        G_loss += args.lambda_feat * vgg_loss
                        vgg_loss = vgg_loss.mean().item()
                    else:
                        vgg_loss = 0.

                    if args.use_ganFeat_loss:
                        df_loss = DFLoss(D_fake_outs, D_real_outs)
                        G_loss += args.lambda_feat * df_loss
                        df_loss = df_loss.mean().item()
                    else:
                        df_loss = 0.

                    if args.use_low_level_loss:
                        ll_loss = LLLoss(fakes, target_layer_img)
                        G_loss += args.lambda_feat * ll_loss
                        ll_loss = ll_loss.mean().item()
                    else:
                        ll_loss = 0.

                    G_loss = G_loss.mean()
                    G_optimizer.zero_grad()
                    G_loss.backward()

                    G_optimizer.step()

                    G_loss = G_loss.item()

                    data_loader1.write(f'Epochs:{epoch}  | Dloss:{D_loss:.6f} | Gloss:{G_loss:.6f}'
                                      f'| GANloss:{gan_loss:.6f} | VGGloss:{vgg_loss:.6f} | DFloss:{df_loss:.6f} '
                                      f'| LLloss:{ll_loss:.6f} | lr_gan:{get_lr(G_optimizer):.8f}')

                    G_loss_list.append(G_loss)
                    D_loss_list.append(D_loss)

                    # tensorboard log
                    if args.tensorboard_log and step % args.tensorboard_log == 0:  # defalut is 5
                        # total_steps = epoch * len(data_loader) + step
                        # sw.add_scalar('Loss1/G', G_loss, total_steps)
                        sw.add_scalar('Loss/G', G_loss, total_steps)
                        sw.add_scalar('Loss/D', D_loss, total_steps)
                        sw.add_scalar('Loss/gan', gan_loss, total_steps)
                        sw.add_scalar('Loss/vgg', vgg_loss, total_steps)
                        sw.add_scalar('Loss/df', df_loss, total_steps)
                        sw.add_scalar('Loss/ll', ll_loss, total_steps)

                        sw.add_scalar('LR/G', get_lr(G_optimizer), total_steps)
                        sw.add_scalar('LR/D', get_lr(D_optimizer), total_steps)

                        sw.add_image('img2/A', tensor2im(layer_imgs.data), total_steps, dataformats='HWC')
                        sw.add_image('img2/B', tensor2im(final_before_img.data), total_steps, dataformats='HWC')

                        sw.add_image('img2/C', tensor2im(target_layer_img.data), total_steps, dataformats='HWC')
                        sw.add_image('img2/fake2', tensor2im(fakes.data), total_steps, dataformats='HWC')

        D_scheduler.step(epoch)
        G_scheduler.step(epoch)


        logger.log(key='D_loss', data=sum(D_loss_list) / float(len(D_loss_list)))
        logger.log(key='G_loss', data=sum(G_loss_list) / float(len(G_loss_list)))

        logger.save_log()
        # logger.visualize()

        model_saver.save('G', G)
        model_saver.save('D', D)

        # model_saver.save('DLV3P', DLV3P)

        model_saver.save('G_optimizer', G_optimizer)
        model_saver.save('D_optimizer', D_optimizer)

        model_saver.save('G_scheduler', G_scheduler)
        model_saver.save('D_scheduler', D_scheduler)


        if epoch == (args.epochs - 1):
            import copy
            args2 = copy.deepcopy(args)
            args2.batch_size = args.batch_size_eval
            eval_fidiou(args, model_G=G)


if __name__ == '__main__':
    args = config()

    # args.label_nc = 5

    from src.pix2pixHD.myutils import seed_torch

    print(f'\nset seed as {args.seed}!\n')
    seed_torch(args.seed)

    train(args, get_dataloader_func=get_pix2pix_maps_dataloader)

pass
