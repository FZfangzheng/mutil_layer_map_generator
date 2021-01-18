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
    # print(len(images))
    return images


def eval_fidiou(args, sw, model_G, model_G1,epoch=-1):
    fake_dir = osp.join(args.save, 'fake_result')
    real_dir = osp.join(args.save, 'real_result')
    eval_dir = osp.join(args.save, 'eval_result')
    create_dir(fake_dir)
    create_dir(real_dir)
    create_dir(eval_dir)
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
            data_loader = get_inter1_dataloader(args, id_layer, train=False)
            data_loader = tqdm(data_loader)
            model_G.eval()
            device = get_device(args)
            model_G = model_G.to(device)

            real_img_dir = os.path.join(real_dir, str(id_layer))
            test_B_dir = os.path.join(args.dataroot, "test", "B", str(id_layer))
            create_dir(test_B_dir)
            create_dir(real_img_dir)

            for i, sample in enumerate(data_loader):
                layer_imgs = sample['A'].to(device)
                final_before_img = sample['B'].to(device)
                gt_imgs = sample['C'].to(device)

                imgs_plus = torch.cat((layer_imgs.float(), final_before_img.float()), 1)
                fakes = model_G(imgs_plus).detach()

                batch_size = layer_imgs.size(0)
                im_name = sample['A_path']
                for b in range(batch_size):
                    file_name = osp.split(im_name[b])[-1].split('.')[0]
                    real_file = osp.join(real_img_dir, f'{file_name}.png')
                    test_B_file = osp.join(test_B_dir, f'{file_name}.png')
                    from_std_tensor_save_image(filename=test_B_file, data=fakes[b].cpu())
                    from_std_tensor_save_image(filename=real_file, data=gt_imgs[b].cpu())
    for layer in range(args.layer_num):
        # start from layer_num
        id_layer = layer + 1
        if layer == 0:
            fake_dir = osp.join(args.save, 'fake_result', str(id_layer))
            before_img_path = os.path.join(args.dataroot, "test", "B", str(id_layer))
            after_img_path = os.path.join(args.dataroot, "test", "B1", str(id_layer))
            create_dir(fake_dir)
            create_dir(after_img_path)
            img_list = make_dataset(before_img_path)
            for img_before in tqdm(img_list):
                img_name = os.path.split(img_before)[1]
                img_after = os.path.join(after_img_path, img_name)
                img_fake = os.path.join(fake_dir, img_name)
                shutil.copy(img_before, img_after)
                shutil.copy(img_before, img_fake)
        else:
            data_loader = get_inter1_dataloader(args, id_layer, train=False, flag=1)
            data_loader = tqdm(data_loader)
            model_G1.eval()
            device = get_device(args)
            model_G1 = model_G1.to(device)

            fake_dir = osp.join(args.save, 'fake_result', str(id_layer))
            test_B_dir = os.path.join(args.dataroot, "test", "B1", str(id_layer))
            create_dir(fake_dir)
            create_dir(test_B_dir)

            for i, sample in enumerate(data_loader):
                layer_imgs = sample['A'].to(device)
                final_before_img = sample['B'].to(device)

                imgs_plus = torch.cat((layer_imgs.float(), final_before_img.float()), 1)
                fakes = model_G1(imgs_plus).detach()

                batch_size = layer_imgs.size(0)
                im_name = sample['A_path']
                for b in range(batch_size):
                    file_name = osp.split(im_name[b])[-1].split('.')[0]
                    fake_file = osp.join(fake_dir, f'{file_name}.png')
                    test_B_file = osp.join(test_B_dir, f'{file_name}.png')
                    from_std_tensor_save_image(filename=fake_file, data=fakes[b].cpu())
                    from_std_tensor_save_image(filename=test_B_file, data=fakes[b].cpu())
    real_img_list = make_dataset(real_dir)
    fake_img_list = make_dataset(fake_dir)
    assert len(real_img_list)==len(fake_img_list)
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
    model_G.train()
    model_G1.train()


def train(args, get_dataloader_func=get_pix2pix_maps_dataloader):
    with open(os.path.join(args.save, 'args.json'), 'w') as f:
        json.dump(vars(args), f)
    logger = Logger(save_path=args.save, json_name='img2map_seg')
    epoch_now = len(logger.get_data('D_loss'))

    model_saver = ModelSaver(save_path=args.save,
                             name_list=['G', 'D', 'G_optimizer', 'D_optimizer',
                                        'G_scheduler', 'D_scheduler',
                                        'G1', 'D1', 'G1_optimizer', 'D1_optimizer',
                                        'G1_scheduler', 'D1_scheduler'])

    sw = SummaryWriter(args.tensorboard_path)

    G = get_G(args, input_nc=6)  # 3+256+1，256为分割网络输出featuremap的通道数
    D = get_D(args, input_nc=6)
    G1 = get_G(args, input_nc=6)  # 3+256+1，256为分割网络输出featuremap的通道数
    D1 = get_D(args, input_nc=6)

    model_saver.load('G', G)
    model_saver.load('D', D)
    model_saver.load('G1', G1)
    model_saver.load('D1', D1)

    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = args.label_nc

    G_optimizer = Adam(G.parameters(), lr=args.G_lr, betas=(args.beta1, 0.999))
    D_optimizer = Adam(D.parameters(), lr=args.D_lr, betas=(args.beta1, 0.999))
    G1_optimizer = Adam(G1.parameters(), lr=args.G_lr, betas=(args.beta1, 0.999))
    D1_optimizer = Adam(D1.parameters(), lr=args.D_lr, betas=(args.beta1, 0.999))

    model_saver.load('G_optimizer', G_optimizer)
    model_saver.load('D_optimizer', D_optimizer)
    model_saver.load('G1_optimizer', G1_optimizer)
    model_saver.load('D1_optimizer', D1_optimizer)

    G_scheduler = get_hinge_scheduler(args, G_optimizer)
    D_scheduler = get_hinge_scheduler(args, D_optimizer)
    G1_scheduler = get_hinge_scheduler(args, G1_optimizer)
    D1_scheduler = get_hinge_scheduler(args, D1_optimizer)

    model_saver.load('G_scheduler', G_scheduler)
    model_saver.load('D_scheduler', D_scheduler)
    model_saver.load('G1_scheduler', G1_scheduler)
    model_saver.load('D1_scheduler', D1_scheduler)

    device = get_device(args)

    GANLoss = get_GANLoss(args)
    if args.use_ganFeat_loss:
        DFLoss = get_DFLoss(args)
    if args.use_vgg_loss:
        VGGLoss = get_VGGLoss(args)
    if args.use_low_level_loss:
        LLLoss = get_low_level_loss(args)

    GAN1Loss = get_GANLoss(args)
    if args.use_ganFeat_loss:
        DF1Loss = get_DFLoss(args)
    if args.use_vgg_loss:
        VGG1Loss = get_VGGLoss(args)
    if args.use_low_level_loss:
        LL1Loss = get_low_level_loss(args)

    if epoch_now == args.epochs or args.epochs == -1:
        print('get final models')
        eval_fidiou(args, sw=sw, model_G=G, model_G1=G1)

    total_steps = 0
    for epoch in range(epoch_now, args.epochs):
        G_loss_list = []
        D_loss_list = []
        G1_loss_list = []
        D1_loss_list = []

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
                    # G_loss.backward(retain_graph=True)
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

                        sw.add_image('img1/A', tensor2im(layer_imgs.data), total_steps, dataformats='HWC')
                        sw.add_image('img1/B', tensor2im(final_before_img.data), total_steps, dataformats='HWC')

                        sw.add_image('img1/C', tensor2im(target_layer_img.data), total_steps, dataformats='HWC')
                        sw.add_image('img1/fake', tensor2im(fakes.data), total_steps, dataformats='HWC')

        D_scheduler.step(epoch)
        G_scheduler.step(epoch)
        logger.log(key='D_loss', data=sum(D_loss_list) / float(len(D_loss_list)))
        logger.log(key='G_loss', data=sum(G_loss_list) / float(len(G_loss_list)))
        model_saver.save('G', G)
        model_saver.save('D', D)
        model_saver.save('D_optimizer', D_optimizer)
        model_saver.save('G_scheduler', G_scheduler)
        model_saver.save('D_scheduler', D_scheduler)
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
                    layer_imgs1 = sample['A'].to(device)
                    final_before_img1 = sample['B'].to(device)
                    target_layer_img1 = sample['C'].to(device)

                    # print(id_layer.shape)
                    imgs_plus1 = torch.cat((layer_imgs1.float(), final_before_img1.float()), 1)
                    # train the Discriminator
                    D1_optimizer.zero_grad()
                    reals_maps1 = torch.cat([layer_imgs1.float(), target_layer_img1.float(), ], dim=1)
                    # reals_maps = torch.cat([imgs.float(), maps.float()], dim=1)

                    fakes1 = G1(imgs_plus1).detach()

                    fake_dir = os.path.join(args.dataroot, "train", "B1", str(id_layer))
                    if not os.path.exists(fake_dir):
                        os.mkdir(fake_dir)
                    batch_size = layer_imgs1.size(0)
                    im_name = sample['A_path']
                    for b in range(batch_size):
                        file_name = osp.split(im_name[b])[-1].split('.')[0]
                        fake_file = osp.join(fake_dir, f'{file_name}.png')
                        from_std_tensor_save_image(filename=fake_file, data=fakes1[b].cpu())



                    fakes_maps1 = torch.cat([layer_imgs1.float(), fakes1.float(), ], dim=1)
                    # fakes_maps = torch.cat([imgs.float(), fakes.float()], dim=1)

                    D1_real_outs = D1(reals_maps1)
                    D1_real_loss = GAN1Loss(D1_real_outs, True)

                    D1_fake_outs = D1(fakes_maps1)
                    D1_fake_loss = GAN1Loss(D1_fake_outs, False)

                    D1_loss = 0.5 * (D1_real_loss + D1_fake_loss)
                    D1_loss = D1_loss.mean()
                    D1_loss.backward()
                    D1_loss = D1_loss.item()
                    D1_optimizer.step()

                    # train generator and encoder
                    # G_optimizer.zero_grad()
                    fakes1 = G1(imgs_plus1)

                    fakes_maps1 = torch.cat([layer_imgs1.float(), fakes1.float()], dim=1)
                    # fakes_maps = torch.cat([imgs.float(), fakes.float()], dim=1)
                    D1_fake_outs = D1(fakes_maps1)

                    gan1_loss = GAN1Loss(D1_fake_outs, True)

                    G1_loss = 0
                    G1_loss += gan1_loss
                    gan1_loss = gan1_loss.mean().item()

                    if args.use_vgg_loss:
                        vgg1_loss = VGG1Loss(fakes1, layer_imgs1)
                        G1_loss += args.lambda_feat * vgg1_loss
                        vgg1_loss = vgg1_loss.mean().item()
                    else:
                        vgg1_loss = 0.

                    if args.use_ganFeat_loss:
                        df1_loss = DF1Loss(D1_fake_outs, D1_real_outs)
                        G1_loss += args.lambda_feat * df1_loss
                        df1_loss = df1_loss.mean().item()
                    else:
                        df1_loss = 0.

                    if args.use_low_level_loss:
                        ll1_loss = LL1Loss(fakes1, target_layer_img1)
                        G1_loss += args.lambda_feat * ll1_loss
                        ll1_loss = ll1_loss.mean().item()
                    else:
                        ll1_loss = 0.

                    G1_loss = G1_loss.mean()
                    G1_optimizer.zero_grad()
                    G1_loss.backward()

                    G1_optimizer.step()

                    G1_loss = G1_loss.item()

                    data_loader1.write(f'Epochs:{epoch}  | Dloss:{D1_loss:.6f} | Gloss:{G1_loss:.6f}'
                                      f'| GANloss:{gan1_loss:.6f} | VGGloss:{vgg1_loss:.6f} | DFloss:{df1_loss:.6f} '
                                      f'| LLloss:{ll1_loss:.6f} | lr_gan:{get_lr(G1_optimizer):.8f}')

                    G1_loss_list.append(G1_loss)
                    D1_loss_list.append(D1_loss)

                    # tensorboard log
                    if args.tensorboard_log and step % args.tensorboard_log == 0:  # defalut is 5
                        # total_steps = epoch * len(data_loader) + step
                        # sw.add_scalar('Loss1/G', G_loss, total_steps)
                        sw.add_scalar('Loss/G1', G1_loss, total_steps)
                        sw.add_scalar('Loss/D1', D1_loss, total_steps)
                        sw.add_scalar('Loss/gan1', gan1_loss, total_steps)
                        sw.add_scalar('Loss/vgg1', vgg1_loss, total_steps)
                        sw.add_scalar('Loss/df1', df1_loss, total_steps)
                        sw.add_scalar('Loss/ll1', ll1_loss, total_steps)

                        sw.add_scalar('LR/G1', get_lr(G1_optimizer), total_steps)
                        sw.add_scalar('LR/D1', get_lr(D1_optimizer), total_steps)
                        sw.add_image('img2/A', tensor2im(layer_imgs.data), total_steps, dataformats='HWC')
                        sw.add_image('img2/B', tensor2im(final_before_img.data), total_steps, dataformats='HWC')
                        sw.add_image('img2/fake2', tensor2im(fakes.data), total_steps, dataformats='HWC')
                        sw.add_image('img2/C', tensor2im(target_layer_img.data), total_steps, dataformats='HWC')


        D1_scheduler.step(epoch)
        G1_scheduler.step(epoch)


        logger.log(key='D1_loss', data=sum(D1_loss_list) / float(len(D1_loss_list)))
        logger.log(key='G1_loss', data=sum(G1_loss_list) / float(len(G1_loss_list)))

        logger.save_log()
        # logger.visualize()


        model_saver.save('G1', G1)
        model_saver.save('D1', D1)
        # model_saver.save('DLV3P', DLV3P)


        model_saver.save('G1_optimizer', G1_optimizer)
        model_saver.save('D1_optimizer', D1_optimizer)
        model_saver.save('G_optimizer', G_optimizer)

        model_saver.save('G1_scheduler', G1_scheduler)
        model_saver.save('D1_scheduler', D1_scheduler)

        if epoch == (args.epochs - 1) or (epoch % 10 == 0):
            import copy
            args2 = copy.deepcopy(args)
            args2.batch_size = args.batch_size_eval
            eval_fidiou(args, sw=sw, model_G=G, model_G1=G1,epoch=epoch)


if __name__ == '__main__':
    args = config()

    # args.label_nc = 5

    from src.pix2pixHD.myutils import seed_torch

    print(f'\nset seed as {args.seed}!\n')
    seed_torch(args.seed)

    train(args, get_dataloader_func=get_pix2pix_maps_dataloader)

pass
