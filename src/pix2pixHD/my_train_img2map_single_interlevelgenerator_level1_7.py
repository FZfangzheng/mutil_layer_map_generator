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
from src.pytorch_ssim import SSIM

from src.pix2pixHD.deeplabv3plus.lovasz_losses import lovasz_softmax

from util.util import tensor2im  # 注意，该函数使用0.5与255恢复可视图像，所以如果是ImageNet标准化的可能会有色差？这里都显示试一下
from src.pix2pixHD.myutils import pred2gray, gray2rgb

from src.pix2pixHD.deeplabv3plus.focal_loss import FocalLoss
from evaluation.fid.fid_score import fid_score
import json
from src.eval.eval_every10epoch import eval_epoch
import cv2

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

def eval_fidiou(args, sw, model_G, model_seg, epoch=-1):
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
            fake_img_path = os.path.join(fake_dir, str(id_layer))
            before_img_path = os.path.join(args.dataroot, "test", "A", str(id_layer))
            after_img_path = os.path.join(args.dataroot, "test", "B", str(id_layer))
            gt_img_path = os.path.join(args.dataroot, "test", "C", str(id_layer))
            create_dir(after_img_path)
            create_dir(real_img_path)
            create_dir(fake_img_path)
            img_list = make_dataset(before_img_path)
            for img_before in img_list:
                img_name = os.path.split(img_before)[1]
                img_after = os.path.join(after_img_path, img_name)
                img_real = os.path.join(real_img_path, img_name)
                img_fake = os.path.join(fake_img_path, img_name)
                img_gt = os.path.join(gt_img_path, img_name)
                shutil.copy(img_before, img_after)
                shutil.copy(img_before, img_fake)
                shutil.copy(img_gt, img_real)
        else:
            data_loader = get_inter1_dataloader(args, id_layer, train=False)
            data_loader = tqdm(data_loader)
            model_G.eval()
            model_seg.eval()
            device = get_device(args)
            model_G = model_G.to(device)
            model_seg = model_seg.to(device)

            fake_img_dir = osp.join(fake_dir, str(id_layer))
            real_img_dir = os.path.join(real_dir, str(id_layer))
            test_B_dir = os.path.join(args.dataroot, "test", "B", str(id_layer))
            create_dir(fake_img_dir)
            create_dir(real_img_dir)
            create_dir(test_B_dir)

            for i, sample in enumerate(data_loader):
                layer_imgs = sample['A'].to(device)
                final_before_img = sample['B'].to(device)
                gt_imgs = sample['C'].to(device)
                imgs_seg = sample['RS'].to(device)  # (shape: (batch_size, 3, img_h, img_w))

                imgs_seg = imgs_seg.cuda() if args.gpu else imgs_seg
                outputs, feature_map = model_seg(imgs_seg)
                soft_outputs = torch.nn.functional.softmax(outputs, dim=1)
                boundary_aware = []
                im_path = sample['A_path']

                transform_list = [transforms.ToTensor()]
                tf_boundary = transforms.Compose(transform_list)

                for i in range(len(im_path)):
                    # 结构是CxHxW，C是类别，对应类别下点为1
                    np_soft_outputs = soft_outputs[i].cpu().detach().numpy()

                    # 构造和原始label相似的结构
                    np_new_outputs = np.zeros((args.fineSize, args.fineSize))
                    for j in range(len(np_soft_outputs)):
                        if j == 0:
                            continue
                        else:
                            np_new_outputs = np_new_outputs + 2 * np_soft_outputs[j]
                    np_new_outputs = abs(np_new_outputs)
                    np_outputs = np_new_outputs.astype(np.uint8)

                    cv_canny = cv2.Canny(np_outputs, 1, 1)
                    cv_canny_tf = tf_boundary(cv_canny)
                    cv_canny_tf = cv_canny_tf.reshape(1, 1, cv_canny_tf.size(1), cv_canny_tf.size(2))
                    # print(cv_canny_tf)
                    boundary_aware.append(cv_canny_tf)
                boundary_aware_tf = boundary_aware[0]
                # print(len(im_path))
                for i in range(len(im_path)):
                    if i == 0:
                        continue
                    else:
                        boundary_aware_tf = torch.cat([boundary_aware_tf, boundary_aware[i]])
                boundary_aware_tf = boundary_aware_tf.to(device)


                imgs_plus = torch.cat((layer_imgs.float(),final_before_img.float(), boundary_aware_tf.float()),1)
                fakes = model_G(imgs_plus).detach()

                batch_size = layer_imgs.size(0)
                im_name = sample['A_path']
                for b in range(batch_size):
                    file_name = osp.split(im_name[b])[-1].split('.')[0]
                    real_file = osp.join(real_img_dir, f'{file_name}.png')
                    fake_file = osp.join(fake_img_dir, f'{file_name}.png')
                    test_B_file = osp.join(test_B_dir, f'{file_name}.png')
                    from_std_tensor_save_image(filename=fake_file, data=fakes[b].cpu())
                    from_std_tensor_save_image(filename=test_B_file, data=fakes[b].cpu())
                    from_std_tensor_save_image(filename=real_file, data=gt_imgs[b].cpu())

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
        sw.add_scalar('eval1/kid_mean', rets[0].kid_mean, int(epoch/10))
        sw.add_scalar('eval1/fid', rets[0].fid, int(epoch / 10))
        sw.add_scalar('eval1/kNN', rets[0].kNN, int(epoch / 10))
        sw.add_scalar('eval1/K_MMD', rets[0].K_MMD, int(epoch / 10))
        sw.add_scalar('eval1/WD', rets[0].WD, int(epoch / 10))
        sw.add_scalar('eval1/_IS', rets[0]._IS, int(epoch/10))
        sw.add_scalar('eval1/_MS', rets[0]._MS, int(epoch / 10))
        sw.add_scalar('eval1/_mse_skimage', rets[0]._mse_skimage, int(epoch / 10))
        sw.add_scalar('eval1/_ssim_skimage', rets[0]._ssim_skimage, int(epoch / 10))
        sw.add_scalar('eval1/_ssimrgb_skimage', rets[0]._ssimrgb_skimage, int(epoch / 10))
        sw.add_scalar('eval1/_psnr_skimage', rets[0]._psnr_skimage, int(epoch / 10))
        sw.add_scalar('eval1/_kid_std', rets[0]._kid_std, int(epoch / 10))
        sw.add_scalar('eval1/_fid_inkid_mean', rets[0]._fid_inkid_mean, int(epoch / 10))
        sw.add_scalar('eval1/_fid_inkid_std', rets[0]._fid_inkid_std, int(epoch / 10))
        sw.add_scalar('eval2/kid_mean', rets[1].kid_mean, int(epoch/10))
        sw.add_scalar('eval2/fid', rets[1].fid, int(epoch / 10))
        sw.add_scalar('eval2/kNN', rets[1].kNN, int(epoch / 10))
        sw.add_scalar('eval2/K_MMD', rets[1].K_MMD, int(epoch / 10))
        sw.add_scalar('eval2/WD', rets[1].WD, int(epoch / 10))
        sw.add_scalar('eval2/_IS', rets[1]._IS, int(epoch/10))
        sw.add_scalar('eval2/_MS', rets[1]._MS, int(epoch / 10))
        sw.add_scalar('eval2/_mse_skimage', rets[1]._mse_skimage, int(epoch / 10))
        sw.add_scalar('eval2/_ssim_skimage', rets[1]._ssim_skimage, int(epoch / 10))
        sw.add_scalar('eval2/_ssimrgb_skimage', rets[1]._ssimrgb_skimage, int(epoch / 10))
        sw.add_scalar('eval2/_psnr_skimage', rets[1]._psnr_skimage, int(epoch / 10))
        sw.add_scalar('eval2/_kid_std', rets[1]._kid_std, int(epoch / 10))
        sw.add_scalar('eval2/_fid_inkid_mean', rets[1]._fid_inkid_mean, int(epoch / 10))
        sw.add_scalar('eval2/_fid_inkid_std', rets[1]._fid_inkid_std, int(epoch / 10))
        sw.add_scalar('eval3/kid_mean', rets[2].kid_mean, int(epoch/10))
        sw.add_scalar('eval3/fid', rets[2].fid, int(epoch / 10))
        sw.add_scalar('eval3/kNN', rets[2].kNN, int(epoch / 10))
        sw.add_scalar('eval3/K_MMD', rets[2].K_MMD, int(epoch / 10))
        sw.add_scalar('eval3/WD', rets[2].WD, int(epoch / 10))
        sw.add_scalar('eval3/_IS', rets[2]._IS, int(epoch/10))
        sw.add_scalar('eval3/_MS', rets[2]._MS, int(epoch / 10))
        sw.add_scalar('eval3/_mse_skimage', rets[2]._mse_skimage, int(epoch / 10))
        sw.add_scalar('eval3/_ssim_skimage', rets[2]._ssim_skimage, int(epoch / 10))
        sw.add_scalar('eval3/_ssimrgb_skimage', rets[2]._ssimrgb_skimage, int(epoch / 10))
        sw.add_scalar('eval3/_psnr_skimage', rets[2]._psnr_skimage, int(epoch / 10))
        sw.add_scalar('eval3/_kid_std', rets[2]._kid_std, int(epoch / 10))
        sw.add_scalar('eval3/_fid_inkid_mean', rets[2]._fid_inkid_mean, int(epoch / 10))
        sw.add_scalar('eval3/_fid_inkid_std', rets[2]._fid_inkid_std, int(epoch / 10))
    model_G.train()
    model_seg.train()



def label_nums(data_loader,label_num=5): # 遍历dataloader，计算其所有图像中分割GT各label的pix总数
    ret=[]
    for i in range(label_num):
        ret.append(0)
    for step, sample in enumerate(data_loader):
        seg=sample["seg"]
        for i in range(label_num):
            ret[i]+=(seg==i).sum().item()
    return ret


def train(args, get_dataloader_func=get_pix2pix_maps_dataloader):
    with open(os.path.join(args.save,'args.json'), 'w') as f:
        json.dump(vars(args), f)
    logger = Logger(save_path=args.save, json_name='img2map_seg')
    epoch_now = len(logger.get_data('D_loss'))

    model_saver = ModelSaver(save_path=args.save,
                             name_list=['G', 'D', 'G_optimizer', 'D_optimizer',
                                        'G_scheduler', 'D_scheduler','DLV3P',"DLV3P_global_optimizer",
                                        "DLV3P_backbone_optimizer","DLV3P_global_scheduler","DLV3P_backbone_scheduler"])

    sw = SummaryWriter(args.tensorboard_path)


    G = get_G(args,input_nc=7) # 3+256+1，256为分割网络输出featuremap的通道数
    D = get_D(args,input_nc=7)
    model_saver.load('G', G)
    model_saver.load('D', D)

    cfg=Configuration()
    cfg.MODEL_NUM_CLASSES=args.label_nc
    DLV3P = deeplabv3plus(cfg)
    if args.gpu:
        # DLV3P=nn.DataParallel(DLV3P)
        DLV3P=DLV3P.cuda()
    model_saver.load('DLV3P', DLV3P)

    G_optimizer = Adam(G.parameters(), lr=args.G_lr, betas=(args.beta1, 0.999))
    D_optimizer = Adam(D.parameters(), lr=args.D_lr, betas=(args.beta1, 0.999))


    seg_global_params, seg_backbone_params=DLV3P.get_paras()
    DLV3P_global_optimizer = torch.optim.Adam([{'params': seg_global_params, 'initial_lr': args.seg_lr_global}], lr=args.seg_lr_global,betas=(args.beta1, 0.999))
    DLV3P_backbone_optimizer = torch.optim.Adam([{'params': seg_backbone_params, 'initial_lr': args.seg_lr_backbone}], lr=args.seg_lr_backbone, betas=(args.beta1, 0.999))

    model_saver.load('G_optimizer', G_optimizer)
    model_saver.load('D_optimizer', D_optimizer)

    model_saver.load('DLV3P_global_optimizer', DLV3P_global_optimizer)
    model_saver.load('DLV3P_backbone_optimizer', DLV3P_backbone_optimizer)

    G_scheduler = get_hinge_scheduler(args, G_optimizer)
    D_scheduler = get_hinge_scheduler(args, D_optimizer)
    DLV3P_global_scheduler=torch.optim.lr_scheduler.LambdaLR(DLV3P_global_optimizer, lr_lambda=lambda epoch:(1 - epoch/args.epochs)**0.9,last_epoch=epoch_now)
    DLV3P_backbone_scheduler = torch.optim.lr_scheduler.LambdaLR(DLV3P_backbone_optimizer,lr_lambda=lambda epoch: (1 - epoch / args.epochs) ** 0.9,last_epoch=epoch_now)

    model_saver.load('G_scheduler', G_scheduler)
    model_saver.load('D_scheduler', D_scheduler)
    model_saver.load('DLV3P_global_scheduler', DLV3P_global_scheduler)
    model_saver.load('DLV3P_backbone_scheduler', DLV3P_backbone_scheduler)

    device = get_device(args)

    GANLoss = get_GANLoss(args)
    if args.use_ganFeat_loss:
        DFLoss = get_DFLoss(args)
    if args.use_vgg_loss:
        VGGLoss = get_VGGLoss(args)
    if args.use_low_level_loss:
        LLLoss = get_low_level_loss(args)

    # CE_loss=nn.CrossEntropyLoss(ignore_index=255)
    LVS_loss = lovasz_softmax
    alpha = [0,0,0]
    for layer in range(args.layer_num):
        # start from layer_num
        id_layer = args.layer_num - layer
        if layer==0:
            continue
        else:
            data_loader_focal = get_inter1_dataloader(args, id_layer, train=True)
            data_loader_focal = tqdm(data_loader_focal)
            alpha_t = label_nums(data_loader_focal,label_num=args.label_nc)
            print(len(alpha_t))
            for i in range(len(alpha)):
                alpha[i] = alpha[i] + alpha_t[i]

    # alpha = [1,1,1,1,1]
    tmp_min = min(alpha)
    assert tmp_min > 0
    for i in range(len(alpha)):
        alpha[i] = tmp_min / alpha[i]
    if args.focal_alpha_revise:
        assert len(args.focal_alpha_revise) == len(alpha)
        for i in range(len(alpha)):
            alpha[i]=alpha[i]*args.focal_alpha_revise[i]
    print(alpha)
    FOCAL_loss = FocalLoss(gamma=2, alpha=alpha)


    if epoch_now==args.epochs or args.epochs==-1:
        print('get final models')
        eval_fidiou(args, sw=sw, model_G=G,model_seg=DLV3P)

    total_steps=0
    for epoch in range(epoch_now, args.epochs):
        G_loss_list = []
        D_loss_list = []
        LVS_loss_list=[]
        FOCAL_loss_list=[]

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

                data_loader = get_inter1_dataloader(args, id_layer, train=True)
                data_loader = tqdm(data_loader)

                for step, sample in enumerate(data_loader):
                    total_steps=total_steps+1
                    layer_imgs = sample['A'].to(device)
                    final_before_img = sample['B'].to(device)
                    target_layer_img = sample['C'].to(device)
                    imgs_seg = sample['RS'].to(device)  # (shape: (batch_size, 3, img_h, img_w))
                    label_imgs = sample['seg'].type(torch.LongTensor).to(device)  # (shape: (batch_size, img_h, img_w))



                    outputs, feature_map = DLV3P(imgs_seg)
                    # print(outputs.shape)
                    soft_outputs = torch.nn.functional.softmax(outputs, dim=1)
                    # print(soft_outputs)
                    # print(soft_outputs.shape)
                    # print(soft_outputs.shape)
                    # print(label_imgs.shape)
                    lvs_loss = LVS_loss(soft_outputs, label_imgs, ignore=255)
                    # lvs_loss = LVS_loss(soft_outputs, label_imgs)
                    lvs_loss_value = lvs_loss.data.cpu().numpy()
                    focal_loss = FOCAL_loss(outputs, label_imgs)
                    focal_loss_value = focal_loss.data.cpu().numpy()

                    seg_loss = (focal_loss + lvs_loss) * 0.5
                    boundary_aware = []
                    im_path = sample['A_path']

                    transform_list = [transforms.ToTensor()]
                    tf_boundary = transforms.Compose(transform_list)

                    for i in range(len(im_path)):
                        # 结构是CxHxW，C是类别，对应类别下点为1
                        np_soft_outputs = soft_outputs[i].cpu().detach().numpy()

                        # 构造和原始label相似的结构
                        np_new_outputs = np.zeros((args.fineSize,args.fineSize))
                        for j in range(len(np_soft_outputs)):
                            if j==0:
                                continue
                            else:
                                np_new_outputs = np_new_outputs + 2*np_soft_outputs[j]
                        np_new_outputs=abs(np_new_outputs)
                        np_outputs = np_new_outputs.astype(np.uint8)

                        cv_canny = cv2.Canny(np_outputs, 1, 1)
                        cv_canny_tf = tf_boundary(cv_canny)
                        cv_canny_tf = cv_canny_tf.reshape(1,1,cv_canny_tf.size(1),cv_canny_tf.size(2))
                        # print(cv_canny_tf)
                        boundary_aware.append(cv_canny_tf)
                    boundary_aware_tf=boundary_aware[0]
                    # print(len(im_path))
                    for i in range(len(im_path)):
                        if i==0:
                            continue
                        else:
                            boundary_aware_tf = torch.cat([boundary_aware_tf, boundary_aware[i]])
                    boundary_aware_tf = boundary_aware_tf.to(device)
                    # print(boundary_aware_tf.shape)
                    # print(layer_imgs.shape)
                    imgs_plus = torch.cat((layer_imgs.float(),final_before_img.float(),boundary_aware_tf.float()),1)

                    # train the Discriminator
                    D_optimizer.zero_grad()
                    reals_maps = torch.cat([layer_imgs.float(), target_layer_img.float(),boundary_aware_tf.float()], dim=1)
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


                    fakes_maps = torch.cat([layer_imgs.float(), fakes.float(),boundary_aware_tf.float()], dim=1)
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


                    fakes_maps = torch.cat([layer_imgs.float(), fakes.float(),boundary_aware_tf.float()], dim=1)
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
                    if args.use_ssim_loss:
                        ssim_loss_f = SSIM(window_size = 11)
                        ssim_loss = -ssim_loss_f(fakes,target_layer_img)
                        G_loss += ssim_loss
                        ssim_loss = ssim_loss.mean().item()
                    else:
                        ssim_loss=0

                    G_loss = G_loss.mean()
                    G_seg_loss = G_loss + seg_loss
                    G_loss = G_loss.item()
                    seg_loss = seg_loss.item()
                    G_optimizer.zero_grad()
                    DLV3P_global_optimizer.zero_grad()  # (reset gradients)
                    DLV3P_backbone_optimizer.zero_grad()

                    G_seg_loss.backward()
                    G_optimizer.step()
                    DLV3P_global_optimizer.step()  # (perform optimization step)
                    DLV3P_backbone_optimizer.step()


                    data_loader.write(f'Epochs:{epoch}  | Dloss:{D_loss:.6f} | Gloss:{G_loss:.6f}'
                                      f'| GANloss:{gan_loss:.6f} | VGGloss:{vgg_loss:.6f} | DFloss:{df_loss:.6f} '
                                      f'| FOCAL_loss:{focal_loss_value:.6f}| LVS_loss:{lvs_loss_value:.6f} '
                                      f'| LLloss:{ll_loss:.6f} | lr_gan:{get_lr(G_optimizer):.8f} | ssim_loss:{ssim_loss:.4f}')

                    G_loss_list.append(G_loss)
                    D_loss_list.append(D_loss)
                    LVS_loss_list.append(lvs_loss_value)
                    FOCAL_loss_list.append(focal_loss_value)

                    # tensorboard log
                    if args.tensorboard_log and step % args.tensorboard_log == 0:  # defalut is 5
                        # total_steps = epoch * len(data_loader) + step
                        # sw.add_scalar('Loss1/G', G_loss, total_steps)
                        sw.add_scalar('Loss/G', G_loss, total_steps)
                        sw.add_scalar('Loss/seg', seg_loss, total_steps)
                        sw.add_scalar('Loss/G_seg', G_seg_loss, total_steps)
                        sw.add_scalar('Loss/D', D_loss, total_steps)
                        sw.add_scalar('Loss/gan', gan_loss, total_steps)
                        sw.add_scalar('Loss/vgg', vgg_loss, total_steps)
                        sw.add_scalar('Loss/df', df_loss, total_steps)
                        sw.add_scalar('Loss/ll', ll_loss, total_steps)
                        sw.add_scalar('Loss/ssim', ssim_loss, total_steps)

                        sw.add_scalar('LR/G', get_lr(G_optimizer), total_steps)
                        sw.add_scalar('LR/D', get_lr(D_optimizer), total_steps)


                        sw.add_image('img2/A', tensor2im(layer_imgs.data), total_steps, dataformats='HWC')
                        sw.add_image('img2/B', tensor2im(final_before_img.data), total_steps, dataformats='HWC')


                        sw.add_image('img2/C', tensor2im(target_layer_img.data), total_steps, dataformats='HWC')
                        sw.add_image('img2/fake', tensor2im(fakes.data), total_steps, dataformats='HWC')
                        sw.add_image('img2/seg', tensor2im(boundary_aware_tf.data), total_steps, dataformats='HWC')


        D_scheduler.step(epoch)
        G_scheduler.step(epoch)
        DLV3P_global_scheduler.step()
        DLV3P_backbone_scheduler.step()



        logger.log(key='D_loss', data=sum(D_loss_list) / float(len(D_loss_list)))
        logger.log(key='G_loss', data=sum(G_loss_list) / float(len(G_loss_list)))
        logger.log(key='LVS_loss', data=sum(LVS_loss_list) / float(len(LVS_loss_list)))
        logger.log(key='FOCAL_loss', data=sum(FOCAL_loss_list) / float(len(FOCAL_loss_list)))
        logger.save_log()
        # logger.visualize()

        model_saver.save('G', G)
        model_saver.save('D', D)
        model_saver.save('DLV3P', DLV3P)

        model_saver.save('G_optimizer', G_optimizer)
        model_saver.save('D_optimizer', D_optimizer)
        model_saver.save('DLV3P_global_optimizer', DLV3P_global_optimizer)
        model_saver.save('DLV3P_backbone_optimizer', DLV3P_backbone_optimizer)

        model_saver.save('G_scheduler', G_scheduler)
        model_saver.save('D_scheduler', D_scheduler)
        model_saver.save('DLV3P_global_scheduler', DLV3P_global_scheduler)
        model_saver.save('DLV3P_backbone_scheduler', DLV3P_backbone_scheduler)
        if epoch==49:
            model_saver.save('G_50', G)
            model_saver.save('D_50', D)
            model_saver.save('DLV3P_50', DLV3P)

            model_saver.save('G_optimizer_50', G_optimizer)
            model_saver.save('D_optimizer_50', D_optimizer)
            model_saver.save('DLV3P_global_optimizer_50', DLV3P_global_optimizer)
            model_saver.save('DLV3P_backbone_optimizer_50', DLV3P_backbone_optimizer)

            model_saver.save('G_scheduler_50', G_scheduler)
            model_saver.save('D_scheduler_50', D_scheduler)
            model_saver.save('DLV3P_global_scheduler_50', DLV3P_global_scheduler)
            model_saver.save('DLV3P_backbone_scheduler_50', DLV3P_backbone_scheduler)


        if epoch == (args.epochs - 1) or (epoch % 10 == 0):
            import copy
            args2=copy.deepcopy(args)
            args2.batch_size=args.batch_size_eval
            eval_fidiou(args, sw=sw, model_G=G,model_seg=DLV3P, epoch=epoch)


if __name__ == '__main__':
    args = config()

    # args.label_nc = 5

    from src.pix2pixHD.myutils import seed_torch
    print(f'\nset seed as {args.seed}!\n')
    seed_torch(args.seed)

    train(args, get_dataloader_func=get_pix2pix_maps_dataloader)

pass
