CUDA_VISIBLE_DEVICES=0 python my_train_img2map_inputsegresult_jointDL1_connect_featuremap_mutil_layer.py \
--dataroot /data/multilayer_map_project/train_data \
--save /data/multilayer_map_project/out_mutil_layer100 \
--gpu 1 \
--epochs 100 \
--batch_size 8 \
--test_batch_size 4 \
--loadSize 256 \
--fineSize 256 \
--crop_size 256 \
--resize_or_crop resize_and_crop \
--feat_num 0 \
--use_instance 0 \
--prefetch 0 \
--label_nc 3 \
--focal_alpha_revise 1 1 1 \
--a_loss 1 1 1 1 \
--use_vgg_loss 1 \
--use_ganFeat_loss 1 \
--use_low_level_loss 1 \
--low_level_loss L1 \
--netG local \
--n_downsample_global 3 \
--if_mutil_layer 1
