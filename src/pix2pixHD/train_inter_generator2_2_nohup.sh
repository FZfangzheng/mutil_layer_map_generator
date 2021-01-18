CUDA_VISIBLE_DEVICES=0 nohup python my_train_img2map_single_interlevelgenerator_level2.py \
--dataroot /data/multilayer_map_project/align_data2_2 \
--save /data/multilayer_map_project/inter2_2 \
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
--if_mutil_layer 0 \
--layer_num 4 > out_inter2.log 2>&1 &
