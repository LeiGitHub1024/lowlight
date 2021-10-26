# Uformer16
# python3 ./train.py --arch Uformer --batch_size 32 --gpu '0,1' \
#     --train_ps 128 --train_dir /cache/SIDD/train --env 16_0701_1 \
#     --val_dir /cache/SIDD/val --embed_dim 16 --warmup

# Uformer32
# python3 ./train.py --arch Uformer --batch_size 32 --gpu '0,1' \
#     --train_ps 128 --train_dir /cache/SIDD/train --env 32_0701_1 \
#     --val_dir /cache/SIDD/val --embed_dim 32 --warmup

    
# UNet
# python3 ./train.py --arch UNet --batch_size 32 --gpu '0,1' \
#     --train_ps 128 --train_dir /cache/SIDD/train --env 32_0701_1 \
#     --val_dir /cache/SIDD/val --embed_dim 32 --warmup

# conda activate uformer

nohup python3 ./train.py --arch Uformer --batch_size 48 --gpu '0' \
    --train_ps 128 --train_dir ../datasets/lol/train --env 1011_mbllen_up \
    --val_dir ../datasets/lol/valid --embed_dim 32 --warmup  --nepoch=250 --lr_initial=0.0001 --resume --pretrain_weights './log/Uformer1011_mbllen_up/models/model_latest.pth' &


python3 ./train.py --arch MyUnet --batch_size 16 --gpu '1,2' \
    --train_ps 256 --train_dir ../datasets/lol/train --env 1025_unet_ps256_l1_lol \
    --val_dir ../datasets/lol/valid --embed_dim 32 --warmup --lr_initial=0.0002 --nepoch=250


python3 ./train.py --arch MyUnet --batch_size 16 --gpu '0,4' \
    --train_ps 256 --train_dir ../datasets/lol/train --env 1025_unet_ps256_ssim_lol \
    --val_dir ../datasets/lol/valid --embed_dim 32 --warmup --lr_initial=0.0002 --nepoch=5


# nohup python3 ./train.py --arch Uformer --batch_size 32 --gpu '0' \
#     --train_ps 128 --train_dir ../datasets/delowlight/lol/train --env 32_0701_1 \
#     --val_dir ../datasets/delowlight/lol/valid --embed_dim 32 --warmup &