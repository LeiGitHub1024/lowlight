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
    --train_ps 128 --train_dir ../datasets/lol/train --env 1020 \
    --val_dir ../datasets/lol/valid --embed_dim 32 --warmup  --nepoch=250 --lr_initial=0.0001 --resume --pretrain_weights './log/Uformer1011_mbllen_up/models/model_latest.pth' &

#Uformer onRD
nohup python3 ./train.py --arch Uformer --batch_size 8 --gpu '0' \
    --train_ps 256 --train_dir ../datasets/minipatches_RD_1 --env 1104 \
    --val_dir ../datasets/eval15_RD_1 --embed_dim 32 --warmup --lr_initial=0.0002 --nepoch=250 &

#UNet onRD
nohup python3 ./train.py --arch UNet --batch_size 8 --gpu '0' \
    --train_ps 256 --train_dir ../datasets/minipatches_RD_1 --env 1107 \
    --val_dir ../datasets/eval15_RD_1 --embed_dim 32 --warmup --lr_initial=0.0002 --nepoch=250 &

#Uformer 
nohup python3 ./train.py --arch Uformer --batch_size 8 --gpu '0' \
    --train_ps 256 --train_dir ../datasets/minipatches --env 1104 \
    --val_dir ../datasets/eval15 --embed_dim 32 --warmup --lr_initial=0.0002 --nepoch=250 &

#UNet 
nohup python3 ./train.py --arch UNet --batch_size 8 --gpu '0' \
    --train_ps 256 --train_dir ../datasets/minipatches --env 1107 \
    --val_dir ../datasets/eval15 --embed_dim 32 --warmup --lr_initial=0.0002 --nepoch=250 &

#UNetBadder
nohup python3 ./train.py --arch UNetBadder --batch_size 8 --gpu '0' \
    --train_ps 256 --train_dir ../datasets/minipatches --env 1108 \
    --val_dir ../datasets/eval15 --embed_dim 32 --warmup --lr_initial=0.0002 --nepoch=250 &


#UNetBadder
nohup python3 ./train.py --arch UNetBadder --batch_size 8 --gpu '0' \
    --train_ps 256 --train_dir ../datasets/our485 --env 1109 \
    --val_dir ../datasets/eval15 --embed_dim 32 --warmup --lr_initial=0.0002 --nepoch=250 &