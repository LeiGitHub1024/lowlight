





# Uformer
nohup python3 ./train.py --arch Uformer --batch_size 4 --gpu '0' \
    --train_ps 256 --train_dir /home/mist/lowlight/datasets/lol/train --env 1_256_250_0\
    --val_dir ../datasets/LOL/valid --embed_dim 32 --warmup --nepoch 250 &

    
# UNet
# python3 ./train.py --arch UNet --batch_size 32 --gpu '0,1' \
#     --train_ps 128 --train_dir /cache/SIDD/train --env 32_0701_1 \
#     --val_dir /cache/SIDD/val --embed_dim 32 --warmup

#stage0
nohup python3 ./train_stage0.py --arch Wformer --batch_size 16 --gpu '0' \
    --train_ps 256 --train_dir ../train_datasets/lol_800/minibatches_RD_0 --env 3.2_RD_0 \
    --val_dir ../train_datasets/lol_800/eval_RD_0 --embed_dim 32 --warmup --nepoch=250  &


#stage1
nohup python3 ./train_stage1.py --arch Wformer --batch_size 8 --gpu '0' \
    --train_ps 256 --train_dir ../train_datasets/lol_800/minibatches_RD_1 --env 3.2.2_RD_1 \
    --val_dir ../train_datasets/lol_800/eval_RD_1 --embed_dim 32 --warmup --lr_initial 0.00005 --nepoch=50 &

# #stage1_resume
# nohup python3 ./train_1_resume.py --arch Uformer --batch_size 16 --gpu '0' \
#     --train_ps 64 --train_dir /home/mist/lowlight/datasets/lol_stage1/train --env 32_1006_1_2 \
#     --val_dir /home/mist/lowlight/datasets/lol_stage1/valid --embed_dim 32  --nepoch 150  --lr_initial 0.000001 &

#Wnet
nohup python3 ./train_stage1.py --arch Wnet --batch_size 8 --gpu '0' \
    --train_ps 256 --train_dir ../datasets/minipatches_RD_1 --env 1 \
    --val_dir ../datasets/eval15_RD_1 --embed_dim 32 --warmup --nepoch=500 &


