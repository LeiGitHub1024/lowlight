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


python3 ./train.py --arch Uformer --batch_size 32 --gpu '0' \
    --train_ps 64 --train_dir /home/mist/lowlight/datasets/illumination_KinD/train --env 32_1003_1 \
    --val_dir /home/mist/lowlight/datasets/illumination_KinD/valid --embed_dim 32 --warmup  --nepoch=100
    # --optimizer adam

# nohup python3 ./train.py --arch Uformer --batch_size 32 --gpu '0' \
#     --train_ps 128 --train_dir ../datasets/delowlight/lol/train --env 32_0701_1 \
#     --val_dir ../datasets/delowlight/lol/valid --embed_dim 32 --warmup &