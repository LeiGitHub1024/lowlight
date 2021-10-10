#stage0
nohup python3 ./train_0.py --arch Uformer --batch_size 32 --gpu '0' \
    --train_ps 64 --train_dir /home/mist/lowlight/datasets/lol_stage0/train --env 32_1006_3_0 \
    --val_dir /home/mist/lowlight/datasets/lol_stage0/valid --embed_dim 32 --warmup  --nepoch=400 &


#stage1
nohup python3 ./train_1.py --arch Uformer --batch_size 16 --gpu '0' \
    --train_ps 64 --train_dir /home/mist/lowlight/datasets/lol_stage1/train --env 32_1005_2_1 \
    --val_dir /home/mist/lowlight/datasets/lol_stage1/valid --embed_dim 32 --warmup  --nepoch=350 &

#stage1_resume
nohup python3 ./train_1_resume.py --arch Uformer --batch_size 16 --gpu '0' \
    --train_ps 64 --train_dir /home/mist/lowlight/datasets/lol_stage1/train --env 32_1005_2_2 \
    --val_dir /home/mist/lowlight/datasets/lol_stage1/valid --embed_dim 32  --nepoch=150  &
