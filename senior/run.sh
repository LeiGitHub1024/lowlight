


# input_dir='/data/private_data/mxy_private/lowlight/datasets/lol/test/input'
# ground_dir='/data/private_data/mxy_private/lowlight/datasets/lol/test/groundtruth'
# output_dir='/data/private_data/mxy_private/lowlight/datasets/lol/test/res/'

input_dir='/data/private_data/mxy_private/lowlight/datasets/test/MEF/input'
output_dir='/data/private_data/mxy_private/lowlight/datasets/test/MEF/res/'

output_mbllen=$output_dir'mbllen'
output_retinex=$output_dir'retinex'
output_kind=$output_dir'kind'
output_tbefn=$output_dir'tbefn'
output_zero=$output_dir'zero'
output_ruas=$output_dir'ruas'
output_gan=$output_dir'gan'
output_rrd=$output_dir'rrd'

mkdir $output_mbllen
mkdir $output_retinex
mkdir $output_kind
mkdir $output_tbefn
mkdir $output_zero
mkdir $output_ruas
mkdir $output_gan
mkdir $output_rrd

cd EnlightenGAN
conda activate mxy
python predict.py --dataroot $input_dir --results_dir $output_gan  --name enlightening --model single --which_direction AtoB --no_dropout --dataset_mode unaligned \
	        	--which_model_netG sid_unet_resize --skip 1 --use_norm 1 --use_wgan 0 --self_attention --times_residual --instance_norm 0 --resize_or_crop='no'\
	        	--which_epoch 200
cd ..

cd KinD
conda activate tf16
python evaluate.py -i $input_dir -r $output_kind
cd ..

cd MBLLEN/main
conda activate tf16
python test.py -i $input_dir -r $output_mbllen -c 0
cd ../..

cd Retinex
conda activate tf16
python main.py --use_gpu=1 --gpu_idx=0 --phase=test --test_dir=$input_dir --save_dir=$output_retinex --decom=0       
cd ..


cd RUAS
conda activate mxy
python test.py --data_path  $input_dir --save_path $output_ruas  --model lol 
cd ..

cd TBEFN
conda activate tf2
python predict_TBEFN_tf2.py -i $input_dir -r $output_tbefn
cd ..

cd Zero-DCE/Zero-DCE_code
conda activate mxy
python lowlight_test.py -i $input_dir 
cd ../..

cd RRDNet 
conda activate mxy
python pipline.py -i $input_dir -r $output_rrd
cd ..











# conda activate mxy
# python psnr_ssim.py -g $ground_dir -r $output_dir
