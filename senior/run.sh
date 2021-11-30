


# input_dir='/home/mist/low-light/datasets/eval15/input'
# ground_dir='/home/mist/low-light/datasets/eval15/groundtruth'
# output_dir='/home/mist/low-light/datasets/eval15/res/'

input_dir='/home/mist/low-light/senior/test/input'
output_dir='/home/mist/low-light/senior/test/ex_final/'
mkdir $output_dir

output_mbllen=$output_dir'mbllen'
output_mbllen1=$output_dir'mbllen1'
output_retinex=$output_dir'retinex'
output_kind=$output_dir'kind'
output_tbefn=$output_dir'tbefn'
output_zero=$output_dir'zero'
output_ruas=$output_dir'ruas'
output_gan=$output_dir'gan'
output_rrd=$output_dir'rrd'
output_dip=$output_dir'dip'
output_llve=$output_dir'llve'
output_lime=$output_dir'lime'
output_sgllie=$output_dir'sgllie'
output_dslr=$output_dir'dslr'


# dslr , 那个需要额外处理，所以暂时不在这跑。
# wformer 因为涉及到高斯，所以不在这测试
# output_wformer=$output_dir'wformer'



# mkdir $output_mbllen
# mkdir $output_mbllen1
# mkdir $output_retinex
# mkdir $output_kind
# mkdir $output_tbefn
mkdir $output_zero
# mkdir $output_ruas
# mkdir $output_gan
# mkdir $output_rrd
# mkdir $output_dip
# mkdir $output_llve
# mkdir $output_lime
# mkdir $output_sgllie
# mkdir $output_dslr

# mkdir $output_wformer
# drbn 去loli平台搞吧，是在不想跑了


# cd EnlightenGAN
# conda activate wformer 
# python predict.py --dataroot $input_dir --results_dir $output_gan  --name enlightening --model single --which_direction AtoB --no_dropout --dataset_mode unaligned \
#   --which_model_netG sid_unet_resize --skip 1 --use_norm 1 --use_wgan 0 --self_attention --times_residual --instance_norm 0 --resize_or_crop='no'\
#   --which_epoch 200
# cd ..


# cd KinD
# conda activate tf16
# python evaluate.py -i $input_dir -r $output_kind
# cd ..

# cd MBLLEN/main
# conda activate tf16
# python test.py -i $input_dir -r $output_mbllen1 -c 0
# cd ../..

# cd RetinexNet
# conda activate tf16
# python main.py --phase=test --test_dir=$input_dir --save_dir=$output_retinex 
# cd ..


# cd RUAS
# conda activate wformer
# python test.py --data_path  $input_dir --save_path $output_ruas  --model lol 
# cd ..

# cd TBEFN
# conda activate tbefn
# python predict_TBEFN_tf2.py -i $input_dir -r $output_tbefn
# cd ..

cd Zero-DCE/Zero-DCE_code
conda activate wformer
python lowlight_test.py -i $input_dir -r $output_zero
cd ../..


# # #速度特别慢,暂时不用他，也不用跑
# cd RRDNet 
# conda activate wformer
# python pipline.py -i $input_dir -r $output_rrd
# cd ..

# # 速度特别慢
# cd RetinexDIP
# conda activate dip
# python Retinexdip.py --input $input_dir --result $output_dip
# cd ..


# cd StableLLVE
# conda activate wformer 
# python test.py --input_dir $input_dir --output_dir $output_llve
# cd ..


# cd LIME
# conda activate wformer 
# python demo.py -f $input_dir -o $output_lime -l 0.15 -g 0.6
# cd ..


# cd SGLLIE
# conda activate wformer
# python test.py --weight_dir weight/Epoch99.pth --input_dir $input_dir --test_dir $output_sgllie
# python resize.py --gt $input_dir --sg $output_sgllie
# cd ..


# cd DSLR 
# conda activate wformer
# python test.py --input_dir $input_dir --output_dir $output_dslr
# python resize.py --gt $input_dir --sg $output_dslr
# cd ..



# cd ../Wformer2.0
# conda activate wformer
# python wformer_test.py --input_dir $input_dir  --result_dir $output_wformer --unpaired True 
# cd ../senior

# conda activate wformer
# python evaluate.py -g $ground_dir -r $output_dir



