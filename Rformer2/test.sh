# wformer2
conda activate wformer
python wformer_test.py --weights '/home/mist/low-light/Wformer2.0/log/Wformer3.2.2_RD_1/models/model_best.pth'\
  --arch 'Wformer'\
  --input_dir '/home/mist/low-light/train_datasets/lol_800/Real_captured/test'\
  --result_dir './res/wformer/lol-v2' \
  # --unpaired True

# wnet
python wformer_test.py --weights './log/Wnet1/models/model_epoch_200.pth' \
  --arch 'Wnet'\
  --input_dir '/home/mist/low-light/imageInPaper/ablation'\
  --result_dir './res/wnet' \
  --unpaired True

# wformer w/o guiding loss
python wformer_test.py --weights './log/Wformer_1114_no_g_loss/model_best.pth'\
  --arch 'Wformer'\
  --input_dir '/home/mist/low-light/imageInPaper/loltestpsnr/input'\
  --result_dir '/home/mist/low-light/imageInPaper/loltestpsnr/wformer_wo_guidingloss' \
  --unpaired True