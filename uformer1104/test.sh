
#unet
python wformer_test.py --weights './log/UNet1107/models/model_epoch_250.pth' \
  --arch 'UNet'\
  --input_dir '/home/mist/low-light/imageInPaper/ablation'\
  --result_dir './res/unet' \
  --unpaired True

  # --input_dir '/home/mist/low-light/datasets/LOL/test'\


#uformer
python wformer_test.py --weights './log/Uformer1104/models/model_epoch_100.pth' \
  --arch 'Uformer'\
  --input_dir '/home/mist/low-light/imageInPaper/ablation'\
  --result_dir './res/uformer' \
  --unpaired True

#unet badder
python wformer_test.py --weights './log/UNetBadder1108/models/model_epoch_50.pth' \
  --arch 'UNetBadder'\
  --input_dir '/home/mist/low-light/datasets/LOL/test'\
  --result_dir './res/unetbadder' \
  # --unpaired True

#unet much badder
python wformer_test.py --weights './log/UNetBadder1109/models/model_latest.pth' \
  --arch 'UNetBadder'\
  --input_dir '/home/mist/low-light/imageInPaper/source/ablation/input'\
  --result_dir './res/unetbadder' \
  --unpaired True


  