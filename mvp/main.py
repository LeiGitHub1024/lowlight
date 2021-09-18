import torch
from torch import nn
import torch.utils.data as Data 
import torch.optim as optim
from tqdm import tqdm
from dataLoader import stl10_loader,mbllen_loader
from network import DenoiseAutoEncoder
from lolPSNR import lolPSNR
from loss import SSIM,LXJ_LOSS

print("GPU:",torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading...")
train_data,val_data = stl10_loader(device)

#数据加载器
train_loader = Data.DataLoader(dataset=train_data,batch_size=32,shuffle=True,num_workers=0)
val_loader = Data.DataLoader(dataset=val_data,batch_size=32,shuffle=True,num_workers=0)

# 构建网络
DAEmodel = DenoiseAutoEncoder().to(device)

# 可以开始训练了，
print("Training...")
LR = 0.0003
epoch_num = 10
optimizer = optim.Adam(DAEmodel.parameters(), lr=LR)
loss_func = LXJ_LOSS()
# nn.MSELoss()

model_path = "autodecode.mdl"
train_num, val_num = 0, 0
for epoch in tqdm(range(epoch_num)):
    train_loss_epoch, val_loss_epoch = 0, 0
    hq ='hq'

    # 训练
    for step, (b_x, b_y) in enumerate(train_loader):
        DAEmodel.train()
        _, output = DAEmodel(b_x)
        loss = loss_func(output, b_y,hq)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item() * b_x.size(0)
        train_num += b_x.size(0)
    # 验证
    for step, (b_x, b_y) in enumerate(val_loader):
        DAEmodel.eval()
        _, output = DAEmodel(b_x)
        loss = loss_func(output, b_y,hq)
        val_loss_epoch += loss.item() * b_x.size(0)
        val_num += b_x.size(0)
    # 计算一个epoch的损失
    train_loss = train_loss_epoch / train_num
    val_loss = val_loss_epoch / val_num
    print("epoch:",epoch+1," train_losss:",train_loss," val_loss:",val_loss)
   
    #保存模型
    torch.save(DAEmodel.state_dict(), model_path)

print("Validating!")
lolPSNR(model_path,device)


#     # 网络训练完后，我们可以随便挑一张图片来测试一下去噪的效果如何，此处我们使用PSNR（峰值信噪比）来度量干净的原图和自编码器输出的去噪图之间的相似性，PSNR越大说明两个图片之间越相似。
# imageindex = 1
# im = X_val[imageindex, ...]
# im = im.unsqueeze(0)
# im_noise = np.transpose(im.cpu().data.numpy(), (0, 3, 2, 1))
# im_noise = im_noise[0, ...]
# # 去噪
# DAEmodel.eval()
# _, output = DAEmodel(im)
# im_denoise= np.transpose(output.cpu().data.numpy(), (0, 3, 2, 1))
# im_denoise = im_denoise[0, ...]
# # 输出
# im = y_val[imageindex, ...]
# im_origin = im.unsqueeze(0)
# im_origin = np.transpose(im_origin.cpu().data.numpy(), (0, 3, 2, 1))
# im_origin = im_origin[0, ...]

# # 计算去噪后的PSNR
# print("加躁后的PSNR:", peak_signal_noise_ratio(im_origin, im_noise))
# print("去噪后的PSNR:", peak_signal_noise_ratio(im_origin, im_denoise))


# # 取一张lol图片，看效果
# high = mpimg.imread('data/lol485/high/2.png') # 
# low = mpimg.imread('data/lol485/low/2.png') # 

# input = np.transpose(low, (2, 1, 0))
# input = torch.tensor(input, dtype=torch.float32).to(device)
# input = input.unsqueeze(0)
# DAEmodel.eval()
# _, output = DAEmodel(input)
# im_enhance= np.transpose(output.cpu().data.numpy(), (0, 3, 2, 1))
# im_enhance = im_enhance[0, ...]

# plt.figure(figsize=[20, 20])
# plt.subplot(1, 3, 1)
# plt.imshow(low)
# plt.axis("off")
# plt.title("Lowlight image")

# plt.subplot(1, 3, 2)
# plt.imshow(im_enhance)
# plt.axis("off")
# plt.title("LowlightEnhance image")

# plt.subplot(1, 3, 3)
# plt.imshow(high)
# plt.axis("off")
# plt.title("Ground truth")
# print("原始的PSNR:", peak_signal_noise_ratio(low, high))
# print("暗光增强后的PSNR:", peak_signal_noise_ratio(im_enhance, high))



# #在loldataset上计算平均psnr

# #获取lol485目录信息

# lolList = os.listdir('./data/lol15/high')
# DAEmodel.eval()
# PSNR_init = []
# PSNR_enhance = []
# PSNR_plus = []
# for imgName in lolList:
#     high = mpimg.imread('data/lol15/high/'+imgName) # 
#     low = mpimg.imread('data/lol15/low/'+imgName) # 
#     input = np.transpose(low, (2, 1, 0))
#     input = torch.tensor(input, dtype=torch.float32).to(device)
#     input = input.unsqueeze(0)
#     _, output = DAEmodel(input)
#     im_enhance= np.transpose(output.cpu().data.numpy(), (0, 3, 2, 1))
#     im_enhance = im_enhance[0, ...]
#     PSNR_init.append(peak_signal_noise_ratio(low, high))
#     PSNR_enhance.append(peak_signal_noise_ratio(im_enhance, high))
#     PSNR_plus.append(peak_signal_noise_ratio(im_enhance, high)-peak_signal_noise_ratio(low, high))
#     # print("原始的PSNR:", peak_signal_noise_ratio(low, high))
#     # print("暗光增强后的PSNR:", peak_signal_noise_ratio(im_enhance, high))
# print(np.mean(PSNR_init))
# print(np.mean(PSNR_enhance))
# print(np.mean(PSNR_plus))