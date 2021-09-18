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

