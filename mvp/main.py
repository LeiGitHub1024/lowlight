import torch
from torch import nn
import torch.utils.data as Data 
import torch.optim as optim
from tqdm import tqdm
from dataLoader import stl10_loader,mbllen_loader
from network import DenoiseAutoEncoder
from lolPSNR import lolPSNR
from loss import SSIM,LXJ_LOSS
from torchvision import models, transforms
from torch.autograd import Variable


print("GPU:",torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading...")
train_data,val_data = stl10_loader(device)

#数据加载器
train_loader = Data.DataLoader(dataset=train_data,batch_size=32,shuffle=True,num_workers=0)
val_loader = Data.DataLoader(dataset=val_data,batch_size=32,shuffle=True,num_workers=0)

# 构建网络
DAEmodel = DenoiseAutoEncoder().to(device)

#vgg
class VGGEncoder(nn.Module):
    def __init__(self):
        super(VGGEncoder, self).__init__()
        VGG = models.vgg16(pretrained=True)
        self.feature = VGG.features
        self.classifier = nn.Sequential(*list(VGG.classifier.children())[:-3])
        pretrained_dict = VGG.state_dict()
        model_dict = self.classifier.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.classifier.load_state_dict(model_dict)
 
    def forward(self, x):
        output = self.feature(x)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        return output
 
VGG_model = VGGEncoder().to(device)
VGG_model = VGG_model.eval()

def vgg_val(img):
    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False).to(device)
    y = VGG_model(x).to(device)

    y = torch.squeeze(y)
    return y
def vgg_loss(x,y):
    size = x.shape[0]
    res = 0
    for i in range(size):
        xxx = vgg_val(x[i])-vgg_val(y[i])
        res = res+ torch.mean(torch.abs(xxx))
    return res/size




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

    # 训练
    for step, (b_x, b_y) in enumerate(train_loader):
        # print(b_x,b_y,b_x.shape,b_y.shape)
        DAEmodel.train()
        _, output = DAEmodel(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item() * b_x.size(0)
        train_num += b_x.size(0)
    # 验证
    for step, (b_x, b_y) in enumerate(val_loader):
        DAEmodel.eval()
        _, output = DAEmodel(b_x)
        loss = loss_func(output, b_y)
        val_loss_epoch += loss.item() * b_x.size(0)
        val_num += b_x.size(0)
        # print(vgg_loss(output,b_y))

    # 计算一个epoch的损失
    train_loss = train_loss_epoch / train_num
    val_loss = val_loss_epoch / val_num
    print("epoch:",epoch+1," train_losss:",train_loss," val_loss:",val_loss)
   
    #保存模型
    torch.save(DAEmodel.state_dict(), model_path)

print("Validating!")
lolPSNR(model_path,device)

