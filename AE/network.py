from torch import nn

# 接下来定义一个DenoiseAutoEncoder类，其中包括了自编码器的encoder和decoder，forward返回原数据编码后的特征映射和特征映射解码后的重构图像。
class DenoiseAutoEncoder(nn.Module):
    def __init__(self):
        super(DenoiseAutoEncoder, self).__init__()
        # Encoder
        self.Encoder = nn.Sequential(
            # param [input_c, output_c, kernel_size, stride, padding]
            nn.Conv2d(3, 64, 3, 1, 1),   # [, 64, 96, 96]
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1), # [, 64, 96, 96]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),             # [, 64, 48, 48]
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1),  # [, 64, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 1, 1), # [, 128, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, 1, 1), # [, 128, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, 1, 1), # [, 256, 48, 48]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                 # [, 256, 24, 24]
            nn.BatchNorm2d(256)   
        )
        
        # decoder
        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3 ,1, 1),   # [, 128, 24, 24]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 3, 2, 1, 1),   # [, 128, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),    # [, 64, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3, 1, 1),      # [, 32, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, 1, 1),      # [, 32, 48, 48]
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),  # [, 16, 96, 96]
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 3, 3, 1, 1),         # [, 3, 96, 96]
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoder = self.Encoder(x)
        decoder = self.Decoder(encoder)
        return encoder, decoder
  