import random
import numpy as np

#why was this below? could the original cell handle that? 
import torch
import torch. nn as nn
import torch.nn.functional as F

# Set seed for reproducibili  ty
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(30) 



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Encoder
        # Now 3x3
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        # Decoder
        self.deconv5 = nn.ConvTranspose2d(512, 128, kernel_size=3, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 16, kernel_size=3, stride=1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)

        # self.conv1 = nn.Conv2d(1, 16, kernel_size=(1,3), stride=(1,1), padding=(0,1))
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=(1,3), stride=(1,1), padding=(0,1))
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=(1,3), stride=(1,1), padding=(0,1))
        # self.conv4 = nn.Conv2d(64, 128, kernel_size=(1,3), stride=(1,1), padding=(0,1))
        # self.conv5 = nn.Conv2d(128, 256, kernel_size=(1,3), stride=(1,1), padding=(0,1))
        
        # self.deconv5 = nn.ConvTranspose2d(512, 128, kernel_size=(1,3), stride=(1,1), padding=(0,1))
        # self.deconv4 = nn.ConvTranspose2d(256, 64, kernel_size=(1,3), stride=(1,1), padding=(0,1))
        # self.deconv3 = nn.ConvTranspose2d(128, 32, kernel_size=(1,3), stride=(1,1), padding=(0,1))
        # self.deconv2 = nn.ConvTranspose2d(64, 16, kernel_size=(1,3), stride=(1,1), padding=(0,1))
        # self.deconv1 = nn.ConvTranspose2d(32, 1, kernel_size=(1,3), stride=(1,1), padding=(0,1))

        
        self.elu = nn.ELU(inplace=True)
    
    def forward(self, x):
        e1 = self.elu(self.conv1(x))
        # print(f'e1:{e1.shape}')
        e2 = self.elu(self.conv2(e1))
        # print(f'e2:{e2.shape}')
        e3 = self.elu(self.conv3(e2))
        # print(f'e3:{e3.shape}')
        e4 = self.elu(self.conv4(e3))
        # print(f'e4:{e4.shape}')
        e5 = self.elu(self.conv5(e4))
        # print(f'e5:{e5.shape}')
        
        #the decoder with skip connections et al.
        d5 = self.elu(self.deconv5(torch.cat([e5, e5], dim=1)))
        # print(f'd5_real:{d5.shape}')
        d4 = self.elu(self.deconv4(torch.cat([d5, e4], dim=1)))
        # print(f'd4_real:{d4.shape}')
        d3 = self.elu(self.deconv3(torch.cat([d4, e3], dim=1)))
        # print(f'd3_real:{d3.shape}')
        d2 = self.elu(self.deconv2(torch.cat([d3, e2], dim=1)))
        # print(f'd2_real:{d2.shape}')
        # d2_real = F.pad(d2_real, (0,1))
        # print(f'd2_realPad:{d2_real.shape}')
        out = self.deconv1(torch.cat([d2, e1], dim=1))
        # print(f'output:{out.shape}')
        
        return out
    
#model = CNN()
#model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))