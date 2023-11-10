import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

VGG16 = [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']


class NN_Model(nn.Module):
    def __init__(self,in_channels,actv,depth_layer=10):
        super().__init__()
        self.in_channels = in_channels
        self.activation = actv
        self.depth = depth_layer
        self.conv_layers = self.create_layers(depth_layer)

        self.fcs = nn.Sequential(
            nn.Linear(49,20,bias=True),
            nn.Dropout(p=0.3),
            nn.Linear(20,10),
            nn.Softmax(dim=-1)
        )

    def forward(self,x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fcs(x)
        return x

        pass

    def create_layers(self,depth):
        layers = []
        in_channels = self.in_channels

        actvationFunction = nn.ReLU()
        if self.activation == "sigmoid":
            actvationFunction = nn.Sigmoid()
        elif self.activation == "tanh":
            actvationFunction = nn.Tanh()


        
        for x in range(depth):
            layers += [
                nn.Conv2d(in_channels=in_channels,out_channels=64,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(64),
                actvationFunction
            ]
            in_channels = 64

        layers += [
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(64,32,3,1,1),
            actvationFunction,
            nn.Conv2d(32,16,3,1,1),
            actvationFunction,
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,1,1,1),
            nn.Dropout(p=0.1),
            ]

        return nn.Sequential(*layers)
    
model = NN_Model(3,"ReLU",5)
x = torch.rand(1,3,28,28)
print(model(x).shape)