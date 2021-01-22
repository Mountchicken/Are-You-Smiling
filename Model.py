import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=7),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=10),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.fc1=nn.Sequential(
            nn.Linear(in_features=10*10*64,out_features=2048),
            nn.Dropout(0.5),
            nn.ReLU())

        self.fc2=nn.Sequential(
            nn.Linear(in_features=2048,out_features=1024),
            nn.Dropout(0.5),
            nn.ReLU())

        self.out=nn.Sequential(
            nn.Linear(in_features=1024,out_features=2))

    def forward(self,t):
        t=torch.tensor(t,dtype=torch.float32)
        t=self.conv1(t)
        t=self.conv2(t)
        t=t.reshape(-1,10*10*64)
        t=self.fc1(t)
        t=self.fc2(t)
        t=self.out(t)
        return t
