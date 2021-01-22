import numpy as np
import h5py
import torch
import math
import torchvision.transforms as transforms
from torch.utils.data import Dataset
def load_dataset():
    train_dataset = h5py.File('datasets/train_happy.h5', "r")
    train_set_image= np.array(train_dataset["train_set_x"]) # your train set features
    train_set_image=train_set_image.astype(float)
    train_set_label = np.array(train_dataset["train_set_y"]) # your train set labels

    test_dataset = h5py.File('datasets/test_happy.h5', "r")
    test_set_image = np.array(test_dataset["test_set_x"]) # your test set features
    test_set_image=test_set_image.astype(float)
    test_set_label = np.array(test_dataset["test_set_y"]) # your test set labels

    train_set=[train_set_image,train_set_label]
    test_set=[test_set_image,test_set_label]
   
    return train_set, test_set
class MyDataset(Dataset):
    def __init__(self,whichset,transform=None):
        train_set,test_set=load_dataset()
        if whichset :
            self.sets=train_set
        else:
            self.sets=test_set
        self.transform=transform
    def __getitem__(self,index):
        image=self.sets[0][index]
        label=self.sets[1][index]
        if self.transform is not None:
            image=self.transform(image)
        return image,label
    def __len__(self):
        return len(self.sets[0])

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([151.9839,146.2642,127.6968],
                        [78.58884375194165, 83.45698052457291, 79.58325255950746])                                
])
def get_train_data_loader(batch_size):
    train_set=MyDataset(whichset=True,transform=transform)
    return torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
def get_test_data_loader(batch_size):
    test_set=MyDataset(whichset=False,transform=transform)
    return torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=True)
'''获取Predict数据，不需要归一化'''
def get_predict_data_loader(batch_size=1):
    predict_image=MyDataset(whichset=False,transform=transform)
    return torch.utils.data.DataLoader(predict_image,batch_size=batch_size,shuffle=True)
    
def Normalize():
    train_set=MyDataset(whichset=True,transform=transform)
    train_loader=torch.utils.data.DataLoader(train_set,batch_size=len(train_set))
    batch=next(iter(train_loader))
    images,labels=batch
    images.shape
    temp_sum=0
    average=[]
    std=[]
    num_of_pixels=len(train_set)*64*64
    for j in range(3):
        for i in range(len(train_set)):
            temp_sum+=images[i][j].sum()
        average.append(temp_sum/num_of_pixels)
        temp_sum=0
    for j in range(3):
        for i in range(len(train_set)):
            temp_sum+=((images[i][j]-average[j]).pow(2)).sum()
        std.append(math.sqrt(temp_sum/num_of_pixels))
        temp_sum=0
    return average,std