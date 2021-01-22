import MyDataset
import torch
import torchvision
import torchvision.transforms as transforms
from Model import Network
from MyDataset import get_test_data_loader

from PIL import Image
Labels=['nothappy','happy']
def predict(image):
    if not torch.is_tensor(image):
        image=image.resize((64,64))
        '''将图像转为tensor'''
        loader=transforms.Compose([transforms.ToTensor()])
        image=loader(image).unsqueeze(dim=0)
    ''' 预测 '''
    network=Network()
    network.eval()
    network.load_state_dict(torch.load('Bestmodel.pkl'))

    pred=network(image).argmax(dim=1)
    return pred

if __name__=='__main__':
    Load_from_file=False
    
    if Load_from_file:
        path=input("Please input the path of the picture")
        image=Image.open(path)
        pred=predict(image)
    else :
        batch=get_test_data_loader(batch_size=1)
        image,label=next(iter(batch))
        pred=predict(image)
        print("the Prediction of ",Labels[label]," is:",Labels[pred])
