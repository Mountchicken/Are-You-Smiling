import torch
import torch.nn as nn
import torch.nn.functional as F
import MyDataset
from Model import Network

def get_num_correct(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def main():
    '''hyperparamters'''
    num_epochs=20
    batch_size=64
    learning_rate=0.001

    '''边训练，边测试精确度，以获得最好模型'''
    test_loader=MyDataset.get_test_data_loader(batch_size=150)
    test_images,test_labels=next(iter(test_loader))
    test_images=test_images.to('cuda')
    test_labels=test_labels.to('cuda')

    '''初始化网络和优化器'''
    network=Network().to('cuda')
    train_loader=MyDataset.get_train_data_loader(batch_size=batch_size)
    optimizer=torch.optim.Adam(network.parameters(),lr=learning_rate)

    Loss=[]
    Correct=[]
    max_accurancy=0
    for epoch in range(num_epochs):
        epoch_loss=0
        epoch_correct=0
        for batch in train_loader:
            images,labels=batch
            images=images.to('cuda')
            labels=labels.to('cuda')
            preds=network(images)
            loss=F.cross_entropy(preds,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss+=loss.item()
            epoch_correct+=get_num_correct(preds,labels)

        Loss.append(epoch_loss)
        Correct.append(epoch_correct)
        '''不是很理想的训练方法，可以试一试'''
        network.eval()
        accurancy=get_num_correct(network(test_images),test_labels)/150
        
        if accurancy>max_accurancy:
            torch.save(network.state_dict(),"./Bestmodel.pkl")
            max_accurancy=accurancy
        print("epoch:",epoch,"epoch correct:", epoch_correct,"epoch loss:",epoch_loss,"Accuracy on test set: ",accurancy)
        
        # accurancy=get_num_correct(network(test_images),test_labels)/150
        # print("epoch:",epoch,"epoch correct:", epoch_correct,"epoch loss:",epoch_loss,"Accuracy on test set: ",accurancy)
        # torch.save(network.state_dict(),"./model.pkl")       
if __name__=='__main__':
    main()
