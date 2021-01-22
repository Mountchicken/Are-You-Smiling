import torch
from Model import Network
from Train import get_num_correct
from MyDataset import get_test_data_loader 

def test():
    test_loader=get_test_data_loader(batch_size=150)
    images,labels=next(iter(test_loader))

    network=Network()
    network.eval()
    network.load_state_dict(torch.load('Bestmodel.pkl'))
    preds=network(images)

    correct_nums=get_num_correct(preds,labels)
    print("the accurancy on the test set is: ",correct_nums/150)
if __name__=='__main__':
    test()