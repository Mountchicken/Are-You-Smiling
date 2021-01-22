import os,sys
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog
from Ui_widget import Ui_AreYouSmling
from MyDataset import get_predict_data_loader
from PIL import Image,ImageQt

import torchvision.transforms as transforms
import torch
import numpy as np
from Model import Network
from Predict import predict
import matplotlib.pyplot as plt
Labels=['😒','😀']
class mywindow(QtWidgets.QWidget,Ui_AreYouSmling):
    def __init__(self):
        super(mywindow,self).__init__()
        self.cwd=os.getcwd()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.read_dataset)
        self.pushButton_2.clicked.connect(self.read_file)
        self.pushButton_3.clicked.connect(self.predict)
    def read_file(self):
        self.label_2.setText(" ")
        self.label_3.setText(" ")
        file,filetype=QFileDialog.getOpenFileName(self,'open image',self.cwd,"*.JPG,*.JPEG,*.png,*.jpg,ALL Files(*)")
        jpg = QtGui.QPixmap(file).scaled(self.label.width(), self.label.height())
        
        self.image=jpg
        self.label.setPixmap(jpg)

    def read_dataset(self):
        self.label_2.setText(" ")
        batch=get_predict_data_loader()
        image,label=next(iter(batch))
        self.image=image #将tensor传走
        self.label_3.setText(Labels[label])
        image=image.squeeze(dim=0)
        unloader=transforms.ToPILImage()
        image=unloader(image)
        pixmap=ImageQt.toqpixmap(image)
        jpg=QtGui.QPixmap(pixmap).scaled(self.label.width(),self.label.height())
        self.label.setPixmap(jpg)
    
       
    def predict(self):
        #先将图片转为PIL形式
        if  torch.is_tensor(self.image):
            image=self.image
        else:
            image=ImageQt.fromqpixmap(self.image)       
        self.label_4.setText('Predicting')
        pred=predict(image)
        self.label_4.setText('Predicted')
        self.label_2.setText(Labels[pred])


if __name__=="__main__":
    app=QtWidgets.QApplication(sys.argv)
    myshow=mywindow()
    myshow.show()
    sys.exit(app.exec_())