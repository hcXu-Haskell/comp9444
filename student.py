#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.


a. The whole assignment base on Resnet 18, one fully connected layers are added to accelerate convergence and 
   improve accuracy.

   The initial, one member tried Alex net, the other worked on Resnet 34. However, the accuracy of Alex net is 
   not ideal. The model of Resnet 34 is too large, and Resnet 34 has the problem of over-fitting. Therefore, 
   after several experiments, we turned to resnet 18 . It is an information technology composed of 18 convolution
   layers, with conv2d, batchnorm2d and the activation function relu. The two convolution layers act as basic blocks. 
   With the increase of channels, L2, L3 and L4 will generate their own channels. The input nodes and output nodes 
   of the two full connection layers are respectively (512, 64), (64, 8)

b. Loss function is Crossentropyloss. Optimiser is Adan

   Because this assignment is the classification of cats of different breeds, cross entropy is better than MSE.
   For example, if there are three types of cat A, B, C, the label is [1, 0, 0], and the network twice predicts 
   the result is [0.9, 0.1, 0.05] and [0.9, 0.3, 0.3]. So for MSE the second prediction is better than the first
   one. Therefore, Crossentropyloss is better than MSE, because MSE may give wrong judgment.
   
   In the selection of optimizer, we chose ADAM first. After several debugging, it was found that ADAM's concussion
   in the late training period would affect the accuracy rate. So the optimizer was replaced with SGD. But then it 
   was found that SGD had the following two problems: 1. The convergence speed was too slow. 2. Due to the high noise 
   of this data set (such as picture 2732 in folder 2, which clearly not a cat but a text flyer), SGD will also oscillate 
   and do not converge. Therefore, the optimizer finally chose ADAM and equipped with dynamic lr to solve the oscillation 
   problem
 
c. For image enhancement, because the cat is not necessarily in the middle of the picture or facing the lens 
   (the cat may lie belly up), random rotation, clipping, and scaling are added.

d. Epochs is 150. Batch size is 128.

   Epochs was initially set at 100, but after 100 epochs, it was found that the network was still convergent 
   (loss decreased, train and test acc increased), so epochs increased to 200.Subsequently, it was found that 
   the convergence of the model could be completed at about 150 epochs, and the epochs was changed to 150
   
   The default batch size is 200. Then it is found that the network does not converge and the loss is getting 
   larger and larger, so the batch size is reduced to 64. However, it is found that 64 converges too slowly and 
   there will be a shock,accuracy will jump form 70 to 60, so it increases to 128.

e. In order to solve the problem of over-fitting, three main measures are taken:

   1. Image enhancement
   2. Use dropout
   3. Reduce network complexity
   
   Image enhancement: on the basis of the original data set, the original image is transformed to generate new data
   to increase the number of samples and improve the diversity of samples.
   
   Use dropout: for the main structure of network (class Network), add drop at two additional full connection layers 
   with a probability of 0.2. (In line 170 and 186). For Resnet itself, residual blocks containing BN layer can inhibit 
   overfitting to a certain extent. However, since the model accuracy improved by about 2% after dropout removal and 
   no fitting occurred, the final model does not used line 170 and 186
   
   Reduce network complexity: Resnet 34 was used initial, but there were fitting problems with Resnet 34, so the 
   network was changed to Resnet 18.

"""


############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):

    if mode == 'train':
        transData = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomApply([transforms.RandomCrop(40)], p=0.4),
            transforms.Resize(80),
            transforms.RandomAffine(degrees=(-45, 45), translate=(0.3, 0.3), scale=(0.8, 1.2)),
        ])
# use Random Affine to Affine transformation images, to rotated, scaling and translate image to finish image enhancement
        return transData
    elif mode == 'test':
        transData = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        return transData


############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.BatchNormal1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.BatchNormal2 = nn.BatchNorm2d(out_channel)
        self.shortcut = shortcut
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.BatchNormal1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.BatchNormal2(out)
        if self.shortcut is None:
            out += x
        else:
            out += self.shortcut(x)
        out = F.relu(out)
        out = self.drop(out)
        return out


class Network(nn.Module):

    def __init__(self, BasicBlock):
        super(Network, self).__init__()
        self.flatten = nn.Flatten()

        # conv layer
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch = nn.BatchNorm2d(64)
        self.relu_1 = nn.ReLU()
        self.pool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # conv layer

        self.shotcut_1 = None
        self.layer1 = BasicBlock(64, 64, stride=1, shortcut=self.shotcut_1)

        self.shortcut_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0, bias=False),
                                        nn.BatchNorm2d(128))
        self.layer2 = BasicBlock(64, 128, stride=2, shortcut=self.shortcut_2)

        self.shortcut_3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0, bias=False),
                                        nn.BatchNorm2d(256))
        self.layer3 = BasicBlock(128, 256, stride=2, shortcut=self.shortcut_3)

        self.shortcut_4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=2, padding=0, bias=False),
                                        nn.BatchNorm2d(512))
        self.layer4 = BasicBlock(256, 512, stride=2, shortcut=self.shortcut_4)

        self.full_connect = nn.Linear(512, 8)
        # self.full_connect2 = nn.Linear(64, 8)
        # self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        # conv layer
        out = self.conv(x)
        out = self.batch(out)
        out = self.relu_1(out)
        out = self.pool_1(out)
        # conv layer

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)  # use average pooling to instead full connect
        out = self.flatten(out)
        out = self.full_connect(out)
        # out = self.drop(out)
        # out = self.full_connect(out)
        return out

    def weight_init(self):
        for m in self.modules():
            weights_init(m)


net = Network(BasicBlock)

############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
optimizer = optim.Adam(net.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.4)  # change learning rate
loss_func = nn.CrossEntropyLoss()


############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

# Normally, the default weight initialization and fixed learing rate
# should work fine. But, we have made it possible for you to define
# your own custom weight initialization and lr scheduler, if you wish.
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
# 80% training, 20% testing
train_val_split = 1
batch_size = 128
epochs = 150


# mark test times
test_mark = 54