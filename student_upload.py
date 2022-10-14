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
"""


############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):

    if mode == 'train':
        transData = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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
        return out


net = Network(BasicBlock)

############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
optimizer = optim.Adam(net.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.5)  # change learning rate
loss_func = nn.CrossEntropyLoss()


############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

# Normally, the default weight initialization and fixed learing rate
# should work fine. But, we have made it possible for you to define
# your own custom weight initialization and lr scheduler, if you wish.
def weights_init(m):
    return


############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
# 80% training, 20% testing
train_val_split = 0.8
batch_size = 128
epochs = 200
