#!/usr/bin/env python3
"""
student_test.py

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
	"""
	Called when loading the data. Visit this URL for more information:
	https://pytorch.org/vision/stable/transforms.html
	You may specify different transforms for training and testing
	"""

	if mode == 'train':
		return transforms.Compose([
			transforms.RandomApply([transforms.RandomRotation(80)], p=0.5),
			transforms.RandomPerspective(distortion_scale=0.3, p=0.5, fill=0),
			transforms.RandomApply([transforms.RandomCrop(48)], p=0.4),
			transforms.Resize(64),
			transforms.ToTensor(),
			transforms.RandomErasing(p=0.2),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
		])

	elif mode == 'test':
		return transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
		])


############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
"""
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
"""


def conv3_3(in_channel, out_channel, stride=1):
	cov = nn.Conv2d(in_channel,
	                out_channel,
	                kernel_size=(3, 3),
	                stride=stride,
	                padding=1,
	                bias=False)
	return cov


class res_block(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1, dp=0.4):
		super(res_block, self).__init__()
		self.activation = nn.ReLU(inplace=True)

		self.conv_1 = conv3_3(in_channels, out_channels, stride=stride)
		self.bat_1 = nn.BatchNorm2d(out_channels)

		self.conv_2 = conv3_3(in_channels, out_channels)
		self.bat_2 = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		res = x
		out = self.conv_1(x)
		out = F.relu(self.bat_1(out), True)
		out = self.conv_2(out)
		out = F.relu(self.bat_2(out), True)
		out = out + res
		out = F.relu(out, True)
		return out


class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()

		self.pre = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False),
		                         nn.BatchNorm2d(16),
		                         nn.ReLU(inplace=True),
		                         nn.MaxPool2d(3, 2, 1))
		self.layer1 = self.make_layer(16, 16, 3)
		self.layer2 = self.make_layer(16, 32, 4, stride=1)
		self.layer3 = self.make_layer(32, 64, 6, stride=1)
		self.layer4 = self.make_layer(64, 64, 3, stride=1)
		self.fc = nn.Linear(256, 8)

	def forward(self, x):
		x = self.pre(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = F.avg_pool2d(x, 7)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x

	def make_layer(self, inchancel, outchancel, num, stride=1):
		shortcut = nn.Sequential(nn.Conv2d(inchancel, outchancel, 1, stride, bias=False))
		layers = []
		layers.append(res_block(inchancel, outchancel, stride, shortcut))

		for i in range(1, num):

			layers.append(res_block(outchancel, outchancel))

		return nn.Sequential(*layers)


net = Network()

############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
optimizer = optim.SGD(net.parameters(), lr=0.001)

loss_func = nn.CrossEntropyLoss()


############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

# Normally, the default weight initialization and fixed learing rate
# should work fine. But, we have made it possible for you to define
# your own custom weight initialization and lr scheduler, if you wish.
def weights_init(m):
	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_normal_(m.weight, mode='fan_out')


scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
# scheduler = None

############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8
batch_size = 64
epochs = 200
