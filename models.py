## TODO: define the convolutional neural network architecture

import torch
from torch import  nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


#practice neural networks for vision
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # (in_channel, number of filters, filter size 
        self.conv1 = nn.Conv2d(1, 32, 5, padding='same')
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.maxPool1 = nn.MaxPool2d(2)
        self.do1 = nn.Dropout(0.2)  # 20% Probability
        self.conv2 = nn.Conv2d(32, 16, 3, padding='same')
        self.act2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        self.maxPool2 = nn.MaxPool2d(2)
        self.do2 = nn.Dropout(0.2)  # 20% Probability
        self.fc1 = nn.Linear(56 * 56 * 16, 512)
        self.do3 = nn.Dropout(0.2)  # 20% Probability
        self.fc2 = nn.Linear(512, 136)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        out = self.conv1(x)
        out = self.act1(out)
        out = self.bn1(out)

        #print(f"this is the first conv size {out.size()}")
        out = self.maxPool1(out)
        out = self.do1(out)
        #print(f"this is the first maxpool size {out.size()}")

        out = self.conv2(out)
        out = self.act2(out)
        out = self.bn2(out)
        #print(f"this is the second conv size {out.size()}")

        out = self.maxPool2(out)
        out = self.do2(out)

        #print(f"this is the second maspool size {out.size()}")

        #block1 = self.maxPool1(self.act1(self.conv1(x)))
        #block2 = self.maxPool2(self.act2(self.conv2(block1)))
        flatten = out.view(-1, 56 * 56 * 16)
        #print(f"This is the flattened {flatten.size()}")
        out = self.fc1(flatten)
        out = self.do3(out)
        out = self.fc2(out)
        # a modified x, having gone through all the layers of your model, should be returned
        return out
