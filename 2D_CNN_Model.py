import torch.nn as nn
import torch.optim as optim

class CNN_2d_Model(nn.Module):

    def __init__(self):
        super(CNN_2d_Model,self).__init__()
        self.conv1=nn.Conv2d(in_channels = 1,
                            out_channels = 128,
                            kernel_size = 3,
                            stride  = 1,
                            padding = 'same')
        self.bn1 = nn.BatchNorm1d()
        self.elu = nn.ELU()
        self.maxpool1 = nn.MaxPool1d(kernel_size = 2, stride = 2)
        self.maxpool2 = nn.MaxPool1d(kernel_size = 4, stride = 4)
        self.drop1 = nn.Dropout()
        self.drop2 = nn.Dropout(0.25)
        self.conv2=nn.Conv2d(in_channels = 1,
                            out_channels = 64,
                            kernel_size = 3,
                            stride  = 1,
                            padding = 'same')

        self.gavgpool = nn.AdaptiveAvgPool2d((1,1))
        self.softMax = nn.Softmax(dim = 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.maxpool1(x)
        x = self.drop1(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.maxpool2(x)
        x = self.drop2(x)
        
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.maxpool2(x)
        x = self.drop2(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.maxpool2(x)
        x = self.drop2(x)

        x = self.gavgpool(x)
        x = self.softMax(x)