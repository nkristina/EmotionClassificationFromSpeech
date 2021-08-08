import torch.nn as nn
import torch.optim as optim

class CNN_2d_Model(nn.Module):

    def __init__(self):
        super(CNN_2d_Model,self).__init__()
        self.conv1=nn.Conv2d(in_channels = 1,
                            out_channels = 128,
                            kernel_size = 3,
                            stride  = 1,
                            padding = 'same',
                            bias = False)
        self.conv2 = nn.Conv2d(in_channels = 128,
                               out_channels = 128,
                               kernel_size = 3,
                               stride  = 1,
                               padding = 'same',
                               bias = False)
        self.conv3=nn.Conv2d(in_channels = 128,
                            out_channels = 64,
                            kernel_size = 3,
                            stride  = 1,
                            padding = 'same',
                            bias = False)
        self.conv4=nn.Conv2d(in_channels = 64,
                            out_channels = 64,
                            kernel_size = 3,
                            stride  = 1,
                            padding = 'same',
                            bias = False)
    
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.elu = nn.ELU()
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 4, stride = 4)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.25)
        self.drop3 = nn.Dropout(0.25)
        self.drop4 = nn.Dropout(0.25)

        self.gavgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64,2)
        self.softMax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.elu(x)
        x = self.maxpool1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        #x = self.bn2(x)
        x = self.elu(x)
        x = self.maxpool2(x)
        x = self.drop2(x)
        
        x = self.conv3(x)
        #x = self.bn3(x)
        x = self.elu(x)
        x = self.maxpool2(x)
        x = self.drop3(x)

        x = self.conv4(x)
        #x = self.bn4(x)
        x = self.elu(x)
        x = self.maxpool2(x)
        x = self.drop4(x)

        x = self.gavgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #x = self.softMax(x)

        return x 

    '''def __init__(self):
        super(CNN_2d_Model,self).__init__()
        self.conv1=nn.Conv2d(in_channels = 1,
                            out_channels = 64,
                            kernel_size = 3,
                            stride  = 1,
                            padding = 'same',
                            bias = False)
        self.conv2 = nn.Conv2d(in_channels = 64,
                               out_channels = 64,
                               kernel_size = 3,
                               stride  = 1,
                               padding = 'same',
                               bias = False)
                               
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.elu = nn.ELU()
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 4, stride = 4)
        self.drop1 = nn.Dropout()
        self.drop2 = nn.Dropout(0.25)

        self.gavgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64,2)'''

    '''def forward(self,x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.elu(x)
        x = self.maxpool1(x)
        #x = self.drop1(x)

        x = self.conv2(x)
        #x = self.bn2(x)
        x = self.elu(x)
        x = self.maxpool2(x)
        #x = self.drop2(x)

        x = self.gavgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x'''