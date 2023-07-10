import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class AttentionLayer(nn.Module):
    def __init__(self,in_channel,reduction): # reduction 减少
        super(AttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(in_channel,in_channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, in_channel),
            nn.Sigmoid()
        )


    def forward(self,x): # torch.Size([1, 256, 64, 64])
        b,c,_,_ = x.size()
        #print(x.shape)
        # x1 = self.avg_pool(x) # torch.Size([1, 256, 1, 1])
        x2 = self.avg_pool(x).view(b,c) # torch.Size([1, 256])
        #print(x2.shape)
        # y1 = self.fc(x2) # torch.Size([1, 256])
        y2 = self.fc(x2).view(b,c,1,1)  # torch.Size([1, 256, 1, 1])
        #print(y2.shape)
        y3 = y2.expand_as(x)
        #print(y3.shape)
        y4 = x * y3
        #print(y4.shape) # torch.Size([1, 256, 64, 64])
        return y4

class BottleNeck(nn.Module):
    expansion = 4
	
    
    '''
    espansion是通道扩充的比例
    注意实际输出channel = middle_channels * BottleNeck.expansion
    '''
    def __init__(self, in_channels, middle_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(middle_channels, middle_channels, stride=stride, kernel_size=3, padding=3, bias=False,dilation=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(middle_channels, middle_channels, stride=stride, kernel_size=3, padding=2, bias=False,dilation=2),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(middle_channels, middle_channels, stride=stride, kernel_size=3, padding=3, bias=False,dilation=5),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(middle_channels, middle_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle_channels * BottleNeck.expansion),
        )
        self.se=AttentionLayer(BottleNeck.expansion*middle_channels,16)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != middle_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, middle_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(middle_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        x1=x
        #print(x1.shape,'初始x1！')
        x1=self.residual_function(x1)
        #print(x1.shape,'残差后x1！')
        x1=self.se(x1)
        #print(x1.shape,'se后x1！')
        #print(x1.shape)
        return nn.ReLU(inplace=True)(x1 + self.shortcut(x))
            

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU()
        )
        self.second = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.first(x)
        out = self.second(out)
        return out
    
    
    
class EAMNet(nn.Module):
    def __init__(self, block, layers, num_classes, input_channels=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 3)

        nb_filter = [64, 128, 256, 512, 1024]

        self.in_channel = nb_filter[0]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        
        self.conv1_0 = self._make_layer(block,nb_filter[1], layers[0], 1)
        self.conv2_0 = self._make_layer(block,nb_filter[2], layers[1], 1)
        self.conv3_0 = self._make_layer(block,nb_filter[3], layers[2], 1)
        self.conv4_0 = self._make_layer(block,nb_filter[4], layers[3], 1)

        self.conv3_1 = VGGBlock((nb_filter[3] + nb_filter[4]) * block.expansion, nb_filter[3],
                            nb_filter[3] * block.expansion)
        self.conv2_2 = VGGBlock((nb_filter[2] + nb_filter[3]) * block.expansion, nb_filter[2],
                            nb_filter[2] * block.expansion)
        self.conv1_3 = VGGBlock((nb_filter[1] + nb_filter[2]) * block.expansion, nb_filter[1],
                            nb_filter[1] * block.expansion)
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1] * block.expansion, nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block,middle_channel, num_blocks, stride):
        '''
        middle_channels中间维度，实际输出channels = middle_channels * block.expansion
        num_blocks，一个Layer包含block的个数
        '''

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, middle_channel, stride))
            self.in_channel = middle_channel * block.expansion
        return nn.Sequential(*layers)


    def forward(self, input):
        #print(input.shape)
        x0_0 = self.conv0_0(input)
        #print(x0_0.shape)
        x1_0 = self.conv1_0(self.pool(x0_0))
        #print(x1_0.shape)
        x2_0 = self.conv2_0(self.pool(x1_0))
        #print(x2_0.shape)
        x3_0 = self.conv3_0(self.pool(x2_0))
        #print(x3_0.shape)
        x4_0 = self.conv4_0(self.pool(x3_0))
        #print(x4_0.shape)
        #print((torch.cat([x3_0, self.up(x4_0)], 1)).shape)

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        #print(x3_1.shape)
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        #print(x2_2.shape)
        # x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        # x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        output = self.avg_pool(x2_2)
        output = output.view(output.size(0), -1)
        output = torch.flatten(output, 1)
        output = self.fc(output)

        return output

# if __name__ == '__main__':
#     net=EAMNet(block=BottleNeck,layers=[3,4,6,3],num_classes=3)
#     x = torch.rand((1, 3, 224, 224))
#     print(net.forward(x).shape)
