import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_plane, out_plane, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_plane, out_plane, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_plane)
        self.conv2 = nn.Conv2d(out_plane, out_plane, 3, stride, padding=1 ,bias=False)
        self.bn2 = nn.BatchNorm2d(out_plane)
        self.conv3 = nn.Conv2d(out_plane, self.expansion*out_plane, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*out_plane)

        self.identity = nn.Sequential()

        if stride != 1 or in_plane != self.expansion * out_plane:
            self.identity = nn.Sequential(nn.Conv2d(in_plane, self.expansion*out_plane, 1, stride, bias=False),
                    nn.BatchNorm2d(self.expansion * out_plane))
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.identity(x)
        out = F.relu(out)

        return out

class FPN(nn.Module):
    def __init__(self, block, nums):
        super(FPN, self).__init__()

        self.in_plane = 64

        self.conv1 = nn.Conv2d(3, 64, 7, 2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        #bottom to top, c2->c3->c4->c5
        self.layer1 = self._make_layer(block, 64, nums[0], stride=1)
        self.layer2 = self._make_layer(block, 128, nums[1], stride=2)
        self.layer3 = self._make_layer(block, 256, nums[2], stride=2)
        self.layer4 = self._make_layer(block, 512, nums[3], stride=2)

        #trans layers: reduce(add) # of channels
        self.top = nn.Conv2d(512*block.expansion, 256, 1, 1, padding=0)
        self.trans1 = nn.Conv2d(256*block.expansion, 256, 1, 1, padding=0)
        self.trans2 = nn.Conv2d(128*block.expansion, 256, 1, 1, padding=0)
        self.trans3 = nn.Conv2d(64*block.expansion, 256, 1, 1, padding=0)

        #extra layers
        self.extra1 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.extra2 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.extra3 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.extra4 = nn.Conv2d(256, 256, 3, 1, padding=1)
        


    def _make_layer(self, block, planes, nums, stride):

        strides = [stride] + [1] * (nums-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_plane, planes, stride))
            self.in_plane = planes * block.expansion

        return nn.Sequential(*layers)
    
    def _upsample_add(self, x, y):

        #pytorch image order: N,C,H,W
        _,_,H,W = y.size()

        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y


    def forward(self, x):

        #build c1...c5
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride = 2, padding=1)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        #top down
        m5 = self.top(c5)
        m4 = self._upsample_add(m5, self.trans1(c4))
        m3 = self._upsample_add(m4, self.trans2(c3))
        m2 = self._upsample_add(m3, self.trans3(c2))

        #extra layers
        p5 = self.extra1(m5)
        p4 = self.extra2(m4)
        p3 = self.extra3(m3)
        p2 = self.extra4(m2)

        return p2, p3, p4, p5































