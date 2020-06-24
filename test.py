from fpn import *
from torchsummary import summary

net = FPN(Bottleneck, [2,2,2,2])

x = torch.randn(1,3,256,256)
y1,y2,y3,y4 = net(x)

print("Model sumamry:")
print(summary(net, (3, 256, 256), device='cpu'))
print(y1.size(), y2.size(), y3.size(), y4.size())
