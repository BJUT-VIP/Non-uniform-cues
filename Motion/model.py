import torch.nn.functional as F
from torch import nn


class Conv3d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv3d_cd, self).__init__()
        self.stride = stride
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if abs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            kernel_diff = self.conv.weight.sum(2).sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None, None]
            shrink = x.shape[2] - out_normal.shape[2]  # no padding in T dim
            out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.stride, padding=0,
                                groups=self.conv.groups) # conv as padding 1 in T dim

            return out_normal - self.theta * out_diff[:,:,shrink//2:(out_diff.shape[2]-shrink//2),:,:]


class M_Net(nn.Module):

    def __init__(self, theta=1):
        super(M_Net, self).__init__()

        self.avgpool = nn.AvgPool3d(kernel_size=(1, 15, 15), stride=(1, 2, 2), padding=(0, 7, 7))

        self.conv1 = nn.Sequential(
            Conv3d_cd(3, 64, kernel_size=3, stride=1, padding=(0, 1, 1), bias=False, theta=theta),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )

        self.lastconv1 = nn.Sequential(
            Conv3d_cd(64, 128, kernel_size=3, stride=1, padding=(0, 1, 1), bias=False, theta=theta),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )

        self.lastconv2 = nn.Sequential(
            Conv3d_cd(128, 64, kernel_size=3, stride=1, padding=(0, 1, 1), bias=False, theta=theta),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )

        self.lastconv3 = nn.Sequential(
            Conv3d_cd(64, 1, kernel_size=3, stride=1, padding=(0, 1, 1), bias=False, theta=theta),
            nn.ReLU(),
        )

    def forward(self, inp):  # [3, 9, 256, 256]
        x0 = self.avgpool(inp)  # [3, 9, 128, 128]
        x1 = self.conv1(x0)  # [64, 7, 128, 128]
        x2 = self.lastconv1(x1)  # [128, 5, 128, 128]
        x3 = self.lastconv2(x2)  # [64, 3, 128, 128]
        x = self.lastconv3(x3)  # [1, 1, 128, 128]
        # maps = x.mean(2).squeeze()  # [128, 128]
        maps = x.mean(2)  # [1, 128, 128]

        return maps, [x0, x1, x2, x3]
