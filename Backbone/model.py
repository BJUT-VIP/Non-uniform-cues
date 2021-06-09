import torch
import torch.nn.functional as F
from torch import nn


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if abs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)

            return out_normal - self.theta * out_diff


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
                                groups=self.conv.groups)  # conv as padding 1 in T dim

            return out_normal - self.theta * out_diff[:, :, shrink // 2:(out_diff.shape[2] - shrink // 2), :, :]


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
        # maps = x.mean(2).squeeze() # [128, 128]
        maps = x.mean(2)  # [1, 128, 128]
        return maps


class T_Net(nn.Module):

    def __init__(self):
        super(T_Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.lastconv1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.lastconv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.lastconv3 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
        )

    def forward(self, x):  # inp [3, 256, 256]
        x1 = self.conv1(x)  # x [64, 128, 128]
        x2 = self.lastconv1(x1)  # x [128, 128, 128]
        x3 = self.lastconv2(x2)  # x [64, 128, 128]
        maps = self.lastconv3(x3)  # x [1, 128, 128]
        return maps


class CDCN(nn.Module):

    def __init__(self, map_size=32, pretrain_Mnet='', pretrain_Tnet='', theta=0.7):
        super(CDCN, self).__init__()

        self.conv1 = nn.Sequential(
            Conv2d_cd(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.Block1 = nn.Sequential(
            Conv2d_cd(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Conv2d_cd(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            Conv2d_cd(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.Block2 = nn.Sequential(
            Conv2d_cd(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Conv2d_cd(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            Conv2d_cd(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.Block3 = nn.Sequential(
            Conv2d_cd(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Conv2d_cd(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            Conv2d_cd(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.lastconv1 = nn.Sequential(
            Conv2d_cd(3 * 128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.lastconv2 = nn.Sequential(
            Conv2d_cd(128, 64, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.lastconv3 = nn.Sequential(
            Conv2d_cd(64, 1, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.ReLU(),
        )

        self.downsample32x32 = nn.Upsample(size=(map_size, map_size), mode='bilinear')

        self.M_Net = M_Net()
        self.M_Net = nn.DataParallel(self.M_Net)
        if pretrain_Mnet:
            resume_data = torch.load(pretrain_Mnet)
            self.M_Net.load_state_dict(resume_data['state_dict'])

        self.T_Net = T_Net()
        self.T_Net = nn.DataParallel(self.T_Net)
        if pretrain_Tnet:
            resume_data = torch.load(pretrain_Tnet)
            self.T_Net.load_state_dict(resume_data['state_dict'])
            self.cutoff = resume_data['cutoff']
            print(self.cutoff)

    def forward(self, inp, lbp):  # x [3, 256, 256]
        with torch.no_grad():
            M_attention = self.M_Net(inp[:, :, 1:, :, :]).detach()
            T_attention = self.T_Net(lbp).unsqueeze(1).detach()
            T_attention = T_attention if T_attention.mean() < self.cutoff[0] else 1

        x = inp[:, :, 0, :, :]

        x0 = self.conv1(x)
        x_Block1 = self.Block1(x0)  # x [128, 128, 128]
        x_Block1 = T_attention * M_attention * x_Block1
        # x_Block1 = M_attention * x_Block1  # replacing the above line with this line can also achieve considerable performance
        x_Block1_32x32 = self.downsample32x32(x_Block1)  # x [128, 32, 32]

        x_Block2 = self.Block2(x_Block1)  # x [128, 64, 64]
        x_Block2_32x32 = self.downsample32x32(x_Block2)  # x [128, 32, 32]

        x_Block3 = self.Block3(x_Block2)  # x [128, 32, 32]
        x_Block3_32x32 = self.downsample32x32(x_Block3)  # x [128, 32, 32]

        x_concat = torch.cat((x_Block1_32x32, x_Block2_32x32, x_Block3_32x32), dim=1)  # x [128*3, 32, 32]

        x = self.lastconv1(x_concat)  # x [128, 32, 32]
        x = self.lastconv2(x)  # x [64, 32, 32]
        map = self.lastconv3(x)  # x [1, 32, 32]

        return map, [lbp, M_attention, x_Block1, x_Block2, x_Block3, x_concat]
