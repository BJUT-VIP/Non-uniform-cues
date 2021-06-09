from torch import nn
import torch.nn.functional as F


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
        return maps, [x1, x2, x3]
