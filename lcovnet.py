# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
from torch.utils.checkpoint import checkpoint


class UnetBlock_Encode(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(UnetBlock_Encode, self).__init__()

        self.in_chns = in_channels
        self.out_chns = out_channel

        self.conv1 = nn.Sequential(
            nn.Conv3d(self.in_chns, self.out_chns, kernel_size=(1, 1, 3),
                      padding=(0, 0, 1)),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU(inplace=True)
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv3d(self.out_chns, self.out_chns, kernel_size=(3, 3, 1),
                      padding=(1, 1, 0), groups=1),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        self.conv2_2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=4, stride=2, padding=1),
            nn.Conv3d(self.out_chns, self.out_chns, kernel_size=1,
                      padding=0),
            nn.BatchNorm3d(self.out_chns),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        )

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x2 = torch.sigmoid(x2)

        x = x1 + x2 * x
        return x

class UnetBlock_Encode_4(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(UnetBlock_Encode_4, self).__init__()

        self.in_chns = in_channels
        self.out_chns = out_channel

        self.conv1 = nn.Sequential(
            nn.Conv3d(self.in_chns, self.out_chns, kernel_size=(1, 1, 3),
                      padding=(0, 0, 1)),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU(inplace=True)
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv3d(self.out_chns, self.out_chns, kernel_size=(3, 3, 1),
                      padding=(1, 1, 0), groups=self.out_chns),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        self.conv2_2 = nn.Sequential(
            nn.Conv3d(self.out_chns, self.out_chns, kernel_size=1,
                      padding=0),
            nn.BatchNorm3d(self.out_chns)
        )

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x2 = torch.sigmoid(x2)
        x = x1 + x2 * x
        return x



class UnetBlock_Down(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(UnetBlock_Down, self).__init__()
        self.avg_pool = nn.AvgPool3d(kernel_size=2)

    def forward(self, x):
        x = self.avg_pool(x)
        return x

class UnetBlock_Up(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(UnetBlock_Up, self).__init__()
        self.conv = self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channel, kernel_size=1,
                      padding=0, groups=1),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners = False)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x

class UNet_Seg(nn.Module):
    def __init__(self, C_in=1, n_classes=1):
        super(UNet_Seg, self).__init__()
        self.in_chns = C_in
        self.n_class = n_classes
        inchn = 32
        self.ft_chns = [inchn, inchn*2, inchn*4, inchn*8]
        self.resolution_level = len(self.ft_chns)

        self.block1 = UnetBlock_Encode(self.in_chns, self.ft_chns[0])

        self.block2 = UnetBlock_Encode(self.ft_chns[0], self.ft_chns[1])

        self.block3 = UnetBlock_Encode(self.ft_chns[1], self.ft_chns[2])

        self.block4 = UnetBlock_Encode_4(self.ft_chns[2], self.ft_chns[3])

        self.block5 = UnetBlock_Encode(2*self.ft_chns[2], self.ft_chns[2])

        self.block6 = UnetBlock_Encode(2*self.ft_chns[1], self.ft_chns[1])

        self.block7 = UnetBlock_Encode(2*self.ft_chns[0], self.ft_chns[0])

        self.down1 = UnetBlock_Down(self.ft_chns[0], self.ft_chns[0])

        self.down2 = UnetBlock_Down(self.ft_chns[1], self.ft_chns[1])

        self.down3 = UnetBlock_Down(self.ft_chns[2], self.ft_chns[2])

        self.up1 = UnetBlock_Up(self.ft_chns[3], self.ft_chns[2])

        self.up2 = UnetBlock_Up(self.ft_chns[2], self.ft_chns[1])

        self.up3 = UnetBlock_Up(self.ft_chns[1], self.ft_chns[0])

        self.conv = nn.Conv3d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, x):
        f1 = self.block1(x)
        d1 = self.down1(f1)

        f2 = self.block2(d1)
        d2 = self.down2(f2)

        f3 = self.block3(d2)
        d3 = self.down3(f3)

        f4 = self.block4(d3)

        f4up = self.up1(f4)
        f3cat = torch.cat((f3, f4up), dim=1)
        f5 = self.block5(f3cat)

        f5up = self.up2(f5)
        f2cat = torch.cat((f2, f5up), dim=1)
        f6 = self.block6(f2cat)

        f6up = self.up3(f6)
        f1cat = torch.cat((f1, f6up), dim=1)
        f7 = self.block7(f1cat)

        output = self.conv(f7)

        return output

class LCOVNet(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(LCOVNet, self).__init__()
        self.seg_network = UNet_Seg(input_channels, n_classes)

    def seg(self, x):
        output = self.seg_network(x)

    def forward(self, x):
        output = self.seg_network(x)
        return output