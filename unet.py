import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from preprocessing import read_tif


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.two_convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.two_convs(x)


class UNet(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):

        super(UNet, self).__init__()
        self.up_layers = nn.ModuleList()
        self.down_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNet
        for feature in features:
            self.down_layers.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.bottom = DoubleConv(features[3], features[3]*2)

        # Up part of UNet
        for feature in reversed(features):
            self.up_layers.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.up_layers.append(DoubleConv(feature*2, feature))

        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x):
        skip_connections = []

        # each: double conv, max pool
        for down in self.down_layers:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottom(x)
        skip_connections = skip_connections[::-1]

        # each: upsample, concat, double conv
        for i in range(0, len(self.up_layers)//2):
            x = self.up_layers[2*i](x)
            skip_connection = skip_connections[(2*i)//2]

            concat_connect = torch.cat((skip_connection, x), dim=1)
            x = self.up_layers[2*i+1](concat_connect)

        return self.final_conv(x)


class BrainMriDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: pd.DataFrame, transform=None):
        super(BrainMriDataset, self).__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # sos here don't forget to import function read_tif from ggf. separate file
        image = read_tif(self.dataset.iloc[index, 0], color=True)
        mask = read_tif(self.dataset.iloc[index, 1], color=False)

        image = np.array(image)/255
        mask = np.array(mask, dtype=np.float32)/255

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']

        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).type(torch.float32)

        mask = np.expand_dims(mask, axis=-1).transpose((2, 0, 1))
        mask = torch.from_numpy(mask).type(torch.float32)

        return image, mask
