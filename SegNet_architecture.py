import torch
import torch.nn as nn

class SegNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.encoder_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)
        self.encoder_conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)
        self.encoder_conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)
        self.encoder_conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)
        self.encoder_conv_5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.pool5 = nn.MaxPool2d(2, 2, return_indices=True)

        self.unpool5 = nn.MaxUnpool2d(2, 2)
        self.decoder_conv_5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.unpool4 = nn.MaxUnpool2d(2, 2)
        self.decoder_conv_4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.unpool3 = nn.MaxUnpool2d(2, 2)
        self.decoder_conv_3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.unpool2 = nn.MaxUnpool2d(2, 2)
        self.decoder_conv_2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.unpool1 = nn.MaxUnpool2d(2, 2)
        self.decoder_conv_1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 3, padding=1)
        )

    def forward(self, x):
        x1 = self.encoder_conv_1(x)
        x1p, idx1 = self.pool1(x1)
        x2 = self.encoder_conv_2(x1p)
        x2p, idx2 = self.pool2(x2)
        x3 = self.encoder_conv_3(x2p)
        x3p, idx3 = self.pool3(x3)
        x4 = self.encoder_conv_4(x3p)
        x4p, idx4 = self.pool4(x4)
        x5 = self.encoder_conv_5(x4p)
        x5p, idx5 = self.pool5(x5)

        d5 = self.unpool5(x5p, idx5, output_size=x5.size())
        d5 = self.decoder_conv_5(d5)
        d4 = self.unpool4(d5, idx4, output_size=x4.size())
        d4 = self.decoder_conv_4(d4)
        d3 = self.unpool3(d4, idx3, output_size=x3.size())
        d3 = self.decoder_conv_3(d3)
        d2 = self.unpool2(d3, idx2, output_size=x2.size())
        d2 = self.decoder_conv_2(d2)
        d1 = self.unpool1(d2, idx1, output_size=x1.size())
        out = self.decoder_conv_1(d1)
        return out
