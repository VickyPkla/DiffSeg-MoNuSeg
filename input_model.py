import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    def __init__(self, channels, growth_rate=8):
        super(DenseBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, growth_rate, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_rate, growth_rate, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_rate, growth_rate, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_rate, growth_rate, 3, padding=1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_rate, channels, 3, padding=1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))

        # Local residual
        return x + x5


class RRDB(nn.Module):
    def __init__(self, channels):
        super(RRDB, self).__init__()

        self.db1 = DenseBlock(channels)
        self.db2 = DenseBlock(channels)
        self.db3 = DenseBlock(channels)

    def forward(self, x):
        residual = x

        out = self.db1(x)
        out = self.db2(out)
        out = self.db3(out)

        # Residual-in-Residual connection
        return residual + out


class RRDBNetSimple(nn.Module):
    def __init__(self, out_channels, num_features=16):
        super(RRDBNetSimple, self).__init__()

        # Input grayscale image
        self.conv_in = nn.Conv2d(3, num_features, 3, padding=1)

        # RRDB block
        self.rrdb = RRDB(num_features)

        # Conv after RRDB
        self.conv_mid = nn.Conv2d(num_features, num_features, 3, padding=1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        # Final output conv
        self.conv_out = nn.Conv2d(num_features, out_channels, 3, padding=1)

    def forward(self, x):

        x = self.conv_in(x)
        residual = x
        x = self.rrdb(x)
        x = x + residual
        x = self.conv_mid(x)
        x = self.lrelu(x)
        x = self.conv_out(x)

        return x

