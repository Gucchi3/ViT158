import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(in_channels)
        self.depthwise_relu = nn.ReLU(inplace=True)

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.pointwise_bn = nn.BatchNorm2d(out_channels)
        self.pointwise_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise_relu(self.depthwise_bn(self.depthwise(x)))
        x = self.pointwise_relu(self.pointwise_bn(self.pointwise(x)))
        return x


class tiny_cnn(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        in_chans = kwargs.get("in_chans", 3)
        num_classes = kwargs.get("num_classes", 10)

        self.stem_conv = nn.Conv2d(in_chans, 12, kernel_size=3, stride=1, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(12)
        self.stem_relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ds_conv1 = DepthwiseSeparableConv(12, 24, stride=1)
        self.ds_conv2 = DepthwiseSeparableConv(24, 48, stride=2)
        self.ds_conv3 = DepthwiseSeparableConv(48, 72, stride=1)
        self.ds_conv4 = DepthwiseSeparableConv(72, 96, stride=2)

        # 32x32 -> stem(32x32) -> pool(16x16) -> ds1(16x16) -> ds2(8x8) -> ds3(8x8) -> ds4(4x4)
        self.fc = nn.Linear(96 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.stem_relu(self.stem_bn(self.stem_conv(x)))
        x = self.pool(x)

        x = self.ds_conv1(x)
        x = self.ds_conv2(x)
        x = self.ds_conv3(x)
        x = self.ds_conv4(x)

        x = torch.flatten(x, 1)

        out = self.fc(x)

        return out


if __name__ == "__main__":
    from torchinfo import summary
    summary(tiny_cnn(), input_size=(1, 3, 32, 32))
