import torch
import torch.nn as nn


def _norm(channels: int) -> nn.Module:
    groups = 8
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 9, out_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.enc1 = ConvBlock(in_channels, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(c2, c3)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(c3, c4)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(c4, c4 * 2)

        self.up4 = nn.ConvTranspose2d(c4 * 2, c4, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(c4 * 2, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 * 2, c1)

        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    @staticmethod
    def _match_size(skip: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _, _, th, tw = target.shape
        _, _, sh, sw = skip.shape
        if sh < th or sw < tw:
            skip = nn.functional.pad(skip, (0, max(0, tw - sw), 0, max(0, th - sh)))
        return skip[:, :, :th, :tw]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        s4 = self.enc4(self.pool3(s3))
        b = self.bottleneck(self.pool4(s4))

        x = self.up4(b)
        x = self.dec4(torch.cat([x, self._match_size(s4, x)], dim=1))
        x = self.up3(x)
        x = self.dec3(torch.cat([x, self._match_size(s3, x)], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, self._match_size(s2, x)], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, self._match_size(s1, x)], dim=1))
        return self.head(x)
