import torch
import torch.nn as nn
import torch.nn.functional as F
# Binary flag is concatenated to input in forward method
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p
    
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels): 
        super(UNet, self).__init__()
        self.down_conv1 = DownSample(in_channels, 64)
        self.down_conv2 = DownSample(64, 128)
        self.down_conv3 = DownSample(128, 256)
        self.down_conv4 = DownSample(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.up_conv1 = UpSample(1024, 512)
        self.up_conv2 = UpSample(512, 256)
        self.up_conv3 = UpSample(256, 128)
        self.up_conv4 = UpSample(128, 64)
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
    def forward(self, x, flag=None):
        # 如果有提供 binary flag，將其與 x 進行串接
        if flag is not None:
            x = torch.cat([x, flag], dim=1)
        B, C, H_orig, W_orig = x.shape
        #原本維度至C,H,W = [1,635,770]
        # Padding to 640 x 784
        x = F.pad(x, (7, 7, 2, 3)) # (Left, Right, Top, Bottom)
        d1, p1 = self.down_conv1(x)
        d2, p2 = self.down_conv2(p1)
        d3, p3 = self.down_conv3(p2)
        d4, p4 = self.down_conv4(p3)
        b = self.bottleneck(p4)
        u1 = self.up_conv1(b, d4)
        u2 = self.up_conv2(u1, d3)
        u3 = self.up_conv3(u2, d2)
        u4 = self.up_conv4(u3, d1)
        out = self.out(u4)
        out = torch.relu(out)
        #裁剪維度至C,H,W = [1,635,770]
        out = out[:, :, 2:-3, 7:-7]

        return out
    
if __name__ == "__main__":
    # 假設原始輸入是 6 channel，加上 binary flag 後變成 7 channel
    model = UNet(in_channels=7, out_channels=1)
    x = torch.randn(2, 6, 635, 770)  # Example input tensor with batch size 2
    flag = torch.randint(0, 2, (2, 1, 635, 770)).float() # Binary flag tensor
    output = model(x, flag)
    print(torch.cat([x, flag], dim=1).shape)
    print(output.shape)  # Should print torch.Size([2, 1, 635, 770])