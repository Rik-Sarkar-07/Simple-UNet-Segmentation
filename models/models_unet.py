import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, num_classes, depth=4, num_channels=32):
        super(UNet, self).__init__()
        self.depth = depth
        channels = [num_channels * (2**i) for i in range(depth)]
        
        # Encoder
        self.downs = nn.ModuleList()
        in_channels = 3
        for out_channels in channels[:-1]:
            self.downs.append(DoubleConv(in_channels, out_channels))
            in_channels = out_channels
        
        # Bottom
        self.bottom = DoubleConv(channels[-2], channels[-1])
        
        # Decoder
        self.ups = nn.ModuleList()
        for i in range(depth-1, 0, -1):
            self.ups.append(nn.ConvTranspose2d(channels[i], channels[i-1], kernel_size=2, stride=2))
            self.ups.append(DoubleConv(channels[i], channels[i-1]))
        
        # Final conv
        self.final_conv = nn.Conv2d(channels[0], num_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = nn.MaxPool2d(2)(x)
        
        # Bottom
        x = self.bottom(x)
        
        # Decoder
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i//2]
            x = torch.cat((skip, x), dim=1)
            x = self.ups[i+1](x)
        
        # Final conv
        x = self.final_conv(x)
        return x