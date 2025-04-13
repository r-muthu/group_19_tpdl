import torch

class ResBlock(torch.nn.Module):
    """
    ResNet block.
    1x1, 3x3, 1x1 kernels.
    """

    EXP = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int=1):
        super(ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.conv3 = torch.nn.Conv2d(out_channels, out_channels * self.EXP, kernel_size=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(out_channels * self.EXP)
        self.relu = torch.nn.ReLU(inplace=True)
        
        self.downsample = None if (stride == 1 and in_channels == out_channels * self.EXP) else torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels * self.EXP, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels * self.EXP)
                )

    def forward(self, x):
        inp = x if self.downsample is None else self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        
        out = self.relu(self.bn2(self.conv2(out)))

        out = self.bn3(self.conv3(out))

        out += inp
        out = self.relu(out)

        return out

class ResLayer(torch.nn.Module):
    """
    Chain of ResNet blocks.
    """
    def __init__(self, in_channels: int, out_channels: int, blocks: int, stride: int=1):
        super(ResLayer, self).__init__()
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride=stride))

        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels * ResBlock.EXP, out_channels))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class SelfAttentionPooling(torch.nn.Module):

    def __init__(self, input_dim: int):
        super(SelfAttentionPooling, self).__init__()
        self.query = torch.nn.Parameter(torch.randn(input_dim))

    def forward(self, x):
        attn_scores = torch.einsum('sbi,i->sb', x, self.query)
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=0)
        attn_weights = attn_weights.unsqueeze(-1)
        return torch.sum(x * attn_weights, dim=0)

class CaiNet(torch.nn.Module):

    def __init__(self, classes: int):
        super(CaiNet, self).__init__()
        self.in_channels = 16
        self.conv = torch.nn.Conv2d(1, 16, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn = torch.nn.BatchNorm2d(16)
        self.relu = torch.nn.ReLU(inplace=True)

        self.res1 = ResLayer(16, 4, 3, 1)
        self.res2 = ResLayer(16, 8, 4, 2)
        self.res3 = ResLayer(32, 16, 6, 2)
        self.res4 = ResLayer(64, 32, 3, 2)

        self.pool = torch.nn.AdaptiveAvgPool2d((1, None))
        self.lstm = torch.nn.LSTM(128, 128, bidirectional=True)

        self.attention = SelfAttentionPooling(input_dim=256)
        self.fc = torch.nn.Linear(256, classes)

    def forward(self, x):

        # input transform
        c1out = self.relu(self.bn(self.conv(x)))

        # through resnet layers
        resout = self.res1(c1out)
        resout = self.res2(resout)
        resout = self.res3(resout)
        resout = self.res4(resout)

        # bi-lstm
        lstmin = self.pool(resout)
        lstmin = lstmin.squeeze(2)
        lstmin = lstmin.permute(2, 0, 1)
        lstmout, _ = self.lstm(lstmin)

        # attention and classification
        attn = self.attention(lstmout)
        pred = self.fc(attn)

        return pred
