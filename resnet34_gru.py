import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic ResNet Block
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# ResNet-34 Backbone
class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 3, 4, 6, 3 blocks for ResNet-34
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 7))  # Keep temporal info in width

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):  # [B, 3, H, W]
        x = self.relu(self.bn1(self.conv1(x)))  # [B, 64, H/2, W/2]
        x = self.maxpool(x)                     # [B, 64, H/4, W/4]
        x = self.layer1(x)                      # [B, 64, H/4, W/4]
        x = self.layer2(x)                      # [B, 128, H/8, W/8]
        x = self.layer3(x)                      # [B, 256, H/16, W/16]
        x = self.layer4(x)                      # [B, 512, H/32, W/32]
        x = self.avgpool(x)                     # [B, 512, 1, 7]
        return x


# Attention-based GRU Model
class ResNetGRUModel(nn.Module):
    def __init__(self, hidden_dim=512, classes=2, num_layers=1, bidirectional=True):
        super(ResNetGRUModel, self).__init__()
        self.resnet = ResNet34()

        self.gru = nn.GRU(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.attn = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.attn_combine = nn.Linear(hidden_dim * 2, 1)

        self.fc_out = nn.Linear(hidden_dim * 2, classes)

    def attention(self, gru_out):
        attn_weights = torch.tanh(self.attn(gru_out))                  # [B, T, H*2]
        attn_weights = self.attn_combine(attn_weights)                # [B, T, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)             # [B, T, 1]
        return attn_weights

    def forward(self, x):  # [B, 3, H, W]
        print("Mode:", "train" if self.training else "eval", "Shape:", x.shape)
        resnet_out = self.resnet(x)                                   # [B, 512, 1, 7]
        resnet_out = resnet_out.squeeze(2).permute(0, 2, 1)           # [B, 7, 512]

        gru_out, _ = self.gru(resnet_out)                             # [B, 7, H*2]
        attn_weights = self.attention(gru_out)                        # [B, 7, 1]
        attn_applied = torch.sum(attn_weights * gru_out, dim=1)       # [B, H*2]

        out = self.fc_out(attn_applied)                               # [B, output_dim]
        return out
