import torch
import torch.nn as nn
import torchvision.models as models

""" Original ResNet-50 Implementation (Commented Out)
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
"""

# Custom ResNet-50 + BiLSTM Model for Spectrogram Classification
class ResNetLSTM(nn.Module):
    def __init__(self, num_classes=10, lstm_hidden=128, lstm_layers=2):
        super(ResNetLSTM, self).__init__()
        
        # Load Pretrained ResNet-50 and Remove Fully Connected Layer
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove FC layer
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=2048, hidden_size=lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True)
        
        # Fully Connected Layer
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape  # (B, T, C, H, W)
        x = x.view(batch_size * seq_len, C, H, W)  # Merge batch & seq for ResNet
        x = self.resnet(x)  # ResNet Feature Extraction
        x = x.view(batch_size, seq_len, -1)  # Restore sequence structure
        
        x, _ = self.lstm(x)  # LSTM Processing
        x = x[:, -1, :]  # Take last LSTM output
        x = self.fc(x)  # Classification
        return x

    """def load_custom_weights(self, weight_path="resnet50.data"):
        self.load_state_dict(torch.load(weight_path))"""