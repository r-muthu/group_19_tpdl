import torch
import torch.nn as nn
import torchvision.models as models
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
    
# Attention mechanism
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, lstm_output):
        # lstm_output: [batch, seq_len, hidden*2]
        energy = torch.tanh(self.attn(lstm_output))               # [batch, seq_len, hidden]
        energy = torch.matmul(energy, self.v)                     # [batch, seq_len]
        attn_weights = F.softmax(energy, dim=1).unsqueeze(1)      # [batch, 1, seq_len]
        context = torch.bmm(attn_weights, lstm_output)            # [batch, 1, hidden*2]
        context = context.squeeze(1)                              # [batch, hidden*2]
        return context
    
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
# Custom ResNet-50 + BiLSTM Model for Spectrogram Classification
class ResNet50(nn.Module):
    def __init__(self, classes=10):
        super(ResNet50, self).__init__()
        self.classes = classes
        # Load Pretrained ResNet-50 and Remove Fully Connected Layer
        self.resnet = models.resnet50(pretrained=True)
        
        # Modify the first convolution layer to accept 1 channel instead of 3
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Modify Global Average Pooling to (4, 8)
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((4, 8))

        # Fully Connected Layer
        self.resnet.fc = nn.Linear(2048*4*8, classes)
        
    def forward(self, x):
        x = self.resnet(x)
        return x

# ResNet + BiLSTM + Attention Model
class ResNet50BiLSTMAttention(nn.Module):
    def __init__(self, classes=10, lstm_hidden=128, lstm_layers=2, attention_size=256):
        super(ResNet50BiLSTMAttention, self).__init__()
        
        # Load Pretrained ResNet-50
        self.resnet = models.resnet50(pretrained=True)
        # Modify the first layer to accept single-channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Modify the avgpool to a smaller spatial output to retain more spatial features
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 7))  # Maintain a bigger spatial output
        
        # Remove the final fully connected layer
        self.resnet.fc = nn.Identity()  # This avoids having a classification head on ResNet

        # BiLSTM Layer
        self.lstm = nn.LSTM(input_size=2048, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        
        # Attention Layer
        self.attn = nn.Linear(lstm_hidden * 2, attention_size)
        self.attn_combine = nn.Linear(attention_size, 1)
        
        # Output Layer
        self.fc_out = nn.Linear(lstm_hidden * 2, classes)
    
    def attention(self, lstm_out):
        attn_weights = torch.tanh(self.attn(lstm_out))  # Apply tanh to each hidden state
        attn_weights = self.attn_combine(attn_weights)  # Combine the weights
        attn_weights = torch.softmax(attn_weights, dim=1)  # Normalize the weights using softmax
        return attn_weights
    
    def forward(self, x):
        # Pass through ResNet
        resnet_out = x
        for name, layer in self.resnet.named_children():
            resnet_out = layer(resnet_out)
            if name == 'avgpool':  # Stop when we reach avgpool layer
                break
        
        # Ensure ResNet output shape is correct
        # Reshape to [B, 7, 2048] for LSTM
        resnet_out = resnet_out.squeeze(2).permute(0, 2, 1)  # [B, 7, 2048]
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(resnet_out)  # Output shape: [batch_size, seq_len, hidden*2]
        
        # Apply Attention Mechanism
        attn_weights = self.attention(lstm_out)
        attn_out = torch.sum(attn_weights * lstm_out, dim=1)  # Weighted sum of LSTM outputs
    
        # Output Layer
        out = self.fc_out(attn_out)  # Final classification output
        
        return out
    
# ResNet34 + BiLSTM + Attention Model
class ResNet34BiLSTMAttention(nn.Module):
    def __init__(self, classes=10, lstm_hidden=128, lstm_layers=2, attention_size=256):
        super(ResNet34BiLSTMAttention, self).__init__()
        
        # Load Pretrained ResNet-50
        self.resnet = models.resnet34(pretrained=True)
        # Modify the first layer to accept single-channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Modify the avgpool to a smaller spatial output to retain more spatial features
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((8, 8))  # Maintain a bigger spatial output
        
        # Remove the final fully connected layer
        self.resnet.fc = nn.Identity()  # This avoids having a classification head on ResNet

        # BiLSTM Layer
        self.lstm = nn.LSTM(input_size=2048, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        
        # Attention Layer
        self.attn = nn.Linear(lstm_hidden * 2, attention_size)
        self.attn_combine = nn.Linear(attention_size, 1)
        
        # Output Layer
        self.fc_out = nn.Linear(lstm_hidden * 2, classes)
    
    def attention(self, lstm_out):
        attn_weights = torch.tanh(self.attn(lstm_out))  # Apply tanh to each hidden state
        attn_weights = self.attn_combine(attn_weights)  # Combine the weights
        attn_weights = torch.softmax(attn_weights, dim=1)  # Normalize the weights using softmax
        return attn_weights
    
    def forward(self, x):
        # Pass through ResNet
        resnet_out = x
        for name, layer in self.resnet.named_children():
            resnet_out = layer(resnet_out)
            if name == 'avgpool':  # Stop when we reach avgpool layer
                break

        resnet_out = resnet_out.view(resnet_out.size(0), 2048, -1).permute(0, 2, 1)  # Flatten spatial dimensions and permute
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(resnet_out)  # Output shape: [batch_size, seq_len, hidden*2]
        
        # Apply Attention Mechanism
        attn_weights = self.attention(lstm_out)
        attn_out = torch.sum(attn_weights * lstm_out, dim=1)  # Weighted sum of LSTM outputs
        
        # Output Layer
        out = self.fc_out(attn_out)  # Final classification output
        
        return out