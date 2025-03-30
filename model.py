import torch.nn as nn
import torchvision.models as models

# Custom ResNet-50 + BiLSTM Model for Spectrogram Classification
class ResNet50(nn.Module):
    def __init__(self, num_classes=10, lstm_hidden=128, lstm_layers=2):
        super(ResNet50, self).__init__()
        self.num_classes = num_classes
        # Load Pretrained ResNet-50 and Remove Fully Connected Layer
        self.resnet = models.resnet50(pretrained=True)
        
        # Modify the first convolution layer to accept 1 channel instead of 3
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Modify Global Average Pooling to (4, 8)
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((4, 8))

        # Fully Connected Layer
        self.resnet.fc = nn.Linear(2048*4*8, num_classes)
        
    def forward(self, x):
        x = self.resnet(x)
        return x