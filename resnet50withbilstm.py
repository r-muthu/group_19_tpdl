import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

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

# ResNet + BiLSTM + Attention Model
class ResNet50BiLSTMAttention(nn.Module):
    def __init__(self, classes=10, lstm_hidden=128, lstm_layers=2, attention_size=256):
        super(ResNet50BiLSTMAttention, self).__init__()
        
        # Load Pretrained ResNet-50
        self.resnet = models.resnet50(pretrained=True)
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
        print("Mode:", "train" if self.training else "eval", "Shape:", x.shape)
        # Pass through ResNet
        resnet_out = self.resnet(x)  # Output shape: [batch_size, channels, height, width]
        
        # Print the shape of resnet_out to debug
        print(f"ResNet Output Shape: {resnet_out.shape}")  # Debugging the shape
        
        # Ensure ResNet output shape is correct
        # Expected shape after avgpool: [batch_size, 2048, 8, 8]
        resnet_out = resnet_out.view(resnet_out.size(0), 2048, -1).permute(0, 2, 1)  # Flatten spatial dimensions and permute
        
        # Print the shape after reshaping
        print(f"ResNet Output Shape after reshaping: {resnet_out.shape}")  # Debugging after reshaping
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(resnet_out)  # Output shape: [batch_size, seq_len, hidden*2]
        
        # Apply Attention Mechanism
        attn_weights = self.attention(lstm_out)
        attn_out = torch.sum(attn_weights * lstm_out, dim=1)  # Weighted sum of LSTM outputs
        
        # Output Layer
        out = self.fc_out(attn_out)  # Final classification output
        
        return out