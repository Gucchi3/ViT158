import torch 
import torch.nn as nn


class tiny_cnn(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.conv0 = nn.Conv2d(3, 12, kernel_size=3, stride=2, padding=1)
        self.bn0   = nn.BatchNorm2d(12)
        self.relu0 = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(24)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(24, 36, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(36)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(36, 48, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(48)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(48, 56, kernel_size=3, stride=2, padding=1)
        self.bn4   = nn.BatchNorm2d(56)
        self.relu4 = nn.ReLU(inplace=True)
        
        # input 32x32x3 ---->>> 4x4x36
        self.fc    = nn.Linear(7*7*56, 10)
        
    def forward(self, x):
        x = self.relu0(self.bn0(self.conv0(x)))
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        
        x = torch.flatten(x, 1)
        
        out = self.fc(x)
        
        return out
        
        


if __name__ == "__main__":
    from torchinfo import summary
    summary(tiny_cnn(), input_size=(1,3,224,224))
