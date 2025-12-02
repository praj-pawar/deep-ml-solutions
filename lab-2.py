import torch
import torch.nn as nn

def build_model() -> nn.Module:
    class TinyNetV2(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Block 1: Conv -> BN -> ReLU -> Pool
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False) # No bias needed with BN!
            self.bn1 = nn.BatchNorm2d(8) # New Layer
            
            # Block 2: Conv -> BN -> ReLU -> Pool
            # 22 channels to save budget for BN
            self.conv2 = nn.Conv2d(8, 22, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(22) # New Layer
            
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(22, 10)
            
            self.act = nn.ReLU()
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            # Block 1
            x = self.conv1(x)
            x = self.bn1(x)  # Normalize
            x = self.act(x)
            x = self.pool(x)
            
            # Block 2
            x = self.conv2(x)
            x = self.bn2(x)  # Normalize
            x = self.act(x)
            x = self.pool(x)
            
            # Output
            x = self.gap(x)
            x = x.flatten(1)
            return self.fc(x)

    return TinyNetV2()