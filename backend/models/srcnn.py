import torch
import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    def __init__(self, scale_factor=4):
        super(SRCNN, self).__init__()
        self.scale_factor = scale_factor
        
        # Feature extraction layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        
        # Non-linear mapping layer
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        
        # Reconstruction layer
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Upscale the input image using bicubic interpolation
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
        
        # Apply the SRCNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        
        return x 