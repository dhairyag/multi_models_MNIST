import torch.nn as nn
import torch.nn.functional as F

###################
# Neural Network Architecture 1
# without dropout
###################

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),    
            nn.BatchNorm2d(8),                
            nn.LeakyReLU(),                    
            nn.Conv2d(8, 8, 3, padding=1, bias=False),     
            nn.BatchNorm2d(8),                
            nn.LeakyReLU(),                    
            nn.MaxPool2d(2, 2),               
        )                                      
        # Receptive Field Calculation:
        # Layer 1: Conv2d(1, 8, 3, padding=1) -> RF = 3, Stride = 1
        # Layer 2: Conv2d(8, 8, 3, padding=1) -> RF = 5, Stride = 1
        # MaxPool2d(2, 2) -> RF = 5 + (2-1)*1 = 6, Stride = 2

        # Second Convolutional Block
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 12, 3, padding=1, bias=False),     
            nn.BatchNorm2d(12),                
            nn.LeakyReLU(),                    
            nn.Conv2d(12, 12, 3, padding=1, bias=False),     
            nn.BatchNorm2d(12),                
            nn.LeakyReLU(),                    
            nn.MaxPool2d(2, 2),               
        )                                      
        # Receptive Field Calculation:
        # Layer 1: Conv2d(8, 12, 3, padding=1) -> RF = 6 + 4 = 10
        # Layer 2: Conv2d(12, 12, 3, padding=1) -> RF = 10 + 4 = 14
        # MaxPool2d(2, 2) -> RF = 14 + (2-1)*2 = 16

        # Third Convolutional Block
        self.conv3 = nn.Sequential(
            nn.Conv2d(12, 16, 3, padding=1, bias=False),    
            nn.BatchNorm2d(16),                
            nn.LeakyReLU(),                    
            nn.Conv2d(16, 16, 3, padding=1, bias=False),   
            nn.BatchNorm2d(16),                
            nn.LeakyReLU(),                    
            nn.MaxPool2d(2, 2),               
        )                                      
        # Receptive Field Calculation:
        # Layer 1: Conv2d(12, 16, 3, padding=1) -> RF = 16 + 8 = 24
        # Layer 2: Conv2d(16, 16, 3, padding=1) -> RF = 24 + 8 = 32
        # MaxPool2d(2, 2) -> RF = 32 + 4 = 36

        # Global Average Pooling Layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Output size of 1x1 for each feature map
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_avg_pool(x)  # Apply global average pooling
        x = x.view(-1, 16)  # Flatten to (batch_size, num_classes)
        return F.log_softmax(x, dim=1)

###################
# Neural Network Architecture 2
# without dropout
###################

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        # Input Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
        ) # output_size = 26
        # Receptive Field Calculation:
        # Layer 1: Conv2d(1, 12, 3, padding=0) -> RF = 3

        # CONVOLUTION BLOCK 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(12, 16, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        ) # output_size = 24
        # Receptive Field Calculation:
        # Layer 2: Conv2d(12, 16, 3, padding=0) -> RF = 3 + 2 = 5

        # TRANSITION BLOCK 1
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12
        # Receptive Field Calculation:
        # Layer 3: Conv2d(16, 10, 1, padding=0) -> RF = 5 + 0 = 5
        # MaxPool2d(2, 2) -> RF = 5 + 2 = 7

        # CONVOLUTION BLOCK 2
        self.conv4 = nn.Sequential(
            nn.Conv2d(10, 12, 3, padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(12),
        ) # output_size = 10
        # Receptive Field Calculation:
        # Layer 4: Conv2d(10, 12, 3, padding=0) -> RF = 7 + 4 = 11
        self.conv5 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(12),
        ) # output_size = 8
        # Receptive Field Calculation:
        # Layer 5: Conv2d(12, 12, 3, padding=0) -> RF = 11 + 4 = 15
        self.conv6 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(12),
        ) # output_size = 6
        # Receptive Field Calculation:
        # Layer 6: Conv2d(12, 12, 3, padding=0) -> RF = 15 + 4 = 19
        self.conv7 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(12),
        ) # output_size = 6
        # Receptive Field Calculation:
        # Layer 7: Conv2d(12, 12, 3, padding=1) -> RF = 19 + 4 = 23
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1
        # Receptive Field Calculation:
        # AvgPool2d(kernel_size=6) -> RF = 23 + 5 = 28

        self.conv8 = nn.Sequential(
            nn.Conv2d(12, 10, 1, padding=0, bias=False),
        ) 
        # Receptive Field Calculation:
        # Conv2d(12, 10, 1, padding=0) -> RF = 28 + 0 = 28

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.gap(x)        
        x = self.conv8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

###################
# Neural Network Architecture 3
# with dropout
###################

class Model3(nn.Module):
    def __init__(self, dropout_value):
        super(Model3, self).__init__()
        # Input Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 26
        # Receptive Field Calculation:
        # Layer 1: Conv2d(1, 12, 3, padding=0) -> RF = 3

        # CONVOLUTION BLOCK 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(12, 16, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 24
        # Receptive Field Calculation:
        # Layer 2: Conv2d(12, 16, 3, padding=0) -> RF = 3 + 2 = 5

        # TRANSITION BLOCK 1
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12
        # Receptive Field Calculation:
        # Layer 3: Conv2d(16, 10, 1, padding=0) -> RF = 5 + 0 = 5
        # MaxPool2d(2, 2) -> RF = 5 + 2 = 7

        # CONVOLUTION BLOCK 2
        self.conv4 = nn.Sequential(
            nn.Conv2d(10, 12, 3, padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 10
        # Receptive Field Calculation:
        # Layer 4: Conv2d(10, 12, 3, padding=0) -> RF = 7 + 4 = 11
        self.conv5 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 8
        # Receptive Field Calculation:
        # Layer 5: Conv2d(12, 12, 3, padding=0) -> RF = 11 + 4 = 15
        self.conv6 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 6
        # Receptive Field Calculation:
        # Layer 6: Conv2d(12, 12, 3, padding=0) -> RF = 15 + 4 = 19
        self.conv7 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 6
        # Receptive Field Calculation:
        # Layer 7: Conv2d(12, 12, 3, padding=1) -> RF = 19 + 4 = 23
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1
        # Receptive Field Calculation:
        # AvgPool2d(kernel_size=6) -> RF = 23 + 5 = 28

        self.conv8 = nn.Sequential(
            nn.Conv2d(12, 10, 1, padding=0, bias=False),
        ) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.gap(x)        
        x = self.conv8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)