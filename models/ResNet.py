import torch
import torch.nn as nn
import torch.nn.functional as F

class BuildingBlock(nn.Module):
    def __init__(self, in_channels, med_channels, out_channels):
        super().__init__()
        self.m_1 = nn.Conv2d(in_channels, med_channels, kernel_size=3, stride=1, padding=1)
        self.m_2 = nn.Conv2d(med_channels, out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        out = self.m_1(x)
        out = F.relu(out)
        out = self.m_2(out)

        return out
    

class ResNet18(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        
        self.resblock2_1 = BuildingBlock(in_channels=64, med_channels=64, out_channels=64)
        self.resblock2_2 = BuildingBlock(in_channels=64, med_channels=64, out_channels=64)
        self.resblock3_1 = BuildingBlock(in_channels=64, med_channels=128, out_channels=128)
        self.resblock3_2 = BuildingBlock(in_channels=128, med_channels=128, out_channels=128)
        self.resblock4_1 = BuildingBlock(in_channels=128, med_channels=256, out_channels=256)
        self.resblock4_2 = BuildingBlock(in_channels=256, med_channels=256, out_channels=256)
        self.resblock5_1 = BuildingBlock(in_channels=256, med_channels=512, out_channels=512)
        self.resblock5_2 = BuildingBlock(in_channels=512, med_channels=512, out_channels=512)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

        self.shortcut3 = nn.Conv2d(64, 128, kernel_size=1, stride=1)
        self.shortcut4 = nn.Conv2d(128, 256, kernel_size=1, stride=1)
        self.shortcut5 = nn.Conv2d(256, 512, kernel_size=1, stride=1)
    
    def forward(self, x):

        # conv1
        out = self.conv1(x)
        out = F.relu(out) 
      
        
        # conv2_x 
        out = self.resblock2_1(out) + out
        out = F.relu(out)
        out = self.resblock2_2(out) + out
        out = F.relu(out)
        
        # conv3_x
        out = self.resblock3_1(out) + self.shortcut3(out)
        out = F.relu(out)
        out = self.resblock3_2(out) + out
        out = F.relu(out)
        
        #conv4_x
        out = self.resblock4_1(out) + self.shortcut4(out)
        out = F.relu(out)
        out = self.resblock4_2(out) + out
        out = F.relu(out)

        # conv5_x
        out = self.resblock5_1(out) + self.shortcut5(out)
        out = F.relu(out)
        out = self.resblock5_2(out) + out
        out = F.relu(out)

        out = self.avgpool(out)
        out = F.relu(out)
        out = self.fc(out.flatten(out, 1))
    
        return out
    

    