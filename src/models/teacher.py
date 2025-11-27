import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class TeacherModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.resnet = resnet18(weights=weights)
        else:
            self.resnet = resnet18(weights=None)
        
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.resnet(x)
    
    def extract_features(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        x = self.resnet.avgpool(x)
        features = torch.flatten(x, 1)
        
        return features
