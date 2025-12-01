import torch
import torch.nn as nn
import torchvision.models as models


def load_teacher_resnet18(num_classes=10):
    teacher = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    in_features = teacher.fc.in_features
    teacher.fc = nn.Linear(in_features, num_classes)

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    return teacher