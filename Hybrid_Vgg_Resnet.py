import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, vgg16, resnet18


class HybridModel(nn.Module):
    def __init__(self, num_classes=10):
        super(HybridModel, self).__init__()

        # Load pre-trained models
        self.vgg19 = vgg19(pretrained = True)
        self.vgg16 = vgg16(pretrained=True)
        self.resnet18 = resnet18(pretrained=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 3, num_classes)  # Adjust the input size of the fully connected layer accordingly

    def forward(self, x):
        x1 = self.vgg16.features(x)
        x1 = self.avgpool(x1)
        x1 = torch.flatten(x1, 1)

        x2 = self.vgg19.features(x)
        x2 = self.avgpool(x2)
        x2 = torch.flatten(x2, 1)

        x3 = self.resnet18.conv1(x)
        x3 = self.resnet18.bn1(x3)
        x3 = self.resnet18.relu(x3)
        x3 = self.resnet18.maxpool(x3)

        x3 = self.resnet18.layer1(x3)
        x3 = self.resnet18.layer2(x3)
        x3 = self.resnet18.layer3(x3)
        x3 = self.resnet18.layer4(x3)

        x3 = self.avgpool(x3)
        x3 = torch.flatten(x3, 1)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.fc(x)
        return x
