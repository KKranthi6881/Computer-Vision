import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from CNN import CustomCNN  # Import the CustomCNN class


class CIFAR10DataSet:
    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                         transform=self.transform)
        self.trainloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                        transform=self.transform)
        self.testloader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers = 2)


