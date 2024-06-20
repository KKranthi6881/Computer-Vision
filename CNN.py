import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the custom CNN architecture
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # Fully connected layer
        self.fc2 = nn.Linear(256, 128)  # Another fully connected layer
        self.fc3 = nn.Linear(128, 10)  # Final output layer for 10 classes
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Output: (batch_size, 32, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))  # Output: (batch_size, 64, 8, 8)
        x = self.pool(F.relu(self.conv3(x)))  # Output: (batch_size, 128, 4, 4)
        x = x.view(-1, 128 * 4 * 4)  # Flatten the tensor to (batch_size, 2048)
        x = F.relu(self.fc1(x))  # Apply ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc2(x))  # Apply ReLU activation
        x = self.fc3(x)  # Output layer
        return x







