# Instantiate the CIFAR-10 dataset

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg19
import numpy as np
from Hybrid_Vgg_Resnet import HybridModel
from CIFAR10 import CIFAR10DataSet
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt

def main():
    class Validation:
        # Instantiate the CIFAR-10 dataset
        dataset = CIFAR10DataSet()

        # Initialize the network, loss function, and optimizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = HybridModel().to(device)  # Instantiate the model and move it to the device

        # Ensure the model has parameters
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

        if len(list(model.parameters())) == 0:
            raise ValueError("Model has no parameters.")

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training and evaluation functions

        # Training function
        def train(model, trainloader, criterion, optimizer, device):
            model.train()
            running_loss = 0.0
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            return running_loss / len(trainloader)

        # Evaluation function
        def evaluate(model, testloader, criterion, device):
            model.eval()
            correct = 0
            total = 0
            test_loss = 0.0
            all_labels = []
            all_predictions = []
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

            accuracy = 100 * correct / total
            precision = precision_score(all_labels, all_predictions, average='macro')
            recall = recall_score(all_labels, all_predictions, average='macro')
            return test_loss / len(testloader), accuracy, precision, recall

        # Training loop
        num_epochs = 20
        train_losses = []
        test_losses = []
        test_accuracies = []
        test_precisions = []
        test_recalls = []

        for epoch in range(num_epochs):
            train_loss = train(model, dataset.trainloader, criterion, optimizer, device)
            test_loss, test_accuracy, precision, recall = evaluate(model, dataset.testloader, criterion, device)

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            test_precisions.append(precision)
            test_recalls.append(recall)

            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}')

            # Save the trained model
        torch.save(model.state_dict(), 'hybrid_cnn_cifar10.pth')

        # Plotting the results
        epochs = range(1, num_epochs + 1)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_losses, 'b', label='Training loss')
        plt.plot(epochs, test_losses, 'r', label='Validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(epochs, test_accuracies, 'b', label='Validation accuracy')
        plt.title('Validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(epochs, test_precisions, 'b', label='Validation precision')
        plt.plot(epochs, test_recalls, 'r', label='Validation recall')
        plt.title('Precision and Recall')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.legend()

        plt.tight_layout()
        plt.show()

            # Visualizing some predictions
        def visualize_predictions(model, testloader, device, num_images = 5):
            model.eval()
            images_shown = 0
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)

                    for i in range(inputs.size(0)):
                        if images_shown >= num_images:
                            break
                        image = inputs[i].cpu().numpy().transpose((1, 2, 0))
                        image = np.clip(image * 0.2023 + 0.4914, 0, 1)
                        label = labels[i].item()
                        pred = predicted[i].item()

                        plt.figure()
                        plt.imshow(image)
                        plt.title(f'True: {label}, Pred: {pred}')
                        plt.show()
                        images_shown += 1
                    if images_shown >= num_images:
                        break

            # Visualize predictions
        visualize_predictions(model, dataset.testloader, device)

if __name__ == '__main__':
    main()
