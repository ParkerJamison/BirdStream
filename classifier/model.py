import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torchvision
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt

from torch.nn.modules.loss import BCEWithLogitsLoss


import cv2 as cv
import os

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def createLoaders():
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),  # make all images 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    dataSet = datasets.ImageFolder("./Images", transform=transform)
    print(type(dataSet))
    length = len(dataSet)

    trainLength = int(length * 0.8)
    testLength = int(length * 0.1)
    valLength = length - trainLength - testLength

    trainSet, valSet, testSet = torch.utils.data.random_split(dataSet, [trainLength, valLength, testLength])

    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=32, shuffle=True)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=32, shuffle=True)
    valLoader = torch.utils.data.DataLoader(valSet, batch_size=32, shuffle=True)

    # get some random training images
    dataiter = iter(testLoader)
    images, _ = next(dataiter)
    # show images
    #imshow(torchvision.utils.make_grid(images))

    return trainLoader, valLoader, testLoader




def main():

    trainLoader, valLoader, testLoader = createLoaders()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained ResNet-50
    model = models.resnet50(pretrained=True)
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    nr_filters = model.fc.in_features  #number of input features of last layer
    model.fc = nn.Linear(nr_filters, 1)

    model.to(device)

    lossFn = BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.01)


    epochs = 20
    print_every = 10
    train_losses, test_losses, accuracies = [], [], []

    for epoch in range(epochs):
        running_loss = 0
        steps = 0
        model.train()
        for inputs, labels in trainLoader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            logps = model(inputs)
            labels = labels.unsqueeze(1).float()

            loss = lossFn(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testLoader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model(inputs)
                        labels = labels.unsqueeze(1).float()
                        test_loss += lossFn(logps, labels).item()
                        # Binary accuracy calculation
                        probs = torch.sigmoid(logps)
                        preds = (probs > 0.5).long()
                        equals = preds == labels.long()
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                train_losses.append(running_loss / print_every)
                test_losses.append(test_loss / len(testLoader))
                accuracies.append(accuracy / len(testLoader))

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Step {steps}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(testLoader):.3f}.. "
                      f"Test accuracy: {accuracy/len(testLoader):.3f}")
                running_loss = 0
                model.train()

# Save model
    torch.save(model, 'birdModel.pth')





if __name__ == '__main__':
    main()

