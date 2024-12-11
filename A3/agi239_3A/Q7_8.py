import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
import os
from torchvision import datasets, transforms
from torch import optim, nn, unsqueeze
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Lambda, Compose
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

torch.manual_seed(77)

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))])

train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = transform, 
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = transform,
    download = True
)
# Split the training data into 50 000 training instances and 10 000 validation instances
traindata, valdata = train_test_split(train_data, test_size=10000, random_state=42)
batch_size = 16
trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valdata, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)

# Define the model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,16,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2),)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32,64,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2),)
        self.out = nn.Linear(64*3*3, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1) # Flatten , same as x = torch.flatten(x, 1)
        output = self.out(x)
        return output

model = CNN()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Use cross-entropy as the loss function, and Adam as the optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(dataloader, model, loss_func, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        y_hot = F.one_hot(y, 10)
        y_hot = y_hot.float()
        # y_hot = torch.zeros(batch_size, 10)
        # y_hot[range(y_hot.shape[0]), y]=1      


        X, y_hot = X.to(device), y_hot.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_func(pred, y_hot)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss


def test(dataloader, model, loss_func):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        y_hot = F.one_hot(y, 10)
        y_hot = y_hot.float()
        # y_hot = torch.zeros(batch_size, 10)
        # y_hot[range(y_hot.shape[0]), y]=1     
        X, y_hot = X.to(device), y_hot.to(device) 
        # Compute prediction error
        pred = model(X)
        test_loss += loss_func(pred, y_hot).item()
        correct += (pred.argmax(axis=1) == y_hot.argmax(axis=1)).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Average loss: {test_loss:>8f} \n")
    return test_loss, correct*100

if __name__ == '__main__':
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 10
    test_loss_list = []
    test_acc = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(trainloader, model, loss_func, optimizer)
        test_loss, acc = test(testloader, model, loss_func)
        test_loss_list.append(test_loss)
        test_acc.append(acc)
    print("Test loss: ", test_loss_list)
    print("Test accuracy: ", test_acc)
