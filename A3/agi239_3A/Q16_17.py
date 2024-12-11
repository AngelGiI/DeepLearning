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
import tarfile, sys

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.Grayscale(num_output_channels=1)])

trainset0 = datasets.ImageFolder('mnist-varres/train', transform=transform)
testset = datasets.ImageFolder('mnist-varres/test', transform=transform)

# Divide data in the three resolutions

train_res_1 = []
train_res_2 = []
train_res_3 = []

test_res_1 = []
test_res_2 = []
test_res_3 = []

for item in trainset0: 
    if item[0].shape[1] == 32: train_res_1.append(item)
    elif item[0].shape[1] == 48: train_res_2.append(item)
    elif item[0].shape[1] == 64: train_res_3.append(item)
        
for item in testset: 
    if item[0].shape[1] == 32: test_res_1.append(item)
    elif item[0].shape[1] == 48: test_res_2.append(item)
    elif item[0].shape[1] == 64: test_res_3.append(item)

batch_size = 8

# Also create a validation set 

trainset_1, valset_1 = train_test_split(train_res_1, test_size=0.9, random_state=42)
trainset_2, valset_2 = train_test_split(train_res_2, test_size=0.9, random_state=42)
trainset_3, valset_3 = train_test_split(train_res_3, test_size=0.9, random_state=42)

# Create data loaders for each resolution

trainloader_1 = torch.utils.data.DataLoader(trainset_1, batch_size=batch_size, shuffle=True, num_workers=2)
trainloader_2 = torch.utils.data.DataLoader(trainset_2, batch_size=batch_size, shuffle=True, num_workers=2)
trainloader_3 = torch.utils.data.DataLoader(trainset_3, batch_size=batch_size, shuffle=True, num_workers=2)

valloader_1 = torch.utils.data.DataLoader(valset_1, batch_size=batch_size, shuffle=True, num_workers=2)
valloader_2 = torch.utils.data.DataLoader(valset_2, batch_size=batch_size, shuffle=True, num_workers=2)
valloader_3 = torch.utils.data.DataLoader(valset_3, batch_size=batch_size, shuffle=True, num_workers=2)

testloader_1 = torch.utils.data.DataLoader(test_res_1, batch_size=batch_size, shuffle=True, num_workers=2)
testloader_2 = torch.utils.data.DataLoader(test_res_2, batch_size=batch_size, shuffle=True, num_workers=2)
testloader_3 = torch.utils.data.DataLoader(test_res_3, batch_size=batch_size, shuffle=True, num_workers=2)

# define the model
N = 81

class Network(nn.Module):

    def __init__(self, mean_pooling):
        
        self.mean_pooling = mean_pooling
        
        super(Network, self).__init__()

        self.conv_neural_network_layers = nn.Sequential(
                
                # first conv layer
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, stride=1),                
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            
                # second conv layer
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),

                # second conv layer
                nn.Conv2d(in_channels=32, out_channels=N, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2))

        #linear layer
        self.linear_layers = nn.Sequential(
                nn.Linear(N, 10))

    # Defining the forward pass 
    def forward(self, x):
        x = self.conv_neural_network_layers(x)

        #global max pooling
        if self.mean_pooling:
            x = torch.flatten(F.adaptive_avg_pool2d(x, (1, 1)), 1)
        
        else:
            x = torch.flatten(F.adaptive_max_pool2d(x, (1, 1)), 1)
        
        #linear layer
        x = self.linear_layers(x)
        
        return x 

def train(dataloader, model, loss_fn, optimizer):
    for i in dataloader:
        size = len(i.dataset)
        model.train()
        for batch, (X, y) in enumerate(i):
            y_hot = F.one_hot(y, 10)
            y_hot = torch.zeros(X.shape[0], 10)
            y_hot[range(y_hot.shape[0]), y]=1      

            X, y_hot = X.to(device), y_hot.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y_hot)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    return loss

def test(dataloader, model, loss_fn):
    for i in dataloader:
        size = len(i.dataset)
        model.eval()
        test_loss, correct = 0, 0

        for batch, (X, y) in enumerate(i):
            y_hot = F.one_hot(y, 10)
            y_hot = torch.zeros(X.shape[0], 10)
            y_hot[range(y_hot.shape[0]), y]=1      

            X, y_hot = X.to(device), y_hot.to(device)

            # Compute prediction error
            pred = model(X)
            test_loss += loss_fn(pred, y_hot).item()
            correct += (pred.argmax(axis=1) == y_hot.argmax(axis=1)).type(torch.float).sum().item()

        test_loss /= 10000
        correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, 100*correct


epochs = 5

model_mean_pooling = Network(mean_pooling=False) # mean_pooling=True for global mean pooling; mean_pooling=False for global max pooling
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_fn = nn.BCEWithLogitsLoss() #nn.BCELoss() 
optimizer = torch.optim.Adam(model_mean_pooling.parameters(), lr=3e-3)


train_res = [trainloader_1, trainloader_2, trainloader_3]
val_res = [valloader_1, valloader_2, valloader_3]
test_res = [testloader_1, testloader_2, testloader_3]


if __name__ == '__main__':
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 5
    test_loss_list = []
    test_acc = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_res, model_mean_pooling, loss_fn, optimizer)
        test_loss, acc = test(val_res, model_mean_pooling, loss_fn)
        test_loss_list.append(test_loss)
        test_acc.append(acc)
    print("-------------------------------\n Training done. Computing test set")
    test_loss, acc = test(test_res, model_mean_pooling, loss_fn)
    test_loss_list.append(test_loss)
    print("Val loss: ", test_loss_list)
