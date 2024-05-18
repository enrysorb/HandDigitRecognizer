import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import accuracy_score

# Definizione delle trasformazioni
transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
])

# Caricamento del dataset
hand_dataset = datasets.ImageFolder(root='hand_cropped', transform=transform)

# Divisione del dataset in training e test set
train_size = int(0.8 * len(hand_dataset))
test_size = len(hand_dataset) - train_size
train_dataset, test_dataset = random_split(hand_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Classe per calcolare la media
class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.num = 0
    
    def add(self, value, num):
        self.sum += value * num
        self.num += num
    
    def value(self):
        try:
            return self.sum / self.num
        except ZeroDivisionError:
            return None

# Parametri di addestramento
learning_rate = 0.001
epochs = 50
momentum = 0.95

# Funzione di addestramento
def train(model, train_loader, test_loader, lr=learning_rate, epochs=epochs, momentum=momentum, logdir="logs"):
    criterion = nn.CrossEntropyLoss()  # Funzione di perdita
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    global_step = 0
    for e in range(epochs):
        print(f"Epoch {e+1}/{epochs}")
        for mode in ["train", "test"]:
            loss_meter.reset()
            acc_meter.reset()

            model.train() if mode == "train" else model.eval()
            loader = train_loader if mode == "train" else test_loader

            with torch.set_grad_enabled(mode == "train"):
                for i, (inputs, labels) in enumerate(loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if mode == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    _, predicted = torch.max(outputs, 1)
                    acc = accuracy_score(labels.cpu(), predicted.cpu())
                    loss_meter.add(loss.item(), inputs.size(0))
                    acc_meter.add(acc, inputs.size(0))

                    print(f"{mode.capitalize()} Loss: {loss_meter.value()}, Accuracy: {acc_meter.value()}")
    
            torch.save(model.state_dict(), f"weights/{model.__class__.__name__}-{e+1}.pth")
    return model

# Modello CNN con un layer convoluzionale aggiunto
class HandModel(nn.Module):
    def __init__(self, out_classes):
        super(HandModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Layer convoluzionale
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Layer di pooling
        self.flatten = nn.Flatten()
        self.hidden_layer = nn.Linear(32 * 200 * 200, 128)  # Layer completamente connesso
        self.activation = nn.ReLU()  # Funzione di attivazione
        self.output_layer = nn.Linear(128, out_classes)  # Layer di output

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Passaggio attraverso il layer convoluzionale e di pooling
        x = self.flatten(x)  # Flattening
        hidden_representation = self.hidden_layer(x)  # Layer completamente connesso
        hidden_representation = self.activation(hidden_representation)  # Funzione di attivazione
        scores = self.output_layer(hidden_representation)  # Layer di output
        return scores

number_of_classes = 6
hand_classifier = HandModel(number_of_classes)
hand_classifier = train(hand_classifier, train_loader, test_loader, lr=learning_rate, epochs=epochs, momentum=momentum, logdir="logs")

print("Model trained successfully!")
