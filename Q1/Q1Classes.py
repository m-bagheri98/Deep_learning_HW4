
import torch
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def one_hot_encode(smiles):
    vocabulary = ['.', '(', ')', '+', '-', '/', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '@', 'B', 'C', 'F', 'H', 'I','N', 'O', 'P', 'S', '[', ']', '\\', 'l', 'n', 'o', 'r', 's', 't']
    char_to_idx = {char: idx for idx, char in enumerate(vocabulary)}
    # Initialize an empty one-hot encoded tensor
    one_hot = torch.zeros(400, len(vocabulary), dtype=torch.float)

    # Convert each character to one-hot encoding
    for i, char in enumerate(smiles):
        if char in char_to_idx:
            one_hot[i, char_to_idx[char]] = 1.0

    one_hot = one_hot.view(400* len(vocabulary),1)

    return one_hot


class BBBPDataset(Dataset):
    def __init__(self, dataa,labell):
        self.dataa = dataa
        self.labell = labell

    def __len__(self):
        return len(self.dataa)

    def __getitem__(self, idx):
        labels = self.labell[idx]
        data = one_hot_encode(self.dataa[idx])
        return data, labels




class Fully_connected_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Fully_connected_net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def train_model(model, epoch_num, train_loader, optimizer, criterion, device):
  loss_epoch = []
  acc_epoch = []
  
  for epoch in range(epoch_num):  
    running_loss = 0.0
    acc = 0.0
    with tqdm(train_loader, unit="batch") as tepoch:
      for data in tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        inputs, labels = data
        labels = labels.float()
        #inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs.permute(0,2,1))
        loss = criterion(outputs.squeeze(), labels.squeeze())
        _, predicted = torch.max(outputs.data, 1)
        total = (labels.size(0))
        correct = (predicted.squeeze() == labels.squeeze()).sum().item()
        acc  += (100 * correct / total)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        tepoch.set_postfix(loss=loss.item())

      loss_epoch.append(running_loss/(len(train_loader)))
      acc_epoch.append(acc/(len(train_loader))) 
  return model



def test_model_fc(model, test_loader):
  correct = 0
  total = 0

  with torch.no_grad():
      for smiles, labels in test_loader:
          # Convert SMILES strings to one-hot encoded tensors

          labels = labels.float()

          # Forward pass
          outputs = model(smiles.permute(0,2,1))
          predicted = torch.round(outputs)

          total += labels.size(0)
          correct += (predicted.squeeze() == labels.squeeze()).sum().item()

  accuracy = 100 * correct / total
  print(f"Test Accuracy: {accuracy:.2f}%")
  return accuracy







class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        output, _ = self.lstm(x)
        flattened = output.contiguous().view(-1, self.hidden_size)
        out = self.relu(self.fc1(flattened))
        out = self.fc2(out)
        return out

def train_LSTM(model, epoch_num, train_loader, optimizer, criterion, device):
  loss_epoch = []
  acc_epoch = []
  
  for epoch in range(epoch_num):  
    running_loss = 0.0
    acc = 0.0
    with tqdm(train_loader, unit="batch") as tepoch:
      for data in tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        inputs, labels = data
        labels = labels.to(device)
        inputs = inputs.to(device)

        optimizer.zero_grad()

        outputs = model(inputs.permute(0,2,1))
        loss = criterion(outputs.squeeze(), labels.squeeze())
        _, predicted = torch.max(outputs.data, 1)
        total = (labels.size(0))
        correct = (predicted.squeeze() == labels.squeeze()).sum().item()
        acc  += (100 * correct / total)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        tepoch.set_postfix(loss=loss.item())

      loss_epoch.append(running_loss/(len(train_loader)))
      acc_epoch.append(acc/(len(train_loader))) 
  return model


def test_model_LSTM(model,test_loader):
  model.eval()
  with torch.no_grad():
      correct = 0
      total = 0
      for sequences, labels in test_loader: 
          sequences = sequences.to(device)
          labels = labels.to(device)
          outputs = model(sequences.permute(0,2,1))
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  print(f"Test Accuracy: {100 * correct/total:.2f}%")
  return (100 * correct/total)



class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Multiply hidden_size by 2 for bidirectional LSTM

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # Multiply num_layers by 2 for bidirectional LSTM
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        out, _ = self.bilstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
