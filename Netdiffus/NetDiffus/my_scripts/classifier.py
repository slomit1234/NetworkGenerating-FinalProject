import pandas as pd
from torch.utils.data import Dataset,DataLoader
import numpy as np
import os

import torch
from torch.nn import Module, Conv2d, ReLU, ELU, MaxPool2d, Flatten, Linear, Softmax, BatchNorm2d, BatchNorm1d, Dropout
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from sklearn.metrics import confusion_matrix
import cv2

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modified Dataset class to handle NumPy arrays from .npz file
class GeneratedDataset(Dataset):
    def __init__(self, images, labels, classes):
        self.images = images
        self.labels = labels
        self.classes = classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        X = torch.tensor(self.images[idx], dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize the images
        y = torch.zeros(self.classes, device=device)
        y[self.labels[idx]] = 1
        return X.to(device), y

# Define the DNN model
class DNN_model(Module):
    def __init__(self, num_channels, classes):
        super(DNN_model, self).__init__()

        self.conv1 = Conv2d(in_channels=num_channels, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.batchnm1 = BatchNorm2d(num_features=32)
        self.relu1 = ELU(alpha=1.0)
        self.conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.batchnm2 = BatchNorm2d(num_features=32)
        self.relu2 = ELU(alpha=1.0)
        self.max_pool1 = MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = Dropout(p=0.1)

        self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.batchnm3 = BatchNorm2d(num_features=64)
        self.relu3 = ReLU()
        self.conv4 = Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.batchnm4 = BatchNorm2d(num_features=64)
        self.relu4 = ReLU()
        self.max_pool2 = MaxPool2d(kernel_size=(2, 2))
        self.dropout2 = Dropout(p=0.1)

        self.conv5 = Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        self.batchnm5 = BatchNorm2d(num_features=128)
        self.relu5 = ReLU()
        self.conv6 = Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        self.batchnm6 = BatchNorm2d(num_features=128)
        self.relu6 = ReLU()
        self.max_pool3 = MaxPool2d(kernel_size=(2, 2))
        self.dropout3 = Dropout(p=0.1)

        self.conv7 = Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        self.batchnm7 = BatchNorm2d(num_features=256)
        self.relu7 = ReLU()
        self.conv8 = Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        self.batchnm8 = BatchNorm2d(num_features=256)
        self.relu8 = ReLU()
        self.max_pool4 = MaxPool2d(kernel_size=(2, 2))
        self.dropout4 = Dropout(p=0.1)

        self.flatten1 = Flatten()
        self.linear1 = Linear(in_features=16384, out_features=128)
        self.batchnm1_1D = BatchNorm1d(num_features=128)
        self.relu9 = ReLU()
        self.dropout5 = Dropout(p=0.7)
        self.linear2 = Linear(in_features=128, out_features=64)
        self.batchnm2_1D = BatchNorm1d(num_features=64)
        self.relu10 = ReLU()
        self.dropout6 = Dropout(p=0.5)
        self.linear3 = Linear(in_features=64, out_features=classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batchnm2(x)
        x = self.relu2(x)
        x = self.max_pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.batchnm3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.batchnm4(x)
        x = self.relu4(x)
        x = self.max_pool2(x)
        x = self.dropout2(x)

        x = self.conv5(x)
        x = self.batchnm5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.batchnm6(x)
        x = self.relu6(x)
        x = self.max_pool3(x)
        x = self.dropout3(x)

        x = self.conv7(x)
        x = self.batchnm7(x)
        x = self.relu7(x)
        x = self.conv8(x)
        x = self.batchnm8(x)
        x = self.relu8(x)
        x = self.max_pool4(x)
        x = self.dropout4(x)

        x = self.flatten1(x)
        x = self.linear1(x)
        x = self.batchnm1_1D(x)
        x = self.relu9(x)
        x = self.dropout5(x)
        x = self.linear2(x)
        x = self.batchnm2_1D(x)
        x = self.relu10(x)
        x = self.dropout6(x)
        x = self.linear3(x)
        return x

def main():
    num_of_vid = 20  # Number of classes for classification
    max_num_of_synth_traces = 400  # Max number of synthesized images for each class
    synth_group_size = 100  # Group size for synthetic images

    # Load your .npz file with the generated images and labels
    npz_data = np.load('samples_512x128x128x3.npz')
    images = npz_data['arr_0']  # Assuming images are stored in 'arr_0'
    labels = npz_data['arr_1']  # Assuming labels are stored in 'arr_1'

    # Create the dataset and dataloader
    dataset_train = GeneratedDataset(images, labels, num_of_vid)
    train_dl = DataLoader(dataset_train, batch_size=32, shuffle=True)

    # Initialize the model
    model = DNN_model(num_channels=3, classes=num_of_vid)
    model.to(device)

    # Define loss and optimizer
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    epochs = 140
    for epoch in range(epochs):
        correct = 0
        total = 0
        for i, (x, y) in enumerate(train_dl):
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            _, predicted = torch.max(yhat.data, 1)
            _, gt = torch.max(y.data, 1)
            total += y.size(0)
            correct += (predicted == gt).sum().item()
            loss.backward()
            optimizer.step()

    # Evaluate accuracy after training
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(train_dl):
            yhat = model(x)
            _, predicted = torch.max(yhat.data, 1)
            _, gt = torch.max(y.data, 1)
            total += y.size(0)
            correct += (predicted == gt).sum().item()

    accuracy = 100 * correct / total
    print(f'Final Accuracy : {accuracy} %')

if __name__ == main():
    main()