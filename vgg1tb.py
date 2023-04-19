import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn           
import torch.optim as optim     
import torch.nn.functional as F 
from torch.utils.data import DataLoader 
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm 

# Setting the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Single Block VGG
VGG = ['M', 32, 'M']

class VGG_net(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 2):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_layers = self.create_conv_layers(VGG)

        self.fcs = nn.Sequential(
            # nn.Linear(56*56*32, 4096), # This has to be put with care
            nn.Linear(56*56*32, 128),
            nn.ReLU(),
            # nn.Dropout(p = 0.5),
            # nn.Linear(4096, 4096),
            # nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(128, out_channels)
        )
    
    def forward(self, x):
        # print("shape1: ",x.shape)
        x = self.conv_layers(x)
        # print("shape2: ",x.shape)
        x = x.reshape(x.shape[0], -1)
        # print("shape3: ",x.shape)
        x = self.fcs(x)
        # print("shape4: ",x.shape)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [nn.Conv2d(in_channels = in_channels, out_channels = out_channels, 
                                    kernel_size = (3,3), stride = (1,1), padding = (1,1)),
                                    # nn.BatchNorm2d(x),
                                    nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))]
        
        return nn.Sequential(*layers)


# For Pre-Processing the data
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 256x256 pixels
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize the image
])

batch_sizes  = [1, 16, 32]
learning_rates = [0.1, 0.01, 0.001, 0.0001]

# Training the Network
num_epochs = 2

for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        step = 0
        model = VGG_net(in_channels=3, out_channels=2).to(device)
        dataset = datasets.ImageFolder("images", transform=transform)

        test_split = .2
        shuffle_dataset = True
        random_seed= 42

        writer = SummaryWriter(f'runs/VGG/Mini Batch size: {batch_size} LR: {learning_rate}') 

        criterion = nn.CrossEntropyLoss()
        optimiser = optim.Adam(model.parameters(), lr = learning_rate)
        # Creating data indices for training and test splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(test_split * dataset_size))

        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_indices, test_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,sampler=test_sampler)

        for epochs in range(num_epochs):
            for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
                data = data.to(device = device)
                target = target.to(device = device)
                # data shape is 64,1,28,28

                # Feed Forward
                score = model(data)
                loss = criterion(score, target)

                # Backprop
                optimiser.zero_grad()
                loss.backward()

                # Calculate 'running' training accuracy
                _, predictions = score.max(1)
                num_correct = (predictions == target).sum()
                running_train_acc = float(num_correct)/float(data.shape[0])

                writer.add_scalar('Training loss', loss, global_step=epochs) 
                writer.add_scalar('Number of correct', num_correct, global_step=epochs)
                writer.add_scalar('Training accuracy', running_train_acc, global_step=epochs)
                step += 1

                # Gradient Descent
                optimiser.step()

print("Done")
