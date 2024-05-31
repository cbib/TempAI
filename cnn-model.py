import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNASequenceClassifier(nn.Module):
    def __init__(self):
        super(RNASequenceClassifier, self).__init__()

#         # Update convolution layers' parameters
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=512, kernel_size=7)
        self.pool = nn.MaxPool1d(kernel_size=2)

#         # Update dropout layer's parameters
        self.dropout1 = nn.Dropout(0.1)

#         # Update dense layers' parameters
        conv_output_size = ((550 - 7 + 1) // 2)
        self.fc1 = nn.Linear(512 * conv_output_size, 512) 
        self.fc2 = nn.Linear(512 , 3)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = self.pool(x)
        x = self.dropout1(x)

#         # Flatten the data for the fully connected layer
        x = torch.flatten(x, 1)  

#         # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
