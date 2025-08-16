import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable

class DispatchNetwork(nn.Module):
    def __init__(self, channel=4,patch_size=64):
        super(DispatchNetwork, self).__init__()
        self.channel = channel
        
        # Define layers for processing input
        self.conv1 = nn.Conv3d(channel, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        
        # Define layers for output
        self.fc = nn.Linear(32 * patch_size**3, channel * channel)
        print('self.fc',self.fc.weight.dtype)
        
        # Define normalization layers
        self.batchnorm1 = nn.BatchNorm3d(64)
        self.batchnorm2 = nn.BatchNorm3d(32)
        
    def forward(self, x):
        # Input x has shape (batchsize, channel, patchsize, patchsize, patchsize)
        batchsize, channel, patch_size, _, _ = x.size()
        
        # Apply convolutional layers
        x = self.batchnorm1(F.relu(self.conv1(x)))
        x = self.batchnorm2(F.relu(self.conv2(x)))
        
        # Reshape for fully connected layer
        x = x.view(batchsize, -1)
        
        # Apply fully connected layer
        x = self.fc(x)
        
        # Reshape output to (batchsize, channel, channel)
        x = x.view(batchsize, self.channel, self.channel)
        
        # Apply softmax along the last dimension to ensure each row sums to 1
        x = F.softmax(x, dim=-1)
        
        return x

class DispatchNetwork1(nn.Module):
    def __init__(self, input_dim):
        super(DispatchNetwork1, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim * input_dim)
        # print('self.fc',self.fc.weight.dtype)
        self.input_dim = input_dim

    def forward(self, x):
        # Assuming x has shape (b, c)
        x = x.float()
        # x = [item.float() for item in x]
        b, c = x.size()
        x = self.fc(x)  # Fully connected layer
        
        # Reshape output to (b, c, c)
        x = x.view(b, self.input_dim, self.input_dim)
        
        # Apply some non-linearity if needed
        # x = torch.sigmoid(x)  # Example non-linearity
        
        return x

class ClsDispatchNet(nn.Module):
    def __init__(self, input_dim:int=4,hidden_dims:Iterable[int]=[4,8,16]) -> None:
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(dims[-1], input_dim*input_dim))
        self.mlp = nn.Sequential(*layers)
        self.num_class = input_dim
    
    def forward(self, x:torch.Tensor):
        x = self.mlp(x)  # (B, C*C)
        return x.view(-1, self.num_class, self.num_class)

# # Example usage:
# # Assuming x, y, z are the dimensions of your input tensor
# x, y, z = 64, 64, 64
# channel = 4  # Number of input channels
# patch_size = 64  # Size of the input patch
# batch_size = 2  # Batch size
# model = DispatchNetwork(channel=channel, patch_size=patch_size)
# input_tensor = torch.randn(batch_size, channel, x, y, z)
# output = model(input_tensor)
# print(output.shape)  # Example of how to use the model with input tensor
