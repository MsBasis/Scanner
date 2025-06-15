#Better late than never
import torch 
import torch.nn as nn 
from Data_Loader import dara_loaders

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,2)
        )
        
    def forward(self,x):
        return self.net(x)


def training_arc(csv, epochs=20, batch_size=128, lr=0.001):
    pass






