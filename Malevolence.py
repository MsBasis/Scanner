#Better late than never
import torch 
import torch.nn as nn 


class MLP(nn.Module):
    def __init__(slef, input_dim):
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














