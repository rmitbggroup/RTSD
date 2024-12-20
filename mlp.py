import torch
import torch.nn as nn


class MLP_REG(nn.Module):
    def __init__(self, input_dim=51, hidden_dim1=128, hidden_dim2=64, hidden_dim3=16):
        super(MLP_REG, self).__init__()
        self.input_dim   = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hidden_dim3
        self.output_dim  = 1
        
        self.hidden1 = nn.Linear(self.input_dim, self.hidden_dim1)
        self.act1 = nn.ReLU()

        self.hidden2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.act2 = nn.ReLU()

        self.hidden3 = nn.Linear(self.hidden_dim2, self.hidden_dim3)
        self.act3 = nn.ReLU()
        
        self.output  = nn.Linear(self.hidden_dim3, self.output_dim)
    
    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.output(x)
        return x