import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim = 64, num_layers=2):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim,1)

    def forward(self,x):
        _,hn = self.rnn(x)
        return self.fc(hn[-1])
    
