import torch
import torch.nn as nn
from torch.nn import functional as F

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.weights = self.linear.weight

    def forward(self, x):
#        print("X_VALUE: ")
#        print(x)
        out = self.linear(x)
#        print("OUT VALUE: ")
#        print(out)
        return (out, self.weights)
