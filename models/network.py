import torch
import torch.nn as nn
from torch.nn import functional as F

class FoodDesertClassifer(nn.Module):
    def __init__(self, input_size=10, hidden_size=10, output_size=2):
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    """
    forward pass the classification neural network. Accepts x as parameter,
    which is the census feature vector for a zipcode.
    """
    def forward(self, x):
        x = x.view(x.size(0), 1) # Flatten tensor

        # First fully connected linear layer.
        out = self.linear(x)
        # First layer activation non-linearity function.
        out = F.relu(out)
        # Get predictions from final output layer.
        out = self.linear2(out)
        return out

