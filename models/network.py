import torch
import torch.nn as nn
from torch.nn import functional as F

class FoodDesertClassifier(nn.Module):
    def __init__(self, input_size=10, hidden_dim_list=[36, 36, 24, 16], output_size=2):
        super(FoodDesertClassifier, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_dim_list[0])
        self.linear2 = nn.Linear(hidden_dim_list[0], hidden_dim_list[1])
        self.linear3 = nn.Linear(hidden_dim_list[1], hidden_dim_list[2])
        self.linear4 = nn.Linear(hidden_dim_list[2], hidden_dim_list[3])
        self.output = nn.Linear(hidden_dim_list[3], output_size)
        self.ReLU = nn.ReLU()

    """
    forward pass the classification neural network. Accepts x as parameter,
    which is the census feature vector for a zipcode.
    """
    def forward(self, x):
        x = self.linear1(x)
        x = self.ReLU(x)
        x = self.linear2(x)
        x = self.ReLU(x)
        x = self.linear3(x)
        x = self.ReLU(x)
        x = self.linear4(x)
        x = self.ReLU(x)
        
        x = self.output(x)
        return x

    def predict(self, x):
        preds = nn.functional.softmax(self.forward(x), dim=0)
        preds.unsqueeze_(0)
        prediction = torch.argmax(preds)

        return prediction

