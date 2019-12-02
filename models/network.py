import torch
import torch.nn as nn
from torch.nn import functional as F

class FoodDesertClassifier(nn.Module):
    def __init__(self, input_size=10, hidden_dim_list=[4, 4, 4, 4], output_size=2):
        super(FoodDesertClassifier, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_dim_list[0])
        self.linear2 = nn.Linear(hidden_dim_list[0], hidden_dim_list[1])
        self.linear3 = nn.Linear(hidden_dim_list[1], hidden_dim_list[2])
        self.linear4 = nn.Linear(hidden_dim_list[2], hidden_dim_list[3])
        #self.input_size = input_size
        self.output = nn.Linear(hidden_dim_list[3], output_size)
        #self.size_list = hidden_dim_list
        #self.softmax = nn.Softmax(dim = 1)
        self.ReLU = nn.ReLU()
        

    """
    forward pass the classification neural network. Accepts x as parameter,
    which is the census feature vector for a zipcode.
    """
    def forward(self, x):
#        x = x.view(x.size(0), 1) # Flatten tensor
#
#        # First fully connected linear layer.
#        out = self.linear(x)
#        # First layer activation non-linearity function.
#        out = F.relu(out)
#        # Get predictions from final output layer.
#        out = self.linear2(out)
#        return out
        
        #x = x.view(x.size(0), 1)
        #---------------------------------------
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
        #x = self.softmax(x)
        
#        x = nn.Linear(self.input_size, self.size_list[0])
#        x = self.ReLU(x)
#        print("first: ", x)
#        for i in range(1, len(self.size_list)):
#            x = nn.Linear(self.size_list[i-1], self.size_list[i])
#            x = self.ReLU(x)
#            print("next: ", x)
#        x = self.output(x)
#        print("last: ", x)
#        return x
        
#        x = self.linear5(x)
#        x = self.ReLU(x)
#        x = self.linear6(x)
#        x = self.ReLU(x)
#        x = self.linear7(x)
#        x = self.ReLU(x)
#        x = self.linear8(x)
#        x = self.ReLU(x)
#        
#        self.linear5 = nn.Linear(hidden_dim_list[3], hidden_dim_list[4])
#        self.linear6 = nn.Linear(hidden_dim_list[4], hidden_dim_list[5])
#        self.linear7 = nn.Linear(hidden_dim_list[5], hidden_dim_list[6])
#        self.linear8 = nn.Linear(hidden_dim_list[6], hidden_dim_list[7])
#        , 4, 4, 4, 4