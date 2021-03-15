import torch
from torch import nn

device = "cpu" # no tengo cuda ni nvidia :(

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 6, 6),
            nn.ReLU(),
            nn.MaxPool2d(3,3),
            nn.Conv2d(6, 16, 6),
            nn.ReLU(),
            nn.MaxPool2d(3,3)
            )

        self.linear = nn.Sequential(
            nn.Linear(16 * 26 * 26, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16,1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.convolutional(x)
        #print(x.shape)
        x = x.view(-1, 16 * 26 * 26)
        prob = self.linear(x)
        prob = prob.permute(1,0).squeeze().float()
        return prob 

