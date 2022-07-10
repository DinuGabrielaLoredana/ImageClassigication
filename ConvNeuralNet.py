import torch.nn as nn


class ConvNeuralNet(nn.Module):
    def __init__(self, layers, linear_layers):
        super(ConvNeuralNet, self).__init__()
        self.layers = layers
        self.linear_layers = linear_layers

    # Progresses data across layers
    def forward(self, x):
        for value in self.layers:
            x = value(x)
        x = x.reshape(x.size(0), -1)
        for value in self.linear_layers:
            x = value(x)
        return x
