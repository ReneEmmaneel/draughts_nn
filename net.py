import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(5, args.conv_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(args.conv_filters),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.net(x)
        return x

class ResidualLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(args.conv_filters, args.conv_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(args.conv_filters),
            nn.ReLU(),
            nn.Conv2d(args.conv_filters, args.conv_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(args.conv_filters)
        )

    def forward(self, x):
        x = x + self.net(x)
        x = nn.functional.relu(x)
        return x

class InputLayer(nn.Module):
    def __init__(self, args):
        super().__init__()

        blocks = [ConvLayer(args)]
        for i in range(args.residual_layers):
            blocks.append(ResidualLayer(args))

        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.net(x)
        return x

class ValueHead(nn.Module):
    def __init__(self, args):
        super().__init__()

        squares = 50

        self.net = nn.Sequential(
            nn.Conv2d(args.conv_filters, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * squares, args.value_head_hidden_layer),
            nn.ReLU(),
            nn.Linear(args.value_head_hidden_layer, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.net(x)
        return x

class PolicyHead(nn.Module):
    def __init__(self, args):
        super().__init__()

        squares = 50

        self.net = nn.Sequential(
            nn.Conv2d(args.conv_filters, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * squares, squares * squares)
        )

    def forward(self, x):
        x = self.net(x)
        return x
