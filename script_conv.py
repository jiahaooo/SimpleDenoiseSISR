import torch
import torch.nn as nn

class NN1(nn.Module):
    def __init__(self):
        super(NN1, self).__init__()
        self.nn = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=(1, 1), bias=True)

    def forward(self, x):
        x = self.nn(x)
        return x

class NN2(nn.Module):
    def __init__(self):
        super(NN2, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1), bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1), bias=True),
        )

    def forward(self, x):
        y = self.nn(x)
        x += y
        return x

class NN3(nn.Module):
    def __init__(self):
        super(NN3, self).__init__()
        self.input_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=(1, 1), bias=True),
        )
        self.hidden_layers = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1), bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1), bias=True),
        )
        self.output_layers = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), padding=(1, 1), bias=True),
        )

    def forward(self, x):
        x = self.input_layers(x)
        y = self.hidden_layers(x)
        y += x
        y = self.output_layers(y)
        return y

if __name__ == '__main__':
    a = torch.randn((4, 1, 256, 256), dtype=torch.float32)
    nn = NN3()
    b = nn(a)
    print(b.shape)