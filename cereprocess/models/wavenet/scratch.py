import torch
from torch import nn
from torch.nn import functional as F
from models.wavenet.wavenet import WaveLayer, WaveBlock
from models.wavenet.alternate import AlternateLayer

class Scratch(nn.Module):
    def __init__(self):
        super(Scratch, self).__init__()
        self.block1 = WaveBlock(22, 16, 3, 4)
        self.block2 = WaveBlock(16, 64, 3, 2)
        # self.dense_layer = nn.Linear(16, 2)

    def forward(self, x):
        x = self.block1(x)
        x = F.avg_pool1d(x, 10)
        x = self.block2(x)
        x = F.avg_pool1d(x, 2)
        x = torch.mean(x, dim = -1)
        return x

class Wave1(nn.Module):
    def __init__(self):
        super(Wave1, self).__init__()
        self.block1 = WaveBlock(22, 32, 3, 4)
        self.dense_layer = nn.Linear(32, 2)
        torch.nn.init.xavier_uniform_(self.dense_layer.weight, gain=1.0, generator=None)

    def forward(self, x):
        x = self.block1(x)
        x = torch.mean(x, dim = -1)
        x = self.dense_layer(x)
        return x


class WaveNetEnd(nn.Module):
    def __init__(self, input_size):
        super(WaveNetEnd, self).__init__()
        self.dense_layer = nn.Linear(input_size, 2)
        torch.nn.init.xavier_uniform_(self.dense_layer.weight, gain=1.0, generator=None)

    def forward(self, x):
        x = self.dense_layer(x)
        # x = F.softmax(x, dim=1)
        return x


class WaveNetLight(nn.Module):
    def __init__(self, input_shape):
        super(WaveNetLight, self).__init__()
        self.wavenet = Scratch()
        self.alternate = AlternateLayer(input_shape)
        self.wavenetend = WaveNetEnd(124)

    def forward(self, x):
        y = self.wavenet(x)
        z = self.alternate(x)
        x = torch.cat((y, z), -1)
        x = self.wavenetend(x)
        return x

