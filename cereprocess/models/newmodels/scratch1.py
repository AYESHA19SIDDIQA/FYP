import torch
from torch import nn
from torch.nn import functional as F
from models.wavenet.wavenet import WaveLayer, WaveBlock, WaveNetEnd


class NewBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dil_rates, bn=False):
        super(NewBlock, self).__init__()
        self.dillayers = nn.ModuleList()
        dilations = [2**i for i in range(dil_rates)]
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1)
        for dilation in dilations:
            self.dillayers.append(WaveLayer(out_channels, kernel_size, dilation, bn))
        torch.nn.init.xavier_uniform_(self.conv1d.weight, gain=1.0, generator=None)

    def forward(self, x):
        x = F.relu(self.conv1d(x))
        for layer in self.dillayers:
            _, ch, features = x.shape
            x1 = layer(x)
            if (features > 16):
                x = F.avg_pool1d(x, kernel_size=4, stride=4)
                x1 = F.max_pool1d(x1, kernel_size=4, stride=4)
            x = x + x1
        return x



class Scratch1(nn.Module):
    def __init__(self, input_size, out_channels, kernel_size, dil_rates, bn=False):
        ch, features = input_size
        super(Scratch1, self).__init__()
        self.newblock1 = NewBlock(ch, out_channels, kernel_size, dil_rates, bn)
        self.newblock2 = NewBlock(ch, out_channels, kernel_size, dil_rates, bn)
        self.newblock3 = NewBlock(out_channels, out_channels, kernel_size, 1, bn)
        self.wavenetend = WaveNetEnd(out_channels)

    def forward(self, x):
        x1 = self.newblock1(x)
        x2 = self.newblock2(x)
        x = x1 + x2
        x = self.newblock3(x)
        x = torch.mean(x, dim=2)
        x = self.wavenetend(x)
        return x

class Scratch2(nn.Module):
    def __init__(self, input_size, out_channels, kernel_size, dil_rates, bn=False):
        super(Scratch2, self).__init__()
