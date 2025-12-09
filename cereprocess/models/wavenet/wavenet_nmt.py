import torch
from torch import nn
from torch.nn import functional as F
from models.wavenet.alternate_nmt import AlternateLayer

class WaveLayer(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation):
        super(WaveLayer, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.filter = nn.Conv1d(in_channels, in_channels, 1)
        self.gate = nn.Conv1d(in_channels, in_channels, 1)

        # Initialize weights
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.filter.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.gate.weight, gain=1.0)
       # self.skip = nn.Conv1d(out_channels, in_channels, 1)
       # self.residual = nn.Conv1d(out_channels, in_channels, 1)
        
    def forward(self, x):
        # x_padded = torch.nn.functional.pad(x, (self.padding, 0))
        output = self.conv(x)
        filter = self.filter(output)
        gate = self.gate(output)
        tanh = self.tanh(filter)
        sig = self.sig(gate)
        z = tanh*sig
        z = z[:,:,:-self.padding]
        x = x + z
        return x

class WaveBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rates):
        super(WaveBlock, self).__init__()
        self.layers = nn.ModuleList()
        dilations = [2**i for i in range(dilation_rates)]
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1)
        for dilation in dilations:
            self.layers.append(WaveLayer(out_channels, kernel_size, dilation))
        torch.nn.init.xavier_uniform_(self.conv1d.weight, gain=1.0, generator=None)

    def forward(self, x):
        x = self.conv1d(x)
        for layer in self.layers:
            x = layer(x)
        return x


class WaveNet(nn.Module):
    def __init__(self):
        super(WaveNet, self).__init__()
        self.block1 = WaveBlock(22, 16, 3, 8)
        self.block2 = WaveBlock(16, 32, 3, 5)
        self.block3 = WaveBlock(32, 64, 3, 3)
        self.block4 = WaveBlock(64, 64, 2, 2)
        self.lstmblock = nn.LSTM(7, 64, 1, batch_first=True)
        self.dense_layer = nn.Linear(64, 2)
        torch.nn.init.xavier_uniform_(self.dense_layer.weight, gain=1.0, generator=None)


    def forward(self, x):
        x = self.block1(x)
        x = F.avg_pool1d(x, 10)
        x = self.block2(x)
        x = F.avg_pool1d(x, 10)
        x = self.block3(x)
        x = F.avg_pool1d(x, 10)
        x = self.block4(x)
        x = F.avg_pool1d(x, 4)
        _, (_, x) = self.lstmblock(x)
        x = x.squeeze(0)
        x = F.dropout(x, 0.5, training=self.training)
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


class WaveNetFull(nn.Module):
    def __init__(self, input_shape):
        super(WaveNetFull, self).__init__()
        self.wavenet = WaveNet()
        self.alternate = AlternateLayer(input_shape)
        self.wavenetend = WaveNetEnd(124)

    def forward(self, x):
        y = self.wavenet(x)
        z = self.alternate(x)
        x = torch.cat((y, z), -1)
        x = self.wavenetend(x)
        return x

        



def get_wavenet_dil():
    return nn.Sequential(WaveNet(), WaveNetEnd(64))
    
def get_wavenet_alt(input_shape):
    return nn.Sequential(AlternateLayer(input_shape), WaveNetEnd(60))
    

