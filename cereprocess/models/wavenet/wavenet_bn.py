import torch
from torch import nn
from torch.nn import functional as F
from models.wavenet.wavenet import WaveLayer, WaveBlock


class WaveNetDP(nn.Module):
    def __init__(self):
        super(WaveNetDP, self).__init__()
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
        x = F.dropout(x, 0.2, training=self.training)
        x = self.block2(x)
        x = F.avg_pool1d(x, 10)
        x = F.dropout(x, 0.2, training=self.training)
        x = self.block3(x)
        x = F.avg_pool1d(x, 10)
        x = F.dropout(x, 0.2, training=self.training)
        x = self.block4(x)
        x = F.avg_pool1d(x, 2)
        x = F.dropout(x, 0.2, training=self.training)
        _, (_, x) = self.lstmblock(x)
        x = x.squeeze(0)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.dense_layer(x)
        return x
