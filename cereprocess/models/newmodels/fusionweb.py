import torch
from torch import nn
from torch.nn import functional as F
from models.scnet.scnet import SILM
from models.wavenet.wavenet import WaveBlock

class FusionMod(nn.Module):
    def __init__(self, in_channels):
        super(FusionMod, self).__init__()
        self.convL1 = nn.Conv1d(in_channels=in_channels, out_channels=8, kernel_size=5, padding=2)
        self.bnL1 = nn.BatchNorm1d(8)
        self.convL2 = nn.Conv1d(in_channels=in_channels + 8, out_channels=16, kernel_size=5, padding=2)
        self.bnL2 = nn.BatchNorm1d(16)

        nn.init.xavier_uniform_(self.convL1.weight)
        nn.init.xavier_uniform_(self.convL2.weight)


        self.convR1 = nn.Conv1d(in_channels=in_channels, out_channels=8, kernel_size=5, padding=2)
        self.bnR1 = nn.BatchNorm1d(8)
        self.convR2 = nn.Conv1d(in_channels=in_channels + 8, out_channels=16, kernel_size=5, padding=2)
        self.bnR2 = nn.BatchNorm1d(16)

        nn.init.xavier_uniform_(self.convR1.weight)
        nn.init.xavier_uniform_(self.convR2.weight)

    def forward(self, input):
        xL1 = F.relu(self.bnL1(self.convL1(input)))
        xR1 = F.relu(self.bnR1(self.convR1(input)))
        x1 = torch.cat((xL1 + xR1, input), dim=1)
        xL2 = F.relu(self.bnL2(self.convL2(x1)))
        xR2 = F.relu(self.bnR2(self.convR2(x1)))
        x = torch.cat((xL2 + xR2, x1), dim=1)
        return x

class FusionSCWave(nn.Module):
    def __init__(self, input_shape, conv1units = 64, waveUnits = 64, dilations=3, concat = True, pool = False):
        super(FusionSCWave, self).__init__()
        channels, features = input_shape
        self.pool = pool
        self.concat = concat
        self.silm = SILM()
        silm_inp = channels + 3
        if (pool):
            silm_inp *= 2
        self.fusion1 = FusionMod(silm_inp)
        self.fusion2 = FusionMod(silm_inp)
        # self.fusion3 = FusionMod(32)
        if (concat==False):
            self.inpUnits = silm_inp + 24
        else:
            self.inpUnits = 2 * (silm_inp + 24)


        self.waveblock = WaveBlock(conv1units, waveUnits, 3, dilations)
        self.conv1 = nn.Conv1d(in_channels=self.inpUnits, out_channels=conv1units, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv1d(in_channels = 56, out_channels = 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(silm_inp)
        self.bn2 = nn.BatchNorm1d(conv1units)
        self.bn3 = nn.BatchNorm1d(waveUnits)
        self.fc = nn.Linear(waveUnits, 2)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = self.silm(x)
        if (self.pool):
            x1 = F.avg_pool1d(x, kernel_size=2, stride=2)
            x2 = F.max_pool1d(x, kernel_size=2, stride=2)
            x = torch.cat((x1, x2), dim=1)
        x = self.bn1(x)
        x1 = self.fusion1(x)
        x2 = self.fusion2(x)
        if (self.concat == True):
            x = torch.concat([x1, x2], dim=1)
        else:
            x = x1 + x2

        #Apply spatial dropout
        x = x.permute(0, 2, 1)
        x = F.dropout2d(x, 0.5, training=self.training)
        x = x.permute(0, 2, 1)
        
        x = F.max_pool1d(x, kernel_size=2, stride=4)
        x = F.relu(self.bn2(self.conv1(x)))
        # x = self.fusion3(x)   
        # x = F.relu(self.bn3(self.conv2(x)))
        x = F.relu(self.bn3(self.waveblock(x)))
        x = torch.mean(x, dim=2)
        x = self.fc(x)
        # return F.softmax(x, dim=-1)
        return x


class FusionEnsemble(nn.Module):
    def __init__(self, input_shape, ensemble_size, conv1Units, waveUnits, concat=False):
        super(FusionEnsemble, self).__init__()
        self.modList = nn.ModuleList()
        self.enSize = ensemble_size
        for i in range(ensemble_size):
            self.modList.append(FusionSCWave(input_shape, conv1Units, waveUnits, concat))
    
    def forward(self, x):
        y = self.modList[0](x)
        for i in range(1, self.enSize):
            y += self.modList[i](x)
        return y


