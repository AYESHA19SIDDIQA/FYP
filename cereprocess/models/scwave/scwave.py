import torch
import torch.nn as nn
import torch.nn.functional as F
from models.wavenet.wavenet import WaveBlock, WaveNet, WaveNetEnd

class SCWave(nn.Module):
    def __init__(self, input_shape, node1 = 64, node2 = 16, node3 = 32, dil1 = 3, dil2 = 2):
        super(SCWave, self).__init__()
        self.mffm_block1 = WaveBlock(50, node1, 3, dil1, True)
        self.mffm_block2 = WaveBlock(50, node1, 3, dil1, True)
        self.mffm_block3 = WaveBlock(node3, node2, 3, dil2, True)
        self.mffm_block4 = WaveBlock(node3, node2, 3, dil2, True)
        self.mffm_block5 = WaveBlock(node3, node3, 3, dil2, True)
        self.conv1 = nn.Conv1d(in_channels=node1, out_channels=node3, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(50)
        self.conv2 = nn.Conv1d(in_channels=node2, out_channels=node3, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(node3)
        self.conv3 = nn.Conv1d(in_channels=node3, out_channels=node3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(node3)
        self.fc = nn.Linear(node3, 2)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        gap = torch.mean(x, dim=1, keepdim=True)
        gsp = torch.std(x, dim=1, keepdim=True)
        gmp, _ = torch.max(x, dim=1, keepdim=True)
        gap = F.dropout(gap, 0.05, training=self.training)
        gmp = F.dropout(gmp, 0.05, training=self.training)
        gsp = F.dropout(gsp, 0.05, training=self.training)
        x = torch.cat((x, gap, gsp, gmp), dim=1)
        x1 = F.avg_pool1d(x, kernel_size=2, stride=2)
        x2 = F.max_pool1d(x, kernel_size=2, stride=2)
        x = torch.cat((x1, x2), dim=1)
        x = self.bn1(x)
        x1 = self.mffm_block1(x)
        x2 = self.mffm_block2(x)
        x = x1 + x2

        #Apply spatial dropout
        x = x.permute(0, 2, 1)
        x = F.dropout2d(x, 0.5, training=self.training)
        x = x.permute(0, 2, 1)
        
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn2(self.conv1(x)))
        x1 = self.mffm_block3(x)
        x2 = self.mffm_block4(x)
        x = x1 + x2
        x = self.bn3(self.conv2(x))
        x = self.mffm_block5(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = self.conv3(x)
        x = torch.mean(x, dim=2)
        x = self.fc(x)
        # return F.softmax(x, dim=-1)
        return x


class SCWaveLight(nn.Module):
    def __init__(self, input_shape, node = 42):
        super(SCWaveLight, self).__init__()
        self.mffm_block1 = WaveBlock(50, node, 3, 4, True)
        self.mffm_block2 = WaveBlock(50, node, 3, 2, True)
        self.conv1 = nn.Conv1d(in_channels=node, out_channels=node, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(50)
        self.bn2 = nn.BatchNorm1d(node)
        self.fc = nn.Linear(node, 2)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        gap = torch.mean(x, dim=1, keepdim=True)
        gsp = torch.std(x, dim=1, keepdim=True)
        gmp, _ = torch.max(x, dim=1, keepdim=True)
        gap = F.dropout(gap, 0.05, training=self.training)
        gmp = F.dropout(gmp, 0.05, training=self.training)
        gsp = F.dropout(gsp, 0.05, training=self.training)
        x = torch.cat((x, gap, gsp, gmp), dim=1)
        x1 = F.avg_pool1d(x, kernel_size=2, stride=2)
        x2 = F.max_pool1d(x, kernel_size=2, stride=2)
        x = torch.cat((x1, x2), dim=1)
        x = self.bn1(x)
        x1 = self.mffm_block1(x)
        x2 = self.mffm_block2(x)
        x = x1 + x2

        #Apply spatial dropout
        x = x.permute(0, 2, 1)
        x = F.dropout2d(x, 0.5, training=self.training)
        x = x.permute(0, 2, 1)
        
        x = F.max_pool1d(x, kernel_size=2, stride=4)
        x = F.relu(self.bn2(self.conv1(x)))
        x = torch.mean(x, dim=2)
        x = self.fc(x)
        # return F.softmax(x, dim=-1)
        return x

class SILM(nn.Module):
    def __init__(self, input_shape):
        super(SILM, self).__init__()

    def forward(self, x):
        gap = torch.mean(x, dim = 1, keepdim=True)
        gsp = torch.std(x, dim=1, keepdim=True)
        gmp, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat((x, gap, gsp, gmp), dim=1)
        return x

class WaveSILM(nn.Module):
    def __init__(self, input_shape):
        super(WaveSILM, self).__init__()
        self.silm = SILM(input_shape)
        channels, features = input_shape
        self.wavenet = WaveNet((channels+3, features))
        self.wavenetend = WaveNetEnd(64)

    def forward(self, x):
        x = self.silm(x)
        x = self.wavenet(x)
        x = self.wavenetend(x)
        return x
