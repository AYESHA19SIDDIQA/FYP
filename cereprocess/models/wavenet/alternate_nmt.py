import torch 
from torch import nn
from torch.nn import functional as F

class AlternateLayer(nn.Module):
    def __init__(self, input_shape):
        self.seq_len, self.input_size = input_shape
        assert self.input_size == 30000
        super(AlternateLayer, self).__init__()
        self.timdistLSTM = nn.LSTM(self.seq_len, 64, 1, batch_first=True)
        self.attFCN = nn.Linear(64, 64)
        self.seqLSTM = nn.LSTM(64, 64, batch_first=True)
        self.findense = nn.Linear(64, 2)
        self.fintanh = nn.Tanh()
        # Applying the Xavier initialization
        torch.nn.init.xavier_uniform_(self.attFCN.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.findense.weight, gain=1.0)


    def forward(self, x):
        x = x.flip(-1)
        batch_size, seq_len, input_dim = x.size()
        x = x.transpose(1, 2)
        x = x.reshape(batch_size*30, 1000, seq_len)
        _, (x, _) = self.timdistLSTM(x)
        x = x.reshape(batch_size, 30, 64)
        att = self.attFCN(x)
        att = F.softmax(att, dim =-1)
        x = x * att
        x, _ = self.seqLSTM(x)
        x = F.dropout(x, 0.2, training = self.training)
        x = self.findense(x)
        x = self.fintanh(x)
        x = x.reshape(batch_size, 60)
        return x

if __name__ == '__main__':
    input_shape = (32, 22, 15000)
    x = torch.randn(input_shape)
    model = AlternateLayer(input_shape)
    output = model(x)
    print(output.shape)

        

