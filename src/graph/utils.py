import torch
import math
import torch.nn as nn
from rdkit import Chem
from rdkit import rdBase
rdBase.DisableLog('rdApp.*')


def split(sm):
    
    arr = []
    i = 0
    while i < len(sm)-1:
        if not sm[i] in ['%', 'C']:
            arr.append(sm[i])
            i += 1
        elif sm[i]=='%':
            arr.append(sm[i:i+3])
            i += 3
        elif sm[i]=='C' and sm[i+1]=='+':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='C' and sm[i+1]=='H' and sm[i+2]=='+':
            arr.append(sm[i:i+3])
            i += 3
        elif sm[i]=='C' and sm[i+1]=='H' and sm[i+2]=='2' and sm[i+3]=='+':
            arr.append(sm[i:i+4])
            i += 4
        else:
            arr.append(sm[i])
            i += 1
    if i == len(sm)-1:
        arr.append(sm[i])
    return ' '.join(arr) 


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
    

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


def sample(msms):
    ret = []
    for msm in msms:
        ret.append(torch.multinomial(msm.exp(), 1).squeeze())
    return torch.stack(ret)

def validity(smiles):
    loss = 0
    for sm in smiles:
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            loss += 1
    return 1-loss/len(smiles)

