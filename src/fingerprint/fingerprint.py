# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from fingerprint.pubchemfp import pubchemfp


atts_out = []

# Device configuration: Use CUDA if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FP(nn.Module):
    atts_out = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self):
        super(FP, self).__init__()
        self.fp_2_dim = 512 
        self.dropout_fp = 0 
        self.cuda = torch.cuda.is_available()
        self.hidden_dim = 1024 
        self.fp_type = 'mixed'
        self.fp_dim = 1489
        self.to(device)
        self.fp_changebit = 0
        self.fc1=nn.Linear(self.fp_dim, self.fp_2_dim)
        self.act_func = nn.ReLU()
        self.fc2 = nn.Linear(self.fp_2_dim, self.hidden_dim)
        self.dropout = nn.Dropout(p=self.dropout_fp)

    

    def forward(self, smiles_list1):
        fp_list=[]
        for i, one in enumerate(smiles_list1):
            fp=[]
            mol = Chem.MolFromSmiles(one)
            fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
            fp_phaErGfp = AllChem.GetErGFingerprint(mol,fuzzIncrement=0.3,maxPath=21,minPath=1)
            fp_pubcfp = pubchemfp.GetPubChemFPs(mol)
            fp.extend(fp_maccs)
            fp.extend(fp_phaErGfp)
            fp.extend(fp_pubcfp)
            fp_list.append(fp)

        if self.fp_changebit is not None and self.fp_changebit != 0:
            fp_list = np.array(fp_list)
            fp_list[:,self.fp_changebit-1] = np.ones(fp_list[:,self.fp_changebit-1].shape)
            fp_list.tolist()

        fp_list = torch.Tensor(fp_list)

        if self.cuda:
            fp_list = fp_list.cuda()
        fp_out = self.fc1(fp_list)
        fp_out = self.dropout(fp_out)
        fp_out = self.act_func(fp_out)
        fp_out = self.fc2(fp_out)
        return fp_out
