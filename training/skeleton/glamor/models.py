import os, os.path as osp 
import sys
import argparse
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('..')
from skeleton.networks import GeomEncoder
from skeleton.skeleton_utils.skeleton_globals import SOS_token, EOS_token, PAD_token


class InverseModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(InverseModel, self).__init__()
        self.pcd_encoder = GeomEncoder(in_dim, hidden_dim)
        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim*2 + 7, 512), 
            nn.ReLU(), 
            nn.Linear(512, out_dim)) 
        self.prior_out_head = nn.Sequential(
            nn.Linear(hidden_dim, 512), 
            nn.ReLU(), 
            nn.Linear(512, out_dim)) 
        
    def forward(self, x, y, t):
        x = self.pcd_encoder(x)
        y = self.pcd_encoder(y)

        x = x.mean(dim=1)
        y = y.mean(dim=1)
        
        h = torch.cat((x, y, t), dim=-1)
        x = self.out_head(h)
        return x

    def prior_forward(self, x):
        x = self.pcd_encoder(x)
        x = x.mean(dim=1)
        x = self.prior_out_head(x)
        return x

class MultiStepDecoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(MultiStepDecoder, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.criterion = nn.NLLLoss(ignore_index=PAD_token)
        
    # def forward(self, x, hidden):
    #     output = self.embedding(x).view(1, 1, -1)
    #     output = F.relu(output)
    #     output, hidden = self.gru(output, hidden)
    #     output = self.log_softmax(self.out(output[0]))
    #     return output, hidden

    def embed(self, x):
        return self.embedding(x)

    def forward(self, x, hidden):
        output, hidden = self.gru(x, hidden)
        output = self.log_softmax(self.out(output)) 
        return output, hidden           