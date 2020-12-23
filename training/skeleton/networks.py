import os, os.path as osp
import sys
import argparse
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from airobot.utils import common

sys.path.append(osp.join(os.environ['CODE_BASE'], 'catkin_ws/src/primitives'))
sys.path.append(osp.join(os.environ['CODE_BASE'], 'training/gat'))
from helper import util2 as util
from models_vae import GeomEncoder


class InverseModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(InverseModel, self).__init__()
        self.pcd_encoder = GeomEncoder(in_dim, hidden_dim)
        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim*2 + 7, 512), 
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


class MultiStepDecoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(MultiStepDecoder, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x, hidden):
        output = self.embedding(x).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden