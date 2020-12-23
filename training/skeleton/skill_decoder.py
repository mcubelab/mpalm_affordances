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

from networks import MultiStepDecoder, InverseModel


class SkillDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SkillDecoder, self).__init__()

        self.task_encoder = InverseModel(in_dim, hidden_dim, hidden_dim)
        self.seq_decoder(hidden_dim, out_dim)

    def forward(self, start, goal, transform, input, hidden):
        h = self.task_encoder(start, goal, transform)
        
