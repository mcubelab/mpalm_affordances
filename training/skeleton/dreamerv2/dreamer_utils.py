import os, os.path as osp
import sys
import argparse
import random
import copy
import numpy as np
import torch

def to_onehot(x, n):
    if x.dim() < 2:
        x = x[:, None]
    s = x.size()
    y_onehot = torch.FloatTensor(s[0], n).to(x.device)

    y_onehot.zero_()
    y_onehot.scatter_(1, x, 1)
    return y_onehot


def to_onehot3d(x, n):
    s = x.size()
    y_onehot = torch.FloatTensor(s[0], s[1], n).to(x.device)
    y_onehot.zero_()
    y_onehot.scatter_(2, x, 1)
    return y_onehot


def process_pointcloud_sequence_batch(pcd):
    pcd = pcd[:, ::int(pcd.size(2)/100)][:, :, :100]
    o_mean = pcd.mean(2)[:, :, None, :]
    pcd = pcd - o_mean
    s = pcd.size()
    pcd = torch.cat((pcd, o_mean.repeat((1, 1, s[2], 1))), dim=3)
    return pcd
    

def prepare_sequence_tokens(seq, to_idx):
    idxs = [to_idx[tok] for tok in seq]
    return torch.Tensor(idxs).long()


SOS_token = 0
EOS_token = 1

class SkillLanguage:
    def __init__(self, name):
        self.name = name
        self.skill2index = {}
        self.skill2count = {}
        self.index2skill = {0: 'PAD', 1: 'SOS', 2: 'EOS'}
        self.n_skills = 0
        
    def add_skill_seq(self, seq):
        for skill in seq.split(' '):
            self.add_skill(skill)
        
    def add_skill(self, skill, keep_count=False):
        if skill not in self.skill2index:
            self.skill2index[skill] = self.n_skills
            self.skill2count[skill] = 1
            self.index2skill[self.n_skills] = skill
            self.n_skills += 1
        else:
            if keep_count:
                self.skill2count[skill] += 1
            else:
                pass

    def process_data(self, data):
        for i in range(len(data)):
            outputs = data[i][1]
            for skill in outputs:
                self.add_skill(skill)
                