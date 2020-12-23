import os, os.path as osp
import sys
import numpy as np
import random
import torch

from airobot.utils import common
sys.path.append(osp.join(os.environ['CODE_BASE'], 'catkin_ws/src/primitives'))
from helper import util2 as util

def sample_data(data_dir):
    fnames = os.listdir(data_dir)
    fname = random.sample(fnames, 1)[0]

    data = np.load(osp.join(data_dir, fname))
    return data


def within_se2_margin(transformation):
    euler = common.rot2euler(transformation[:-1, :-1])
    return np.abs(euler[0]) < np.deg2rad(20) and np.abs(euler[1]) < np.deg2rad(20)


def process_observations(o, o_):
    o = o[::int(o.size)]


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
        self.index2skill = {0: 'SOS', 1: 'EOS'}
        self.n_skills = 2
        
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
            