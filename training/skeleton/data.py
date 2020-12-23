import os, os.path as osp
import sys
import random
import torch.utils.data as data
import numpy as np
import copy


sys.path.append(osp.join(os.environ['CODE_BASE'], 'catkin_ws/src/primitives'))
from helper import util2 as util
from airobot.utils import common

class SkillDataset(data.Dataset):
    def __init__(self, train=False, aug=True):
        """Initialize this dataset class.
        """
        self.aug = aug
        self.base_path = osp.join(os.environ['CODE_BASE'], 'catkin_ws/src/primitives/data/play_transitions/test_run')
        np_files = os.listdir(self.base_path)
        np_files = sorted(np_files)
        self.data = [osp.join(self.base_path, np_file) for np_file in np_files]
        idx = int(len(self.data) * 0.9)

        if train:
            self.data = self.data[:idx]
        else:
            self.data = self.data[idx:]

        if aug:
            self.augment()

    def augment(self, max_steps=2):
        self.aug_data = []
        for i in range(len(self.data)):
            if i == len(self.data) - 1:
                break
            
            data1 = np.load(self.data[i], allow_pickle=True)
            a = bytes(data1['action_type']).decode('utf-8')

            action_seq = a + ' EOS'

            data = {}
            data['observation'] = data1['observation']
            data['next_observation'] = data1['next_observation']
            data['contact'] = data1['contact']
            data['subgoal'] = data1['subgoal'].astype(np.float32)
            data['actions'] = action_seq            
            
            self.aug_data.append(data)

            # initialize the multistep sequences with the intiail action and subgoal
            action_seq2 = action_seq
            prev_subgoal = util.matrix_from_pose(util.list2pose_stamped(data1['subgoal']))
            for j in range(i + 1, i + max_steps):
                data2 = np.load(self.data[j], allow_pickle=True)
                a2 = bytes(data2['action_type']).decode('utf-8')

                action_seq2 = action_seq2.split('EOS')[0] + a2 + ' EOS'
                subgoal = np.matmul(
                    util.matrix_from_pose(util.list2pose_stamped(data2['subgoal'])), 
                    prev_subgoal).astype(np.float32)

                data = {}
                data['observation'] = data1['observation']
                data['next_observation'] = data2['next_observation']  # get the last step point cloud
                data['contact'] = data1['contact']
                data['subgoal'] = util.pose_stamped2np(util.pose_from_matrix(subgoal)).astype(np.float32)
                data['actions'] = action_seq2

                self.aug_data.append(data)
                prev_subgoal = subgoal

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        """
        if self.aug:
            data = self.aug_data[index]
            action = data['actions']

        else:
            path = self.data[index]
            data = np.load(path, allow_pickle=True)
            action = bytes(data['action_type']).decode('utf-8')
            action = action + ' EOS'

        subgoal = data['subgoal']
        contact = data['contact']
        if contact.shape[0] == 7:
            contact = np.concatenate((contact, contact), axis=0)
        observation = data['observation']
        next_observation = data['next_observation']

        observation = observation[::int(observation.shape[0]/100)][:100]
        next_observation = next_observation[::int(next_observation.shape[0]/100)][:100]

        o_mean, o_mean_ = np.mean(observation, axis=0), np.mean(next_observation, axis=0)
        observation = observation - o_mean
        next_observation = next_observation - o_mean_

        observation = np.concatenate((observation, np.tile(o_mean, (observation.shape[0], 1))), axis=1)
        next_observation = np.concatenate((next_observation, np.tile(o_mean_, (next_observation.shape[0], 1))), axis=1)

        return subgoal, contact, observation, next_observation, action
        # return subgoal
    

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)