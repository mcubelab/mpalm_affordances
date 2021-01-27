import os, os.path as osp
import sys
import random
import torch.utils.data as data
import numpy as np
import copy

from scipy.spatial.transform import Rotation as R
from dreamer_utils import SkillLanguage, prepare_sequence_tokens


class SkeletonTransitionDataset(data.Dataset):
    def __init__(self, train=False, overfit=False):
        """Initialize this dataset class.
        """
        # self.base_path = osp.join(os.environ['CODE_BASE'], 'catkin_ws/src/primitives/data/skeleton_policy_dev')
        self.base_path = osp.join(os.getcwd(), 'sample_skeleton_data')
        np_files = os.listdir(self.base_path)
        np_files = sorted(np_files)

        self.data = [osp.join(self.base_path, np_file) for np_file in np_files]

        action_sequence = 'pull-right grasp pull-right EOS'

        self.skill_lang = SkillLanguage('default')
        self.skill_lang.add_skill_seq(action_sequence)
        token_action_sequence = prepare_sequence_tokens(action_sequence.split(' '), self.skill_lang.skill2index) 

        self.action_sequence = action_sequence
        self.token_action_sequence = token_action_sequence

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        """
        path = self.data[index]
        data = np.load(path, allow_pickle=True)        
        action = self.token_action_sequence
        observation = data['pointclouds']
        table = data['table_pointcloud'][:500]

        mask = np.zeros((observation.shape[0], observation.shape[1], 1), dtype=np.int32)
        mask[np.where(np.abs(observation[:, :, -1] < 0.01))] = 1

        reward = np.zeros((observation.shape[0] + 1, 1, 1))
        reward[-2:] = 1
        # rewards[:-2] = -1

        goal = data['transformation_des_list'][0]
        goal_q = R.from_matrix(goal[:-1, :-1]).as_quat()
        goal_t = goal[:-1, -1]
        # transformation_des = np.tile(np.concatenate((goal_t, goal_q)), (observation.shape[0] + 1, 1))
        # goal_pcd = np.tile(observation[-1], (observation.shape[0] + 1, 1, 1))
        transformation_des = np.concatenate((goal_t, goal_q))
        goal_pcd = observation[-1]

        return observation, action, table, mask, reward, transformation_des, goal_pcd

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)                