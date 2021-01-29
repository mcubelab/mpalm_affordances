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
        self.base_path = osp.join(os.environ['CODE_BASE'], 'catkin_ws/src/primitives/data/skeleton_policy_dev_aug')
        # self.base_path = osp.join(os.getcwd(), 'sample_skeleton_data')
        np_files = os.listdir(self.base_path)
        np_files = sorted(np_files)

        self.data = [osp.join(self.base_path, np_file) for np_file in np_files]
        self.max_sequence_length = 4

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        """
        path = self.data[index]
        data = np.load(path, allow_pickle=True)        

        # pad the action sequence and the observation sequence based on the maximum sequence length in the dataset
        num_actions = len(data['skeleton'])
        action = ' '.join(data['skeleton'].astype('str').tolist()) + ' EOS' * (self.max_sequence_length - num_actions)
        observation = data['pointclouds']
        if observation.shape[0] < self.max_sequence_length:
            observation = np.concatenate((observation, observation[-1][None, :, :]), axis=0)
        table = data['table_pointcloud'][:500]

        mask = np.zeros((observation.shape[0], observation.shape[1], 1), dtype=np.int32)
        mask[np.where(np.abs(observation[:, :, -1] < 0.01))] = 1

        reward = np.zeros((observation.shape[0] + 1, 1, 1))
        reward[num_actions:] = 1

        goal = data['transformation_des_list'][0]
        goal_q = R.from_matrix(goal[:-1, :-1]).as_quat()
        goal_t = goal[:-1, -1]

        transformation_des = np.concatenate((goal_t, goal_q))
        goal_pcd = observation[-1]

        return observation, action, table, mask, reward, transformation_des, goal_pcd

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)                


class SkeletonDataset(data.Dataset):
    def __init__(self, train=False, overfit=False):
        """Initialize this dataset class.
        """
        self.base_path = osp.join(os.environ['CODE_BASE'], 'catkin_ws/src/primitives/data/skeleton_samples')
        np_files = os.listdir(self.base_path)
        np_files = sorted(np_files)
        self.data = [osp.join(self.base_path, np_file) for np_file in np_files]
        idx = int(len(self.data) * 0.9)
        overfit_idx = int(len(self.data) * 0.1)

        if train:
            self.data = self.data[:idx] if not overfit else self.data[:overfit_idx]
        else:
            self.data = self.data[idx:]


    def __getitem__(self, index):
        """Return a data point and its metadata information.
        """
        path = self.data[index]
        data = np.load(path, allow_pickle=True)
        action = data['skeleton'].item().decode('utf-8')
        action = action + ' EOS'
        action_list = action.split(' ')
        new_action_list = []
        for a in action_list:
            if 'pull' in a:
                new_action_list.append('pull_right')
            else:
                new_action_list.append(a)
        action_str = ' '.join(new_action_list)
        num_of_actions = len(new_action_list)

        subgoal = data['transformation_desired']
        contact = np.zeros(14)
        x_table, y_table = np.linspace(0, 0.5, 23), np.linspace(-0.4, 0.4, 23)
        xx, yy = np.meshgrid(x_table, y_table)
        table_pts = []
        for i in range(xx.shape[0]):
            for j in range(yy.shape[0]):
                pt = [xx[i, j], yy[i, j], np.random.random() * 0.002 - 0.001]
                table_pts.append(pt)
        table = np.asarray(table_pts)

        observation = data['observation']
        
        o = np.ones((observation.shape[0], 4)).T
        o[:-1, :] = observation.T
        goal_pcd = np.matmul(subgoal, o).T[:, :-1].squeeze()
        goal_q = R.from_matrix(subgoal[:-1, :-1]).as_quat()
        goal_t = subgoal[:-1, -1]
        transformation_des = np.concatenate((goal_t, goal_q))        

        observation = observation[::int(observation.shape[0]/100)][:100]
        goal_pcd = goal_pcd[::int(goal_pcd.shape[0]/100)][:100]

        reward_step = num_of_actions
        mask = np.zeros((observation.shape[0], 1), dtype=np.int32)

        return observation, action_str, table, mask, reward_step, transformation_des, goal_pcd
    

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)           