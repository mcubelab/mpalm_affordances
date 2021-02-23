import os, os.path as osp
import sys
import random
import torch.utils.data as data
import numpy as np
import copy

from scipy.spatial.transform import Rotation as R

class SkillPlayDataset(data.Dataset):
    def __init__(self, train=False, aug=True, max_steps=2):
        """Initialize this dataset class.
        """
        self.aug = aug
        self.base_path = osp.join(os.environ['CODE_BASE'], 'catkin_ws/src/primitives/data/play_transitions/test_run')
        # self.base_path = osp.join(os.getcwd(), 'data/play_transitions_test')
        np_files = os.listdir(self.base_path)
        np_files = sorted(np_files)
        self.data = [osp.join(self.base_path, np_file) for np_file in np_files]
        idx = int(len(self.data) * 0.9)

        if train:
            self.data = self.data[:idx]
        else:
            self.data = self.data[idx:]

        if aug:
            self.augment(max_steps=max_steps)

    def augment(self, max_steps=2):
        self.aug_data = []
        for i in range(len(self.data)):
            if i == len(self.data) - max_steps:
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
            prev_subgoal = np.eye(4)
            prev_subgoal[:-1, :-1] = R.from_quat(data1['subgoal'][3:]).as_matrix()
            prev_subgoal[:-1, -1] = data1['subgoal'][:3]
            for j in range(i + 1, i + max_steps):
                data2 = np.load(self.data[j], allow_pickle=True)
                a2 = bytes(data2['action_type']).decode('utf-8')

                action_seq2 = action_seq2.split('EOS')[0] + a2 + ' EOS'
                current_subgoal_mat = np.eye(4)
                current_subgoal_mat[:-1, -1] = data2['subgoal'][:3]
                current_subgoal_mat[:-1, :-1] = R.from_quat(data2['subgoal'][3:]).as_matrix()
                subgoal = np.matmul(current_subgoal_mat, prev_subgoal).astype(np.float32)

                data = {}
                data['observation'] = data1['observation']
                data['next_observation'] = data2['next_observation']  # get the last step point cloud
                data['contact'] = data1['contact']
                data['subgoal'] = np.concatenate((subgoal[:-1, -1], R.from_matrix(subgoal[:-1, :-1]).as_quat()), axis=0)
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

        subgoal = data['subgoal'].astype(np.float32)
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


class SkeletonTransitionDataset(data.Dataset):
    def __init__(self, train=False, overfit=False):
        """Initialize this dataset class.
        """
        # self.base_path = osp.join(os.environ['CODE_BASE'], 'catkin_ws/src/primitives/data/skeleton_policy_dev_aug')
        self.base_path = osp.join(os.getcwd(), 'data/skeleton_policy_dev_aug')
        # self.base_path = osp.join(os.getcwd(), 'sample_skeleton_data')
        np_files = os.listdir(self.base_path)
        np_files = sorted(np_files)

        self.data = [osp.join(self.base_path, np_file) for np_file in np_files]
        self.max_sequence_length = 4

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
        # self.base_path = osp.join(os.getcwd(), 'data/skeleton_samples')
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
    

class SkeletonDatasetGlamor(data.Dataset):
    def __init__(self, train=False, overfit=False, append_table=False):
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
        
        self.append_table = append_table

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        """
        path = self.data[index]
        data = np.load(path, allow_pickle=True)
        action = bytes(data['skeleton']).decode('utf-8')
        action = action + ' EOS'
        new_pull = np.random.choice(['pull_right', 'pull_left'], 1)[0]
        action = action.replace('pull', new_pull)

        if self.append_table:
            # we want to pre-process all actions to have the suffix "_table" at the end, indicating the surface that was used
            act_list = action.split(' ')
            new_act_list = []
            for act in act_list:
                if 'EOS' in act:
                    new_act_list.append(act)
                    continue
                act = act + '_table'
                new_act_list.append(act)
            action = ' '.join(new_act_list)

        subgoal = data['transformation_desired']
        contact = np.zeros(14)
        observation = data['observation']
        o = np.ones((observation.shape[0], 4)).T
        o[:-1, :] = observation.T
        next_observation = np.matmul(subgoal, o).T[:, :-1].squeeze()
        subgoal_quat = R.from_matrix(subgoal[:-1, :-1]).as_quat()
        subgoal = np.concatenate((subgoal[:-1, -1], subgoal_quat), axis=0)


        observation = observation[::int(observation.shape[0]/100)][:100]
        next_observation = next_observation[::int(next_observation.shape[0]/100)][:100]

        o_mean, o_mean_ = np.mean(observation, axis=0), np.mean(next_observation, axis=0)
        observation = observation - o_mean
        next_observation = next_observation - o_mean_

        observation = np.concatenate((observation, np.tile(o_mean, (observation.shape[0], 1))), axis=1)
        next_observation = np.concatenate((next_observation, np.tile(o_mean_, (next_observation.shape[0], 1))), axis=1)

        return subgoal, contact, observation, next_observation, action

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)

