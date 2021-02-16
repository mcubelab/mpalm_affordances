"Inspired by https://github.com/yusukeurakami/plan2explore-pytorch/blob/p2e/memory.py"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim

import numpy as np
import sys
import argparse
import time
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

sys.path.append('..')
from skeleton_utils.utils import prepare_sequence_tokens
from skeleton_utils.skeleton_globals import SOS_token, EOS_token, PAD_token


class TransitionBuffer(object):
    def __init__(self, size, observation_n, action_n, device, max_seq_length=5, goal_n=None):
        self.size = size
        self.device = device
        self.observation_n = observation_n
        self.action_n = action_n

        # self.observations = np.empty((size, observation_n), dtype=np.float32)
        # self.next_observations = np.empty((size, observation_n), dtype=np.float32)
        
        # observation size is N x M, due to using point clouds as input
        self.observations = np.empty((size, observation_n[0], observation_n[1]), dtype=np.float32)
        self.next_observations = np.empty((size, observation_n[0], observation_n[1]), dtype=np.float32)
        self.actions = np.empty((size, action_n), dtype=np.float32)
        self.rewards = np.empty((size, ), dtype=np.float32)
        self.not_done = np.empty((size, 1), dtype=np.float32)

        if goal_n is not None:
            self.desired_goals = np.empty((size, goal_n), dtype=np.float32)
            self.achieved_goals = np.empty((size, goal_n), dtype=np.float32)

        self.index = 0
        self.skill_index = 0
        self.full = False
        self.timesteps, self.episodes = 0, 0

        self.gc = True if goal_n is not None else False

        # RPO specific stuff
        # self.skill_params = deque(maxlen=size)
        self.skill_params = {}
        subgoal_n, contact_n, mask_n = 7, 14, observation_n[0]
        self.skill_params['subgoals'] = np.empty((size, subgoal_n), dtype=np.float32)
        self.skill_params['contacts'] = np.empty((size, contact_n), dtype=np.float32)
        self.skill_params['masks'] = np.empty((size, mask_n), dtype=np.float32)

    def append(self, observation, action, next_observation, reward, done, achieved_goal=None, desired_goal=None):
        '''
        Put new experience in the buffer, tracking the index of where we're at and
        if we're full or not
        '''
        self.observations[self.index] = observation
        self.actions[self.index] = action
        self.next_observations[self.index] = next_observation
        self.rewards[self.index] = reward
        self.not_done[self.index] = not done

        if self.gc and achieved_goal is not None and desired_goal is not None:
            self.desired_goals[self.index] = desired_goal
            self.achieved_goals[self.index] = achieved_goal

        self.index += 1
        if self.index >= self.size:
            self.index = 0
        self.full = self.full or self.index == 0

        self.timesteps += 1
        self.episodes = self.episodes + 1 if done else self.episodes

    def append_skill_params(self, contact, subgoal, mask):
        """
        Put new continuous skill parameters into the buffer
        """
        self.skill_params['contacts'][self.index] = contact
        self.skill_params['subgoals'][self.index] = subgoal
        self.skill_params['masks'][self.index] = mask

        self.skill_index += 1
        if self.skill_index >= self.size:
            self.skill_index = 0

    def _sample_index(self, sequence_length):
        '''
        Get a valid set of indices based on how many transitions we have and
        how long of a sequence we want to sample. In this case, "sequence_length"
        specified a maximum sequence length, since we may have targets in the buffer
        that we want to train on which are below this sequence length.
        '''
        valid = False
        max_index = self.size if self.full else self.index
        while not valid:
            index = np.random.randint(max_index - sequence_length)
            indices = np.arange(index, index + sequence_length)
            if self.gc:
                goals = self.desired_goals[indices]
                # valid = not self.index in indices[1:] and (goals == goals[0]).all()
                valid = not self.index in indices[1:]
            else:
                valid = not self.index in indices[1:]
        return indices

    def _get_batch(self, indices, n, sequence_length):
        ''' 
        Formatting data to put into the batch
        '''
        indices = indices.T.reshape(-1)  # TODO: check if we should have transpose here
        
        obs = self.observations[indices].reshape(sequence_length, n, self.observation_n[0], self.observation_n[1])
        next_obs = self.next_observations[indices].reshape(sequence_length, n, self.observation_n[0], self.observation_n[1])
        acts = self.actions[indices].reshape(sequence_length, n, -1)
        reward = self.rewards[indices].reshape(sequence_length, n, -1)
        not_done = self.not_done[indices].reshape(sequence_length, n, -1)
        if not self.gc:
            return obs, acts, next_obs, reward, not_done
        else:
            des_goal = self.desired_goals[indices].reshape(sequence_length, n, -1)
            ach_goal = self.achieved_goals[indices].reshape(sequence_length, n, -1)
            return obs, acts, next_obs, reward, not_done, des_goal, ach_goal

    def sample(self, n, sequence_length):
        '''
        Sample a batch of transitions. (n sequences, each of some sequence length)
        '''
        indices_list = []
        for _ in range(n):
            indices_list.append(self._sample_index(sequence_length))
            # indices_list.append(self._sample_index(sequence_length).tolist())
        indices = np.asarray(indices_list)
        batch = self._get_batch(indices, n, sequence_length)
        return [torch.as_tensor(item).to(self.device) for item in batch]

    def _sample_sg_index(self):
        """
        Get the indices corresponding to a random full sequence of (start, goal, actions)
        """
        valid = False
        max_index = self.size if self.full else self.index
        while not valid:
            cleaned_done_buffer = np.logical_not(np.round(self.not_done).astype(np.bool))
            dones = np.where(cleaned_done_buffer)[0]

            # sample full sequences by construction for now
            rand_index = np.random.randint(dones.shape[0])
            done_idx, start_idx = dones[rand_index] + 1, dones[rand_index - 1] + 1
            if done_idx > max_index or start_idx > max_index:
                continue
            indices = np.arange(start_idx, done_idx)
            valid = not self.index in indices[1:]
        # print('got indices: ', indices)
        return start_idx, done_idx

    def _get_sg_batch(self, sg_indices, n):
        ''' 
        Formatting (start, goal, action_seq) data to put into the batch. Must deal with
        variable length sequences here

        Args:
            sg_indices (np.ndarray): Each n x 2 array, where each row contains
                values indicating the start_index and goal_index to obtain from the buffers
            n (int): Number of sequences to obtain total (batch size)
        '''
        # just get start and goal for point clouds, and subgoal pose + contacts from first step
        obs = self.observations[sg_indices[:, 0]].reshape(n, self.observation_n[0], self.observation_n[1]) 
        goal_obs = self.observations[sg_indices[:, 1]].reshape(n, self.observation_n[0], self.observation_n[1])
        subgoal_pose = self.skill_params['subgoals'][sg_indices[:, 0]].reshape(n, -1)
        contact_pose = self.skill_params['contacts'][sg_indices[:, 0]].reshape(n, -1)
        
        # get all intermediate actions, TODO: find way to speed this up
        token_seq = []
        for i in range(sg_indices.shape[0]):
            indices = np.arange(sg_indices[i, 0], sg_indices[i, 1])
            tok_seq = torch.Tensor(self.actions[indices]).long()
            # tok_seq = self.actions[indices]
            token_seq.append(tok_seq)
        return subgoal_pose, contact_pose, obs, goal_obs, token_seq

    def sample_sg(self, n):
        """
        Sample a batch of transitions. (n sequences, each UP to some sequence length).
        Each transition only contains information about the start state of the sequence,
        the goal state of the transition (final state), and the actions that were taken
        to get from start to goal
        """
        indices_list = []
        for _ in range(n):
            indices_list.append(self._sample_sg_index())
        indices = np.asarray(indices_list)

        sg, c, o, o_, token_seq = self._get_sg_batch(indices, n)
        padded_seq_batch = torch.nn.utils.rnn.pad_sequence(
            token_seq, 
            batch_first=True,
            padding_value=PAD_token).to(self.device)
        batch = [
            torch.as_tensor(sg).to(self.device),
            torch.as_tensor(c).to(self.device),
            torch.as_tensor(o).to(self.device),
            torch.as_tensor(o_).to(self.device),
            padded_seq_batch
        ]
        return batch
