"Inspired by https://github.com/yusukeurakami/plan2explore-pytorch/blob/p2e/memory.py"
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim

import numpy as np
import sys
import argparse
import time
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class TransitionBuffer(object):
    def __init__(self, size, observation_n, action_n, device, goal_n=None):
        self.size = size
        self.device = device
        self.observation_n = observation_n
        self.action_n = action_n

        self.observations = np.empty((size, observation_n), dtype=np.float32)
        self.actions = np.empty((size, action_n), dtype=np.float32)
        self.next_observations = np.empty((size, observation_n), dtype=np.float32)
        self.rewards = np.empty((size, ), dtype=np.float32)
        self.not_done = np.empty((size, 1), dtype=np.float32)

        if goal_n is not None:
            self.desired_goals = np.empty((size, goal_n), dtype=np.float32)
            self.achieved_goals = np.empty((size, goal_n), dtype=np.float32)

        self.index = 0
        self.full = False
        self.timesteps, self.episodes = 0, 0

        self.gc = True if goal_n is not None else False

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

    def _sample_index(self, sequence_length):
        '''
        Get a valid set of indices based on how many transitions we have and
        how long of a sequence we want to sample
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
        indices = indices.T.reshape(-1)
        
        obs = self.observations[indices].reshape(sequence_length, n, -1)
        acts = self.actions[indices].reshape(sequence_length, n, -1)
        next_obs = self.next_observations[indices].reshape(sequence_length, n, -1)
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
        indices = []
        for _ in range(n):
            indices.append(self._sample_index(sequence_length))
        indices = np.asarray(indices)
        batch = self._get_batch(indices, n, sequence_length)
        return [torch.as_tensor(item).to(self.device) for item in batch]
