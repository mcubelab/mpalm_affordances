import os, os.path as osp
import sys
import numpy as np
import random


class TaskSampler:
    def __init__(self, problems_dir, cfg):
        self.cfg = cfg
        self.problems_dir = problems_dir
        self.easy_dirname = osp.join(problems_dir, self.cfg.EASY_PROBLEMS)
        self.medium_dirname = osp.join(problems_dir, self.cfg.MEDIUM_PROBLEMS)
        self.hard_dirname = osp.join(problems_dir, self.cfg.HARD_PROBLEMS)
 
        self.difficulties = ['easy', 'medium', 'hard']
        self.difficulties_kv = {}
        for i, diff in enumerate(self.difficulties):
            self.difficulties_kv[diff] = i

        self._setup_problems()
    
    def _setup_problems(self):
        self.easy_problems = []
        self.medium_problems = []
        self.hard_problems = []

        for fname in os.listdir(self.easy_dirname):
            full_fname = osp.join(self.easy_dirname, fname)
            self.easy_problems.append(full_fname)

        for fname in os.listdir(self.medium_dirname):
            full_fname = osp.join(self.medium_dirname, fname)
            self.medium_problems.append(full_fname)

        for fname in os.listdir(self.hard_dirname):
            full_fname = osp.join(self.hard_dirname, fname)
            self.hard_problems.append(full_fname)               

        self.problems = [
            self.easy_problems,
            self.medium_problems,
            self.hard_problems
        ]         

    def sample(self, difficulty='easy'):
        assert difficulty in self.difficulties, 'Difficulty not recognized'
        diff_idx = self.difficulties_kv[difficulty]
        problems = self.problems[diff_idx]

        problem = random.sample(problems, 1)[0]
        problem_data = np.load(problem, allow_pickle=True)
        pointcloud = problem_data['observation']  # point cloud
        transformation_des = problem_data['transformation_desired']  # desired transformation
        return pointcloud, transformation_des
            