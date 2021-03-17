import os, os.path as osp
import sys
import numpy as np
import random

from airobot import set_log_level, log_debug, log_info, log_warn, log_critical


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

    def fake_table_pcd(self):
        """
        Function to create a table point cloud, if one is not included in the problem file

        Returns:
            np.ndarray: N x 3 array with points in table point cloud
        """
        x_table, y_table = np.linspace(0, 0.5, 23), np.linspace(-0.4, 0.4, 23)
        xx, yy = np.meshgrid(x_table, y_table)
        table_pts = []
        for i in range(xx.shape[0]):
            for j in range(yy.shape[0]):
                pt = [xx[i, j], yy[i, j], np.random.random() * 0.002 - 0.001]
                table_pts.append(pt)
        table = np.asarray(table_pts)
        return table

    def sample(self, difficulty='easy'):
        assert difficulty in self.difficulties, 'Difficulty not recognized'
        diff_idx = self.difficulties_kv[difficulty]
        problems = self.problems[diff_idx]

        problem = random.sample(problems, 1)[0]
        problem_data = np.load(problem, allow_pickle=True)

        pointcloud = problem_data['observation']  # point cloud
        transformation_des = problem_data['transformation_desired']  # desired transformation
        # check if we have saved the data about different placement surfaces. If none, just use table
        if 'surfaces' in problem_data.files:
            surfaces = problem_data['surfaces'].item()
        else:
            log_warn('WARNING: Default option for shelf point cloud is same as table at the moment')
            surfaces = {'table': self.fake_table_pcd(), 'shelf': self.fake_table_pcd()}  # TODO: MAKE A DEFAULT OPTION FOR SHELF
        
        # check if we have saved the data about which placement surface was used for the task
        if 'task_surfaces' in problem_data.files:
            task_surfaces = problem_data['task_surfaces'].item()
        else:
            task_surfaces = {'start': 'table', 'goal': 'table'}
        return pointcloud, transformation_des, surfaces, task_surfaces
            
    def sample_full(self, difficulty='easy'):
        assert difficulty in self.difficulties, 'Difficulty not recognized'
        diff_idx = self.difficulties_kv[difficulty]
        problems = self.problems[diff_idx]

        problem = random.sample(problems, 1)[0]
        problem_data = np.load(problem, allow_pickle=True)

        fname = problem_data['stl_file'].item()
        start_pose = problem_data['start_pose']
        goal_pose = problem_data['goal_pose']

        pointcloud = problem_data['observation']  # point cloud
        transformation_des = problem_data['transformation_desired']  # desired transformation
        # check if we have saved the data about different placement surfaces. If none, just use table
        if 'surfaces' in problem_data.files:
            surfaces = problem_data['surfaces'].item()
        else:
            log_warn('WARNING: Default option for shelf point cloud is same as table at the moment')
            surfaces = {'table': self.fake_table_pcd(), 'shelf': self.fake_table_pcd()}  # TODO: MAKE A DEFAULT OPTION FOR SHELF
        
        # check if we have saved the data about which placement surface was used for the task
        if 'task_surfaces' in problem_data.files:
            task_surfaces = problem_data['task_surfaces'].item()
        else:
            task_surfaces = {'start': 'table', 'goal': 'table'}
        return start_pose, goal_pose, fname, pointcloud, transformation_des, surfaces, task_surfaces


if __name__ == "__main__":
    import rospkg
    from rpo_planning.config.explore_task_cfg import get_task_cfg_defaults
    rospack = rospkg.RosPack()
    # create task sampler
    task_cfg_file = osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/config/task_cfgs/default_problems.yaml')
    task_cfg = get_task_cfg_defaults()
    task_cfg.merge_from_file(task_cfg_file)
    task_cfg.freeze()
    task_sampler = TaskSampler(
        osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/data/training_tasks'), 
        task_cfg)

    from IPython import embed
    embed()
