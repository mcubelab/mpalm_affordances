import os, os.path as osp
import sys
import numpy as np
import random

from airobot import set_log_level, log_debug, log_info, log_warn, log_critical


class TaskSampler:
    def __init__(self, problems_dir, cfg):
        self.cfg = cfg
        self.problems_dir = problems_dir
        self.assets_dirname = osp.join(problems_dir, self.cfg.ASSETS_DIR)
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

        self.default_scene_pcd = self.fake_scene_pcd()

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

    def fake_shelf_pcd(self):
        """
        Function to create a shelf point cloud, if one is not included in the problem file

        Returns:
            np.ndarray: N x 3 array with points in shelf point cloud
        """
        raise NotImplementedError

    def fake_scene_pcd(self):
        """
        Function to create a full scene point cloud

        Returns:
            np.ndarray: N x 3 array with points in default scene point cloud 
        """
        pcd_data = np.load(osp.join(self.assets_dirname, self.cfg.DEFAULT_SCENE_POINTCLOUD_FILE))
        pointcloud = pcd_data['pcd']
        # TODO: add noise
        return pointcloud

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
        """
        Sample one of the tasks that has been pre-specified. Provides the information
        needed to input to the neural network to predict a plan skeleton, along with 
        the info needed to run the RPO planner (including object point cloud at start,
        full scene point cloud, desired transformation). Additionally, return all the 
        information needed to reset the simulation to the state which is represented by 
        the task (object geometry file, start/goal pose).

        Args:
            difficulty (str, optional): The difficulty level of the task to 
                sample ['easy', 'medium', 'hard']. Defaults to 'easy'.

        Returns:
            8-element tuple containing:
                np.ndarray - 6DOF start pose of the object, in [x,y,z,qx,qy,qz,qw] form
                np.ndarray - 6DOF goal pose of the object, in [x,y,z,qx,qy,qz,qw] form
                str - Filename of the object .stl file that was used for this sample
                np.ndarray - Start point cloud 
                np.ndarray - 6DOF transformation desired
                dict - keys 'table' and 'shelf', correspond to segmented parts of the pointcloud
                    correponding to discrete placement surfaces, with values which are np.ndarray
                    point clouds
                dict - keys 'start' and 'goal', with values which are strings representing
                    which surface the object started on and which one it ended on.
                np.ndarray - Full scene point cloud, with no segmentation applied (just cropping,
                    to focus on the region near the table)
        """
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

        # get scene pcd and process it to look like object is there (i.e., simulate occlusion)
        scene_pcd = self._process_scene_with_object(self.default_scene_pcd, pointcloud)

        return start_pose, goal_pose, fname, pointcloud, transformation_des, surfaces, task_surfaces, scene_pcd

    def _process_scene_with_object(self, scene_pcd, obj_pcd):
        """
        Internal function to apply some processing to the scene pointcloud, which may
        have been collected with no object in the scene, to look as if the start point cloud
        was included

        Args:
            scene_pcd (np.ndarray): Scene point cloud with no object (i.e., just table and shelf)
            obj_pcd (np.ndarray): Segmented object point cloud

        Returns:
            np.ndarray - New fused point cloud with both scene and object
        """
        ### option 1 ###
        # Find bottom points of object
        # Find nearby points in scene
        # Remove these nearby points in scene, to simulate occlusion

        ### other option ###
        # Project all of point cloud to the z-plane (just take (x, y) coords)
        # Get convex hull of these points
        # Check all points in the scene and remove the ones that are inside this convex hull
        # Can speed this up with some heuristics

        ### after simulating occlusion, add the object point cloud in ###
        # just conatenate
        obj_pcd = obj_pcd[::int(obj_pcd.shape[0]/100)][:100]
        full_pcd = np.concatenate((scene_pcd, obj_pcd), axis=0)
        full_pcd = full_pcd[::int(full_pcd.shape[0]/100)][:100]
        return full_pcd


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
