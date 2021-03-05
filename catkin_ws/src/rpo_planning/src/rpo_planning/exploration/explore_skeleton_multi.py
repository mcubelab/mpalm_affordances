import os
import os.path as osp
import sys
import time
import argparse
import numpy as np
import rospy
import rospkg 
import signal
import threading
import pickle
import open3d
import copy
import random
from multiprocessing import Pipe, Queue, Manager, Process

from airobot import Robot
from airobot.utils import common
from airobot import set_log_level, log_debug, log_info, log_warn, log_critical
import pybullet as p

from rpo_planning.utils import common as util
from rpo_planning.execution.motion_playback import OpenLoopMacroActions
from rpo_planning.config.base_skill_cfg import get_skill_cfg_defaults
from rpo_planning.config.explore_task_cfg import get_task_cfg_defaults
from rpo_planning.config.multistep_eval_cfg import get_multistep_cfg_defaults
from rpo_planning.config.explore_cfgs.default_skill_names import get_skillset_cfg
from rpo_planning.robot.multicam_env import YumiMulticamPybullet 
from rpo_planning.utils.object import CuboidSampler
from rpo_planning.utils.pb_visualize import GoalVisual
from rpo_planning.utils.data import MultiBlockManager
from rpo_planning.utils.motion import guard
from rpo_planning.utils.planning.pointcloud_plan import PointCloudNode
from rpo_planning.utils.planning.skeleton import SkillSurfaceSkeleton
from rpo_planning.utils.visualize import PCDVis, PalmVis
from rpo_planning.utils.exploration.task_sampler import TaskSampler
from rpo_planning.utils.exploration.skeleton_processor import (
    process_skeleleton_prediction, separate_skills_and_surfaces)
from rpo_planning.utils.exploration.replay_data import rpo_plan2lcm
from rpo_planning.utils.exploration.multiprocess_explore import PlanningWorkerManager

from rpo_planning.skills.samplers.pull import PullSamplerBasic, PullSamplerVAE
from rpo_planning.skills.samplers.push import PushSamplerBasic, PushSamplerVAE
from rpo_planning.skills.samplers.grasp import GraspSamplerBasic, GraspSamplerVAE
from rpo_planning.skills.primitive_skills import (
    GraspSkill, PullRightSkill, PullLeftSkill, PushRightSkill, PushLeftSkill
)
from rpo_planning.motion_planning.primitive_planners import (
    grasp_planning_wf, pulling_planning_wf, pushing_planning_wf
)
from rpo_planning.exploration.environment.play_env import PlayEnvironment, PlayObjects
from rpo_planning.exploration.skeleton_sampler import SkeletonSampler
from rpo_planning.pointcloud_planning.rpo_learner import PointCloudTreeLearner


class SkillExplorer(object):
    def __init__(self, skills):
        self.skills = skills

    def sample_skill(self, strategy=None):
        return np.random.choice(self.skills.keys(), 1)


def main(args):
    set_log_level('debug')

    rospack = rospkg.RosPack()
    rospy.init_node('MultiProcess')
    skillset_cfg = get_skillset_cfg()
    signal.signal(signal.SIGINT, util.signal_handler)

    np.random.seed(args.np_seed)

    # create manager for multiple RPO planning processes
    global_manager = Manager()
    global_result_queue = Queue()
    rpo_planning_manager = PlanningWorkerManager(
        global_result_queue=global_result_queue,
        global_manager=global_manager,
        skill_names=skillset_cfg.SKILL_NAMES,
        num_workers=args.num_workers)
    
    # create task sampler
    task_cfg_file = osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/config/task_cfgs', args.task_config_file + '.yaml')
    task_cfg = get_task_cfg_defaults()
    task_cfg.merge_from_file(task_cfg_file)
    task_cfg.freeze()
    task_sampler = TaskSampler(
        osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning', args.task_problems_path), 
        task_cfg)

    # interface to obtaining skeleton predictions from the NN (handles LCM message passing and skeleton processing)
    skeleton_policy = SkeletonSampler()

    # initialize all workers with starter task/prediction
    for worker_id in range(args.num_workers):
        # sample a task
        pointcloud, transformation_des, surfaces, task_surfaces = task_sampler.sample('easy')
        pointcloud_sparse = pointcloud[::int(pointcloud.shape[0]/100), :][:100]
        scene_pointcloud = np.concatenate(surfaces.values())

        # run the policy to get a skeleton
        predicted_skeleton, predicted_inds = skeleton_policy.predict(pointcloud_sparse, transformation_des)
        # log_debug('predicted: ', predicted_skeleton)

        _, predicted_surfaces = separate_skills_and_surfaces(predicted_skeleton, skillset_cfg)

        target_surfaces = []
        for surface_name in predicted_surfaces:
            surface_pcd = surfaces[surface_name]
            target_surfaces.append(surface_pcd)

        # create unified skeleton object (contains all info about target point clouds and skeleton for planner)
        plan_skeleton = SkillSurfaceSkeleton(
            predicted_skeleton,
            predicted_inds,
            skillset_cfg=skillset_cfg,
            skeleton_surface_pcds=target_surfaces,
        )

        # create dictionary with data for RPO planner
        planner_inputs = {}
        planner_inputs['pointcloud_sparse'] = pointcloud_sparse
        planner_inputs['pointcloud'] = pointcloud
        planner_inputs['transformation_des'] = transformation_des
        planner_inputs['plan_skeleton'] = plan_skeleton
        planner_inputs['max_steps'] = args.max_steps
        planner_inputs['timeout'] = 30

        # send inputs to planner
        rpo_planning_manager.put_worker_work_queue(worker_id, planner_inputs)
        rpo_planning_manager.sample_worker(worker_id)

    running = True
    # loop while we're training and running
    while running:
        ### poll the global queue
        if global_result_queue.empty():
            time.sleep(0.0001)
            continue
        else:
            ### get the results (should contain buffer data and worker id info)
            res = global_result_queue.get()

            ### get id info and buffer data
            res_worker_id = res['worker_id']
            res_data = res['data']
            if res_data is not None:
                log_debug('Got results from RPO planning worker ID: %d' % res_worker_id)

                ### add the data to the buffer
                skeleton_policy.add_to_replay_buffer(rpo_plan2lcm(res_data))

            ### get a new task and a new prediction
            pointcloud, transformation_des, surfaces, task_surfaces = task_sampler.sample('easy')
            pointcloud_sparse = pointcloud[::int(pointcloud.shape[0]/100), :][:100]
            scene_pointcloud = np.concatenate(surfaces.values())

            # run the policy to get a skeleton
            predicted_skeleton, predicted_inds = skeleton_policy.predict(pointcloud_sparse, transformation_des)
            # log_debug('Predicted plan skeleton: ', predicted_skeleton)

            _, predicted_surfaces = separate_skills_and_surfaces(predicted_skeleton)
            
            target_surfaces = []
            for surface_name in predicted_surfaces:
                surface_pcd = surfaces[surface_name]
                target_surfaces.append(surface_pcd)

            # create unified skeleton object (contains all info about target point clouds and skeleton for planner)
            plan_skeleton = SkillSurfaceSkeleton(
                predicted_skeleton,
                predicted_inds,
                skeleton_surface_pcds=target_surfaces
            )

            # create dictionary with data for RPO planner
            planner_inputs = {}
            planner_inputs['pointcloud_sparse'] = pointcloud_sparse
            planner_inputs['pointcloud'] = pointcloud
            planner_inputs['transformation_des'] = transformation_des
            planner_inputs['plan_skeleton'] = plan_skeleton
            planner_inputs['max_steps'] = args.max_steps
            planner_inputs['timeout'] = 30

            ### send inputs to planner
            rpo_planning_manager.put_worker_work_queue(res_worker_id, planner_inputs)
            rpo_planning_manager.sample_worker(worker_id)
    rpo_planning_manager.stop_all_workers()
    log_info('Ending!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # general environment/execution args
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('-v', '--visualize', action='store_true')
    parser.add_argument('--np_seed', type=int, default=0)
    parser.add_argument('--multi', action='store_true')
    parser.add_argument('--trimesh_viz', action='store_true')
    parser.add_argument('--ignore_physics', action='store_true')
    parser.add_argument('-r', '--rolling', type=float, default=0.0, help='rolling friction value for pybullet sim')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--camera_inds', nargs='+', type=int)
    parser.add_argument('--pcd_noise', action='store_true')
    parser.add_argument('--pcd_noise_std', type=float, default=0.0025)
    parser.add_argument('--pcd_noise_rate', type=float, default=0.00025)
    parser.add_argument('--pcd_scalar', type=float, default=0.9)

    # save data args
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--save_data_dir', type=str, default='play_transitions')
    parser.add_argument('--exp', type=str, default='debug')

    # configuration 
    parser.add_argument('--task_config_file', type=str, default='default_problems')
    parser.add_argument('--task_problems_path', type=str, default='data/training_tasks')
    parser.add_argument('--full_skeleton_prediction', action='store_true')
    parser.add_argument('--predict_while_planning', action='store_true')

    # point cloud planner args
    parser.add_argument('--max_steps', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=2)

    args = parser.parse_args()
    main(args)

