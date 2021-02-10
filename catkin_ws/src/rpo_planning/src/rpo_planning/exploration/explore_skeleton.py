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

from airobot import Robot
from airobot.utils import common
import pybullet as p

from rpo_planning.utils import common as util
from rpo_planning.execution.motion_playback import OpenLoopMacroActions
from rpo_planning.config.base_skill_cfg import get_skill_cfg_defaults
from rpo_planning.config.explore_task_cfg import get_task_cfg_defaults
from rpo_planning.config.multistep_eval_cfg import get_multistep_cfg_defaults
from rpo_planning.robot.multicam_env import YumiMulticamPybullet 
from rpo_planning.utils.object import CuboidSampler
from rpo_planning.utils.pb_visualize import GoalVisual
from rpo_planning.utils.data import MultiBlockManager
from rpo_planning.utils.motion import guard
from rpo_planning.utils.planning.pointcloud_plan import PointCloudNode
from rpo_planning.utils.visualize import PCDVis, PalmVis
from rpo_planning.utils.exploration.task_sampler import TaskSampler

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
# from replay_buffer import TransitionBuffer


class SkillExplorer(object):
    def __init__(self, skills):
        self.skills = skills

    def sample_skill(self, strategy=None):
        return np.random.choice(self.skills.keys(), 1)


def main(args):
    rospack = rospkg.RosPack()
    skill_config_path = osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/config/skill_cfgs')
    pull_cfg_file = osp.join(skill_config_path, 'pull') + ".yaml"
    pull_cfg = get_multistep_cfg_defaults()
    pull_cfg.merge_from_file(pull_cfg_file)
    pull_cfg.freeze()

    grasp_cfg_file = osp.join(skill_config_path, 'grasp') + ".yaml"
    grasp_cfg = get_multistep_cfg_defaults()
    grasp_cfg.merge_from_file(grasp_cfg_file)
    grasp_cfg.freeze()

    push_cfg_file = osp.join(skill_config_path, 'push') + ".yaml"
    push_cfg = get_multistep_cfg_defaults()
    push_cfg.merge_from_file(push_cfg_file)
    push_cfg.freeze()

    cfg = pull_cfg

    signal.signal(signal.SIGINT, util.signal_handler)
    rospy.init_node('PlayExplore')

    np.random.seed(args.np_seed)
    # initialize airobot and modify dynamics
    yumi_ar = Robot('yumi_palms',
                    pb=True,
                    pb_cfg={'gui': args.visualize,
                            'opengl_render': False},
                    arm_cfg={'self_collision': False,
                             'seed': args.np_seed})

    yumi_ar.arm.go_home(ignore_physics=True)

    yumi_gs = YumiMulticamPybullet(
        yumi_ar,
        cfg,
        exec_thread=False,
    )

    if args.sim:
        r_gel_id = cfg.RIGHT_GEL_ID
        l_gel_id = cfg.LEFT_GEL_ID
        table_id = cfg.TABLE_ID

        alpha = cfg.ALPHA
        K = cfg.GEL_CONTACT_STIFFNESS
        restitution = cfg.GEL_RESTITUION

        p.changeDynamics(
            yumi_ar.arm.robot_id,
            r_gel_id,
            lateralFriction=0.5,
            restitution=restitution,
            contactStiffness=K,
            contactDamping=alpha*K,
            rollingFriction=args.rolling
        )

        p.changeDynamics(
            yumi_ar.arm.robot_id,
            l_gel_id,
            lateralFriction=0.5,
            restitution=restitution,
            contactStiffness=K,
            contactDamping=alpha*K,
            rollingFriction=args.rolling
        )

        p.changeDynamics(
            yumi_ar.arm.robot_id,
            table_id,
            lateralFriction=0.1
        )

    if args.baseline:
        print('LOADING BASELINE SAMPLERS')
        pull_sampler = PullSamplerBasic()
        grasp_sampler = GraspSamplerBasic(None)
        push_sampler = PushSamplerVAE()
    else:
        print('LOADING LEARNED SAMPLERS')
        pull_sampler = PullSamplerVAE()
        push_sampler = PushSamplerVAE()
        grasp_sampler = GraspSamplerVAE(default_target=None)

    pull_right_skill = PullRightSkill(
        pull_sampler,
        yumi_gs,
        pulling_planning_wf,
        ignore_mp=False,
        avoid_collisions=True
    )

    pull_left_skill = PullLeftSkill(
        pull_sampler,
        yumi_gs,
        pulling_planning_wf,
        ignore_mp=False,
        avoid_collisions=True
    )

    push_right_skill = PushRightSkill(
        push_sampler,
        yumi_gs,
        pushing_planning_wf,
        ignore_mp=False,
        avoid_collisions=True
    )

    push_left_skill = PushLeftSkill(
        push_sampler,
        yumi_gs,
        pushing_planning_wf,
        ignore_mp=False,
        avoid_collisions=True
    )

    grasp_skill = GraspSkill(grasp_sampler, yumi_gs, grasp_planning_wf)
    grasp_pp_skill = GraspSkill(grasp_sampler, yumi_gs, grasp_planning_wf, pp=True)

    skills = {}
    skills['pull_right'] = pull_right_skill
    skills['pull_left'] = pull_left_skill
    # skills['grasp'] = grasp_skill
    # skills['grasp_pp'] = grasp_pp_skill
    # skills['push_right'] = push_right_skill
    # skills['push_left'] = push_left_skill

    # create exploring agent
    agent = SkillExplorer(skills)

    pb_info = None
    if args.sim:
        pb_info = {}
        pb_info['object_id'] = None
        pb_info['object_mesh_file'] = None
        pb_info['pb_client'] = yumi_ar.pb_client.get_client_id()

    action_planner = OpenLoopMacroActions(
        cfg,
        yumi_gs,
        pb=args.sim,
        pb_info=pb_info
    )

    cuboid_sampler = CuboidSampler(
        osp.join(
            os.environ['CODE_BASE'],
            'catkin_ws/src/config/descriptions/meshes/objects/cuboids/nominal_cuboid.stl'),
        pb_client=yumi_ar.pb_client)
    cuboid_fname_template = osp.join(
        os.environ['CODE_BASE'],
        'catkin_ws/src/config/descriptions/meshes/objects/cuboids/')
    cuboid_manager = MultiBlockManager(
        cuboid_fname_template,
        cuboid_sampler,
        robot_id=yumi_ar.arm.robot_id,
        table_id=cfg.TABLE_ID,
        r_gel_id=cfg.RIGHT_GEL_ID,
        l_gel_id=cfg.LEFT_GEL_ID)

    # visualization stuff
    palm_mesh_file = osp.join(os.environ['CODE_BASE'], cfg.PALM_MESH_FILE)
    table_mesh_file = osp.join(os.environ['CODE_BASE'], cfg.TABLE_MESH_FILE)
    viz_palms = PalmVis(palm_mesh_file, table_mesh_file, cfg)

    # create interface to make guarded movements
    guarder = guard.GuardedMover(robot=yumi_gs, pb_client=yumi_ar.pb_client.get_client_id(), cfg=cfg)

    # create manager
    env = PlayEnvironment(pb_client=yumi_ar.pb_client, robot=yumi_gs, cuboid_manager=cuboid_manager, cuboid_sampler=cuboid_sampler, target_surface_ids=[cfg.TABLE_ID])

    # sample some objects
    env.sample_objects(n=1)

    # instantiate objects in the world at specific poses
    env.initialize_object_states()
    # create task sampler
    task_cfg_file = osp.join(os.environ['CODE_BASE'], args.task_config_path, args.task_config_file + '.yaml')
    task_cfg = get_task_cfg_defaults()
    task_cfg.merge_from_file(task_cfg_file)
    task_cfg.freeze()
    task_sampler = TaskSampler(args.task_problems_path, task_cfg)

    from IPython import embed
    embed()

    # create replay buffer
    # experience_buffer = TransitionBuffer(size, observation_n, action_n, device, goal_n=None)
    
    skeleton_policy = SkeletonSampler()

    while True:
        # sample a task
        pointcloud, transformation_des = task_sampler.sample('easy')

        if args.full_skeleton_prediction:
            # run the policy to get a skeleton
            predicted_skeleton = skeleton_policy.predict(pointcloud, transformation_des)
        else:
            predicted_skeleton = None

        # setup planner
        planner = PointCloudTreeLearner(
            start_pcd=pointcloud,
            trans_des=transformation_des,
            skeleton=predicted_skeleton,
            skills=skills,
            max_steps=args.max_steps,
            start_pcd_full=None,
            target_surfaces=None,
            skeleton_policy=skeleton_policy
        )

        if predicted_skeleton is not None:
            plan, info = planner.plan()
        else:
            plan, info = planner.plan_max_length()

        # unpack info and transitions for saving in replay buffer
        for i in range(len(plan) - 1):
            observation = plan[i]['observation']
            action = plan[i]['action']
            next_observation = plan[i+1]['observation']
            reward = plan[i]['reward']
            done = plan[i]['done']
            des_goal = transformation_des
            ach_goal = plan[i]['achieved_goal']
            skeleton_policy.experience_buffer.append(observation, action, next_observation, reward, done, ach_goal, des_goal)

        # train models
        skeleton_policy.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--save_data_dir', type=str, default='play_transitions')
    parser.add_argument('--exp', type=str, default='debug')
    parser.add_argument('--config_package_path', type=str, default='catkin_ws/src/config/')
    parser.add_argument('--example_config_path', type=str, default='catkin_ws/src/primitives/config')
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
    parser.add_argument('--task_config_path', type=str, default='catkin_ws/src/primitives/task_config')
    parser.add_argument('--task_config_file', type=str, default='default_problems')
    parser.add_argument('--task_problems_path', type=str, default='data/training_tasks')

    args = parser.parse_args()
    main(args)
