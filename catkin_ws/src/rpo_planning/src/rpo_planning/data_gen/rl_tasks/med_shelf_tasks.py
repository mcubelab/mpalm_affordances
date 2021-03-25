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
from rpo_planning.config.rl_task_gen_cfg import get_task_gen_cfg_defaults
from rpo_planning.robot.multicam_env import YumiMulticamPybullet 
from rpo_planning.utils.object import CuboidSampler
from rpo_planning.utils.pb_visualize import GoalVisual
from rpo_planning.utils.data import MultiBlockManager
from rpo_planning.utils.motion import guard
from rpo_planning.utils.planning.pointcloud_plan import PointCloudNode
from rpo_planning.utils.visualize import PCDVis, PalmVis

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


def signal_handler(sig, frame):
    """
    Capture exit signal from the keyboard
    """
    print('Exit')
    sys.exit(0)


def main(args):
    # example_config_path = osp.join(os.environ['CODE_BASE'], args.example_config_path)
    rospack = rospkg.RosPack()
    skill_config_path = osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/config/skill_cfgs')
    pull_cfg_file = osp.join(skill_config_path, 'pull') + ".yaml"
    pull_cfg = get_task_gen_cfg_defaults()
    pull_cfg.merge_from_file(pull_cfg_file)
    pull_cfg.freeze()

    cfg = pull_cfg

    rospy.init_node('GenShelfTasks')
    signal.signal(signal.SIGINT, signal_handler)

    data_seed = args.np_seed
    np.random.seed(data_seed)

    yumi_ar = Robot('yumi_palms',
                    pb=True,
                    pb_cfg={'gui': args.visualize,
                            'opengl_render': False},
                    arm_cfg={'self_collision': False,
                             'seed': data_seed})

    r_gel_id = cfg.RIGHT_GEL_ID
    l_gel_id = cfg.LEFT_GEL_ID
    table_id = cfg.TABLE_ID

    yumi_gs = YumiMulticamPybullet(
        yumi_ar,
        cfg,
        exec_thread=False,
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
        table_id=table_id,
        r_gel_id=r_gel_id,
        l_gel_id=l_gel_id)

    cuboid_fname = cuboid_manager.get_cuboid_fname()
    goal_visualization = True 

    obj_id, sphere_ids, mesh, goal_obj_id = \
        cuboid_sampler.sample_cuboid_pybullet(
            cuboid_fname,
            goal=goal_visualization,
            keypoints=False)

    cuboid_manager.filter_collisions(obj_id, goal_obj_id)

    if goal_visualization:
        trans_box_lock = threading.RLock()
        goal_viz = GoalVisual(
            trans_box_lock,
            goal_obj_id,
            yumi_ar.pb_client.get_client_id(),
            cfg.OBJECT_POSE_3,
            show_init=False)

    p.changeDynamics(
        obj_id,
        -1,
        lateralFriction=1.0
    )

    # prep visualization tools
    palm_mesh_file = osp.join(os.environ['CODE_BASE'],
                              cfg.PALM_MESH_FILE)
    table_mesh_file = osp.join(os.environ['CODE_BASE'],
                               cfg.TABLE_MESH_FILE)
    viz_palms = PalmVis(palm_mesh_file, table_mesh_file, cfg)
    viz_pcd = PCDVis()

    yumi_ar.pb_client.remove_body(obj_id)
    if goal_visualization:
        yumi_ar.pb_client.remove_body(goal_obj_id)

    # if args.bookshelf and args.demo:
    #     obs, pcd = yumi_gs.get_observation(
    #         obj_id=obj_id,
    #         robot_table_id=(yumi_ar.arm.robot_id, 28))
    #     shelf_pcd = open3d.geometry.PointCloud()
    #     shelf_pcd.points = open3d.utility.Vector3dVector(np.concatenate(obs['table_pcd_pts']))
    #     shelf_pointcloud = np.asarray(shelf_pcd.points)
    #     z_sort = np.sort(shelf_pointcloud[:, 2])[::-1]
    #     top_z_2 = z_sort[10]
    #     shelf_target_surface = shelf_pointcloud[np.where(shelf_pointcloud[:, 2] > 0.9*top_z_2)[0], :]
    #     target_surface_skeleton = [None, None, None, shelf_target_surface, shelf_target_surface]
    #     skeleton = ['pull_right', 'grasp', 'pull_right', 'grasp_pp', 'pull_left']

    if args.demo_type == 'cuboid_bookshelf':
        problems_file = osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning', args.planning_problems_dir, 'bookshelf_cuboid/bookshelf_problems_formatted.pkl')
    elif args.demo_type == 'bookshelf':
        problems_file = osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning', args.planning_problems_dir, 'bookshelf_1/bookshelf_problems_formatted.pkl')
    else:
        raise ValueError('Demo type not recognized')

    # define possible options for how task can be defined
    surfaces = {'table': 0, 'shelf': 1}
    surface_ids = {'table': cfg.TABLE_ID, 'shelf': cfg.SHELF_ID}
    surface_boundaries = {
        'table': {'x': [0.15, 0.325], 'y': [-0.4, 0.4]},
        'shelf': {'x': [0.485, 0.525], 'y': [-0.35, 0.35]}
    }

    # problems_file = osp.join(os.environ['CODE_BASE'], 'catkin_ws/src/primitives/planning_problems/stacking_cuboids_0/stacking_cuboids_problems_0_formatted.pkl')
    with open(problems_file, 'rb') as f:
        problems_data = pickle.load(f)

    prob_inds = np.arange(len(problems_data), dtype=np.int64).tolist()
    data_inds = np.arange(len(problems_data[0]['problems']), dtype=np.int64).tolist()

    task_save_dir = 'data/training_tasks/medium_rearrangement_problems'
    save_dir = osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning', task_save_dir)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    
    samples = 0

    while True:
        # prob_ind = 8
        # data_ind = 15

        # ### intro figure data:
        # obj_fname = osp.join(os.environ['CODE_BASE'], 'catkin_ws/src/config/descriptions/meshes/objects/cuboids/test_cuboid_smaller_4867.stl')
        # ###

        prob_ind = prob_inds[np.random.randint(len(prob_inds))]
        data_ind = data_inds[np.random.randint(len(data_inds))]

        problem_data = problems_data[prob_ind]['problems'][data_ind]
        stl_file = problems_data[prob_ind]['object_name'].split('catkin_ws/')[1]
        obj_fname = osp.join(os.environ['CODE_BASE'], 'catkin_ws', stl_file)
        obj_name = obj_fname.split('.stl')[0].split('/meshes/objects/cuboids/')[1]
        scale = problems_data[prob_ind]['object_scale']
        start_pose = problem_data['start_vis'].tolist()
        goal_pose = problem_data['goal_vis'].tolist()
        transformation_des = problem_data['transformation']

        # put object into work at start_pose, with known obj_fname
        yumi_ar.pb_client.remove_body(obj_id)
        if goal_visualization:
            yumi_ar.pb_client.remove_body(goal_obj_id)


        obj_id, sphere_ids, mesh, goal_obj_id = \
            cuboid_sampler.sample_cuboid_pybullet(
                obj_fname,
                goal=goal_visualization,
                keypoints=False,
                scale=scale)

        # sample which surface the object will start and end on
        start_surface, goal_surface = random.sample(surfaces.keys(), 1)[0], random.sample(surfaces.keys(), 1)[0]

        # sample position and orientation corresponding to this
        start_stable = random.sample(mesh.compute_stable_poses()[0], 1)[0]
        goal_stable = random.sample(mesh.compute_stable_poses()[0], 1)[0]

        # get positions and orientations
        sg_info = {
            'start': {'surface': start_surface, 'stable': start_stable, 'pos': None},
            'goal': {'surface': goal_surface, 'stable': goal_stable, 'pos': None}
        }
        for key, val in sg_info.items():
            x_lim, y_lim = surface_boundaries[val['surface']]['x'], surface_boundaries[val['surface']]['y']
            x_pos = np.random.random_sample() * (max(x_lim) - min(x_lim)) + min(x_lim)
            y_pos = np.random.random_sample() * (max(y_lim) - min(y_lim)) + min(y_lim)
            sg_info[key]['pos'] = [x_pos, y_pos]
            add_z = cfg.DEFAULT_Z if val['surface'] is 'table' else cfg.DEFAULT_SHELF_Z

            position = [x_pos, y_pos, val['stable'][2, -1] + add_z]
            orientation = util.pose_stamped2list(util.body_world_yaw(util.pose_from_matrix(val['stable'])))[3:]
            pose = position + orientation
            sg_info[key]['pose_list'] = pose
        
        print('Start: ', sg_info['start']['pose_list'], sg_info['start']['surface'])
        print('Goal: ', sg_info['goal']['pose_list'], sg_info['goal']['surface'])
        
        start_pose = sg_info['start']['pose_list']
        goal_pose = sg_info['goal']['pose_list']

        p.resetBasePositionAndOrientation(
            obj_id,
            start_pose[:3],
            start_pose[3:])
        
        if goal_visualization:
            goal_viz.update_goal_obj(goal_obj_id)
            goal_viz.update_goal_state(goal_pose)
            goal_viz.hide_goal_obj()
            cuboid_manager.filter_collisions(obj_id, goal_obj_id)

            time.sleep(0.25)

        p.changeDynamics(
            obj_id,
            -1,
            lateralFriction=1.0
        )
    

        yumi_ar.arm.set_jpos(cfg.RIGHT_OUT_OF_FRAME +
                                cfg.LEFT_OUT_OF_FRAME,
                                ignore_physics=True)

        time.sleep(0.25)

        real_start_pos = p.getBasePositionAndOrientation(obj_id)[0]
        real_start_ori = p.getBasePositionAndOrientation(obj_id)[1]
        real_start_pose = list(real_start_pos) + list(real_start_ori)
        start_pose = real_start_pose

        if goal_visualization:
            real_goal_pos = p.getBasePositionAndOrientation(goal_obj_id)[0]
            real_goal_ori = p.getBasePositionAndOrientation(goal_obj_id)[1]
            real_goal_pose = list(real_goal_pos) + list(real_goal_ori)

            transformation_des = util.matrix_from_pose(
                util.get_transform(util.list2pose_stamped(real_goal_pose), util.list2pose_stamped(real_start_pose))
            )

            goal_pose = real_goal_pose

        # get table observation
        obs, pcd = yumi_gs.get_observation(
            obj_id=obj_id,
            robot_table_id=(yumi_ar.arm.robot_id, table_id))

        pointcloud_pts = np.asarray(obs['down_pcd_pts'][:100, :], dtype=np.float32)
        pointcloud_pts_full = np.asarray(np.concatenate(obs['pcd_pts']), dtype=np.float32)
        table_surface = np.concatenate(obs['table_pcd_pts'], axis=0)[::500, :]
    
        # get shelf observation    
        obs, pcd = yumi_gs.get_observation(
            obj_id=obj_id,
            robot_table_id=(yumi_ar.arm.robot_id, cfg.SHELF_ID))
        shelf_pointcloud = np.concatenate(obs['table_pcd_pts'], axis=0)[::500, :]
        z_sort = np.sort(shelf_pointcloud[:, 2])[::-1]
        top_z_2 = z_sort[10]
        shelf_surface = shelf_pointcloud[np.where(shelf_pointcloud[:, 2] > 0.9*top_z_2)[0], :]

        # store sample                
        o = pointcloud_pts_full
        transformation_desired = transformation_des
        skeleton = [None] 
        surfaces = {'table': table_surface, 'shelf': shelf_surface}
        task_surfaces = {'start': sg_info['start']['surface'], 'goal': sg_info['goal']['surface']}

        sample_fname = osp.join(save_dir, '%d.npz' % samples)
        
        if goal_visualization:
            goal_viz.update_goal_state(goal_pose)
            # goal_viz.show_goal_obj()
        
        if args.display_info:
            info_strings = {}
            info_strings['sample_fname'] = sample_fname
            info_strings['t_des'] = transformation_desired
            info_strings['skeleton'] = skeleton
            info_strings['start'] = start_pose
            info_strings['goal'] = goal_pose
            info_strings['stl'] = stl_file
            info_strings['start_surface'] = sg_info['start']['surface']
            info_strings['goal_surface'] = sg_info['goal']['surface']
            print('Sample info: ')
            for k, v in info_strings.items():
                print(k, v)
            print('\n\n\n')
        
        np.savez(sample_fname,
            observation=o,
            transformation_desired=transformation_desired,
            skeleton=skeleton,
            start_pose=start_pose,
            goal_pose=goal_pose,
            stl_file=stl_file,
            surfaces=surfaces,
            task_surfaces=task_surfaces
        )

        samples += 1
        if samples > args.total_samples:
            break
    
    print('done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--planning_problems_dir', type=str, default='data/planning_problems')
    parser.add_argument('-v', '--visualize', action='store_true')
    parser.add_argument('--np_seed', type=int, default=0)
    parser.add_argument('--goal_viz', action='store_true')
    parser.add_argument('--demo_type', type=str, default='cuboid_bookshelf')
    parser.add_argument('--predict_skeleton', action='store_true')
    parser.add_argument('--display_info', action='store_true')
    parser.add_argument('--total_samples', type=int, default=1000)

    args = parser.parse_args()
    main(args)

