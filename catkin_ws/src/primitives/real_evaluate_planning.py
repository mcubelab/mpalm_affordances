import os
import os.path as osp
import sys
import time
import argparse
import numpy as np
import rospy
import signal
import threading
import pickle
import open3d
import trimesh
import copy

from airobot import Robot
from airobot.utils import common

sys.path.append(osp.join(os.environ['CODE_BASE'], 'catkin_ws/src/primitives'))

from helper import util
from macro_actions import OpenLoopMacroActions

from multistep_planning_eval_cfg import get_cfg_defaults
from data_gen_utils import YumiCamsGSReal
import simulation
from helper.aruco_pose import ArucoPose
from helper import registration as reg
from helper.pointcloud_planning import PointCloudTree
from helper.pointcloud_planning_utils import PointCloudNode
from helper.pull_samplers import PullSamplerBasic, PullSamplerVAEPubSub
from helper.grasp_samplers import GraspSamplerVAEPubSub, GraspSamplerBasic
from helper.push_samplers import PushSamplerVAEPubSub
from helper.skills import GraspSkill, PullRightSkill, PullLeftSkill, PushRightSkill, PushLeftSkill

from planning import grasp_planning_wf, pulling_planning_wf, pushing_planning_wf
from eval_utils.visualization_tools import PCDVis, PalmVis
from eval_utils.experiment_recorder import GraspEvalManager

def signal_handler(sig, frame):
    """Capture exit signal from keyboard

    Args:
        sig ([type]): [description]
        frame ([type]): [description]
    """
    print('Exit')
    sys.exit(0)


def main(args):
    signal.signal(signal.SIGINT, signal_handler)
    example_config_path = osp.join(os.environ['CODE_BASE'], args.example_config_path)
    # get cfgs for each primitive
    pull_cfg_file = osp.join(example_config_path, 'pull') + '.yaml'
    pull_cfg = get_cfg_defaults()
    pull_cfg.merge_from_file(pull_cfg_file)
    pull_cfg.freeze()

    grasp_cfg_file = osp.join(example_config_path, 'grasp') + '.yaml'
    grasp_cfg = get_cfg_defaults()
    grasp_cfg.merge_from_file(grasp_cfg_file)
    grasp_cfg.freeze()

    push_cfg_file = osp.join(example_config_path, 'push') + '.yaml'
    push_cfg = get_cfg_defaults()
    push_cfg.merge_from_file(push_cfg_file)
    push_cfg.freeze()

    # for now use pull as main cfg
    cfg = pull_cfg

    # create airobot interface
    yumi_ar = Robot('yumi_palms', pb=False)

    # initialize robot params
    yumi_ar.arm.right_arm.set_speed(100, 50)
    yumi_ar.arm.left_arm.set_speed(100, 50)
    yumi_ar.arm.right_arm.start_egm()
    yumi_ar.arm.left_arm.start_egm()

    # create YumiReal interface
    yumi_gs = YumiCamsGSReal(yumi_ar, cfg, n_cam=2)
    time.sleep(1.0)
    _, _ = yumi_gs.move_to_joint_target_mp(yumi_ar.arm.right_arm.cfgs.ARM.HOME_POSITION, yumi_ar.arm.left_arm.cfgs.ARM.HOME_POSITION, execute=True)
    time.sleep(1.0)

    # create action planner
    action_planner = OpenLoopMacroActions(cfg=cfg, robot=yumi_gs, pb=False)

    # directories used internally for hacky Python 2 to Python 3 pub/sub (get NN predictions using filesystem)
    pred_dir = cfg.PREDICTION_DIR
    obs_dir = cfg.OBSERVATION_DIR
    if not osp.exists(pred_dir):
        os.makedirs(pred_dir)
    if not osp.exists(obs_dir):
        os.makedirs(obs_dir)

    # setup samplers
    if args.baseline:
        print('LOADING BASELINE SAMPLERS')
        pull_sampler = PullSamplerBasic()
        grasp_sampler = GraspSamplerBasic(None)
        push_sampler = PushSamplerVAEPubSub(obs_dir=obs_dir, pred_dir=pred_dir)
    else:
        print('LOADING LEARNED SAMPLERS')
        pull_sampler = PullSamplerVAEPubSub(obs_dir=obs_dir, pred_dir=pred_dir)
        grasp_sampler = GraspSamplerVAEPubSub(default_target=None,
                                              obs_dir=obs_dir, pred_dir=pred_dir)
        push_sampler = PushSamplerVAEPubSub(obs_dir=obs_dir, pred_dir=pred_dir)

    # setup skills
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

    grasp_skill = GraspSkill(
        grasp_sampler,
        yumi_gs,
        grasp_planning_wf
    )

    grasp_pp_skill = GraspSkill(
        grasp_sampler,
        yumi_gs,
        grasp_planning_wf,
        pp=True
    )

    skills = {}
    skills['pull_right'] = pull_right_skill
    skills['pull_left'] = pull_left_skill
    skills['grasp'] = grasp_skill
    skills['grasp_pp'] = grasp_pp_skill
    skills['push_right'] = push_right_skill
    skills['push_left'] = push_left_skill

    # empty the directories so nothing is messed up from previous runs
    pred_fnames, obs_fnames = os.listdir(pred_dir), os.listdir(obs_dir)
    if len(pred_fnames) > 0:
        for fname in pred_fnames:
            os.remove(osp.join(pred_dir, fname))
    if len(obs_fnames) > 0:
        for fname in obs_fnames:
            os.remove(osp.join(obs_dir, fname))

    # prepare for visualization tools
    palm_mesh_file = osp.join(os.environ['CODE_BASE'], cfg.PALM_MESH_FILE)
    table_mesh_file = osp.join(os.environ['CODE_BASE'], cfg.TABLE_MESH_FILE)
    vis_palms = PalmVis(palm_mesh_file, table_mesh_file, cfg)

    # get aruco utils
    aruco = ArucoPose()

    transformation_des_last = np.eye(4)

    # get it going
    while not rospy.is_shutdown():
        ### setup task ###

        # make sure robot is at home
        time.sleep(1.0)
        _, _ = yumi_gs.move_to_joint_target_mp(yumi_ar.arm.right_arm.cfgs.ARM.HOME_POSITION, yumi_ar.arm.left_arm.cfgs.ARM.HOME_POSITION, execute=True)
        time.sleep(5.0)

        # get goal pose
        g = raw_input('***Waiting for you to move to target pose, or enter r if you want to repeat the last run...***')
        if not g == 'r':
            goal_pose = util.pose_from_matrix(aruco.get_current_pose())
            # obs_goal, pcd_goal = yumi_gs.get_observation()
            # pcd_goal.colors = open3d.utility.Vector3dVector(np.tile(np.array([0, 255, 0]), (np.asarray(pcd_goal.colors).shape[0], 1)))
            # obs_goal, _ = yumi_gs.get_observation()
            # goal_pts_full = np.asarray(np.concatenate(obs_goal['pcd_pts']), dtype=np.float32)
            # pcd_goal = trimesh.PointCloud(goal_pts_full)
            # pcd_goal.colors = np.tile(np.array([0, 255, 0]), (pcd_goal.vertices.shape[0], 1))

            # get start pose
            raw_input('***Waiting for you to move to start pose...***')
            start_pose = util.pose_from_matrix(aruco.get_current_pose())
            # obs_start, pcd_start = yumi_gs.get_observation()
            # pcd_start.colors = open3d.utility.Vector3dVector(np.tile(np.array([0, 0, 255]), (np.asarray(pcd_start.colors).shape[0], 1)))
            # obs_start, _ = yumi_gs.get_observation()
            # start_pts_full = np.asarray(np.concatenate(obs_start['pcd_pts']), dtype=np.float32)
            # pcd_start = trimesh.PointCloud(start_pts_full)
            # pcd_start.colors = np.tile(np.array([0, 0, 255]), (pcd_start.vertices.shape[0], 1))


            # compute transformation_des
            print('***Computing relative transformation from start to goal***')
            transformation_des_coarse = util.matrix_from_pose(util.get_transform(goal_pose, start_pose)) 
            transformation_des = copy.deepcopy(transformation_des_coarse)
            transformation_des_last = transformation_des
        else:
            transformation_des = transformation_des_last

        # use ICP to refine desired transformation
        # transformation_des = reg.full_registration_np(
        #     source_np=start_pts_full, 
        #     target_np=goal_pts_full, 
        #     init_trans=None)


        # transformation_des = reg.refine_registration(
        #     source=pcd_start, 
        #     target=pcd_goal, 
        #     init_trans=transformation_des_coarse,
        #     voxel_size=0.005,
        #     distance_threshold=0.005,
        #     max_iteration=500
        #     ).transformation

        obs, pcd = yumi_gs.get_observation()
        table_pts = np.concatenate(obs['table_pcd_pts'], axis=0)[::125, :]

        # obj_pcd_start = open3d.geometry.PointCloud()
        # obj_pcd_goal_coarse = open3d.geometry.PointCloud()
        # obj_pcd_goal_fine = open3d.geometry.PointCloud()
        # table_pcd = open3d.geometry.PointCloud()
        obj_pcd_start = trimesh.PointCloud(np.concatenate(obs['pcd_pts']))
        # obj_pcd_goal_coarse = trimesh.PointCloud(start_pts_full)
        obj_pcd_goal_fine = trimesh.PointCloud(np.concatenate(obs['pcd_pts']))
        table_pcd = trimesh.PointCloud(table_pts)        

        # obj_pcd_start.colors = np.asarray(pcd.colors)

        # obj_pcd_start.points = copy.deepcopy(pcd.points)
        # obj_pcd_start.colors = copy.deepcopy(pcd.colors)

        # obj_pcd_goal_coarse.points = copy.deepcopy(pcd.points)
        # obj_pcd_goal_coarse.colors = open3d.utility.Vector3dVector(np.tile(np.array([255, 255, 0]), (np.asarray(obj_pcd_goal_coarse.points).shape[0], 1)))
        # obj_pcd_goal_coarse.transform(transformation_des_coarse)
        # obj_pcd_goal_coarse.colors = np.tile(np.array([255, 255, 0]), (obj_pcd_goal_coarse.vertices.shape[0], 1))
        # obj_pcd_goal_coarse.apply_transform(transformation_des_coarse)        
        
        # obj_pcd_goal_fine.points = copy.deepcopy(pcd.points)
        # obj_pcd_goal_fine.colors = open3d.utility.Vector3dVector(np.tile(np.array([255, 0, 255]), (np.asarray(obj_pcd_goal_fine.points).shape[0], 1)))
        # obj_pcd_goal_fine.transform(transformation_des)
        obj_pcd_goal_fine.colors = np.tile(np.array([255, 0, 255]), (obj_pcd_goal_fine.vertices.shape[0], 1))
        obj_pcd_goal_fine.apply_transform(transformation_des)        
        
        table_pcd.points = open3d.utility.Vector3dVector(table_pts)
        # open3d.visualization.draw_geometries([pcd_start, pcd_goal, obj_pcd_start, obj_pcd_goal_coarse, obj_pcd_goal_fine, table_pcd])        
        scene = trimesh.Scene()
        # scene.add_geometry([pcd_start, pcd_goal, obj_pcd_start, obj_pcd_goal_coarse, obj_pcd_goal_fine, table_pcd])
        scene.add_geometry([obj_pcd_start, obj_pcd_goal_fine, table_pcd])
        scene.show()

        # transformation_des = copy.deepcopy(transformation_des_coarse)

        # get skeleton
        valid_skeleton = False
        skeleton = raw_input('***Please input desired plan skeleton, from options ["pg", "gp", "pgp"]***\n\n')
        while not valid_skeleton:
            if skeleton not in ['pg', 'gp', 'pgp']:
                skeleton = raw_input('***Skeleton not recognized, Please input desired plan skeleton, from options ["pg", "gp", "pgp"]***\n\n')
            else:
                valid_skeleton = True
        assert skeleton in ['pg', 'gp', 'pgp'], 'Skeleton not recognized, exiting'

        # setup skeleton
        target_surface_skeleton = None
        if skeleton == 'pg':
            skeleton = ['pull_right', 'grasp']
        elif skeleton == 'gp':
            skeleton = ['grasp', 'pull_right']
        elif skeleton == 'pgp':
            skeleton = ['pull_right', 'grasp', 'pull_right']
        else:
            raise ValueError('Unrecognized plan skeleton')

        # get observation
        obs, pcd = yumi_gs.get_observation(color_seg=args.color_seg)

        # process observation
        pointcloud_pts = np.asarray(obs['down_pcd_pts'][:100, :], dtype=np.float32)
        pointcloud_pts_full = np.asarray(np.concatenate(obs['pcd_pts']), dtype=np.float32)
        table_pts = np.concatenate(obs['table_pcd_pts'], axis=0)[::50, :]

        # only use table pointcloud during grasp sampling at the moment
        grasp_sampler.update_default_target(table_pts)

        # setup planner
        planner = PointCloudTree(
            start_pcd=pointcloud_pts,
            trans_des=transformation_des,
            skeleton=skeleton,
            skills=skills,
            max_steps=args.max_steps,
            start_pcd_full=pointcloud_pts_full,
            target_surfaces=target_surface_skeleton
        )

        plan_valid = raw_input('***Ready to plan, press enter to begin planning, or press q to quit***')
        if str(plan_valid) == 'q':
            print('Starting over')
            continue
        # plan, returns None if no plan found before timeout
        if args.no_skeleton:
            plan_total = planner.plan_max_length()
            skeleton = []
            if plan_total is not None:
                for node in plan_total:
                    if node.skill is not None:
                        skeleton.append(node.skill)
        else:
            plan_total = planner.plan()

        if plan_total is None:
            print('Could not find plan')
            continue

        plan = copy.deepcopy(plan_total[1:])

        # get palm poses from plan
        palm_pose_plan = []
        for i, node in enumerate(plan):
            palms = copy.deepcopy(node.palms)
            if 'pull' in skeleton[i]:
                palms[2] -= 0.0015
            palm_pose_plan.append(palms)

        # observe results
        full_plan = []
        for i in range(len(plan)):
            if 'pull' in skeleton[i]:
                local_plan = pulling_planning_wf(
                    util.list2pose_stamped(palm_pose_plan[i]),
                    util.list2pose_stamped(palm_pose_plan[i]),
                    util.pose_from_matrix(plan[i].transformation)
                )
            elif 'grasp' in skeleton[i]:
                local_plan = grasp_planning_wf(
                    util.list2pose_stamped(palm_pose_plan[i][7:]),
                    util.list2pose_stamped(palm_pose_plan[i][:7]),
                    util.pose_from_matrix(plan[i].transformation)
                )
            elif 'push' in skeleton[i]:
                local_plan = pushing_planning_wf(
                    util.list2pose_stamped(palm_pose_plan[i]),
                    util.list2pose_stamped(palm_pose_plan[i]),
                    util.pose_from_matrix(plan[i].transformation)
                )
            full_plan.append(local_plan)

        from IPython import embed
        embed()

        if args.trimesh_viz:
            for ind in range(len(plan_total) - 1):
                # ind = 0
                pcd_data = {}
                pcd_data['start'] = plan_total[ind].pointcloud_full
                pcd_data['object_pointcloud'] = plan_total[ind].pointcloud_full
                pcd_data['transformation'] = np.asarray(util.pose_stamped2list(util.pose_from_matrix(plan_total[ind+1].transformation)))
                pcd_data['contact_world_frame_right'] = np.asarray(plan_total[ind+1].palms[:7])
                if 'pull' in skeleton[ind]:
                    pcd_data['contact_world_frame_left'] = np.asarray(plan_total[ind+1].palms[:7])
                else:
                    pcd_data['contact_world_frame_left'] = np.asarray(plan_total[ind+1].palms[7:])
                scene = vis_palms.vis_palms_pcd(pcd_data, world=True, centered=False, corr=False)
                scene.show()

                # ind = 0
                pcd_data = {}
                pcd_data['start'] = plan_total[ind].pointcloud_full
                pcd_data['object_pointcloud'] = plan_total[ind].pointcloud_full
                pcd_data['transformation'] = np.asarray(util.pose_stamped2list(util.pose_from_matrix(plan_total[ind+1].transformation)))
                pcd_data['contact_world_frame_right'] = np.asarray(plan_total[ind+1].palms_raw[:7])
                if 'pull' in skeleton[ind]:
                    pcd_data['contact_world_frame_left'] = np.asarray(plan_total[ind+1].palms_raw[:7])
                else:
                    pcd_data['contact_world_frame_left'] = np.asarray(plan_total[ind+1].palms_raw[7:])
                scene = vis_palms.vis_palms_pcd(pcd_data, world=True, centered=False, corr=False)
                scene.show()                

        if args.rviz_viz:
            for ii, skill in enumerate(skeleton):
                local_plan = full_plan[ii]
                for i in range(10):
                    simulation.visualize_object(
                        start_pose,
                        filepath="package://config/descriptions/meshes/objects/realsense_box_experiments.stl",
                        name="/object_initial",
                        color=(1., 0., 0., 1.),
                        frame_id="/yumi_body",
                        scale=(1., 1., 1.))
                    simulation.visualize_object(
                        goal_pose,
                        filepath="package://config/descriptions/meshes/objects/realsense_box_experiments.stl",
                        name="/object_final",
                        color=(0., 0., 1., 1.),
                        frame_id="/yumi_body",
                        scale=(1., 1., 1.))
                    rospy.sleep(.1)
                simulation.simulate_palms(local_plan)

        plan_valid = raw_input('***Ready for execution, press enter to begin plan playback***, or press q to quit***\n')
        if str(plan_valid) == 'q':
            print('Starting over')
            continue        
        try:
            for i, skill in enumerate(skeleton):
                step = i+1
                next_step_str = '***Ready for step %d in skill execution, press enter to proceed***' % step
                raw_input(next_step_str)
                if 'left' in skill:
                    arm = 'left'
                    action_planner.active_arm = 'left'
                    action_planner.inactive_arm = 'right'
                else:
                    arm = 'right'
                    action_planner.active_arm = 'right'
                    action_planner.inactive_arm = 'left'

                if 'pull' in skill or 'push' in skill:
                    skill_cfg = pull_cfg if 'pull' in skill else push_cfg
                    _, _ = yumi_gs.move_to_joint_target_mp(skill_cfg.RIGHT_INIT, cfg.LEFT_INIT, execute=True)
                    time.sleep(5.0)
                    # make sure robot is actually at this configuration
                    yumi_ar.arm.set_jpos(skill_cfg.RIGHT_INIT + skill_cfg.LEFT_INIT, wait=True)
                    action_planner.playback_single_arm(skill, full_plan[i][0])
                    time.sleep(0.5)
                    action_planner.single_arm_retract(arm=arm, repeat=4)
                    _, _ = yumi_gs.move_to_joint_target_mp(skill_cfg.RIGHT_INIT, cfg.LEFT_INIT, execute=True)
                    time.sleep(3.0)
                elif 'grasp' in skill:
                    _, _ = yumi_gs.move_to_joint_target_mp(grasp_cfg.RIGHT_INIT, grasp_cfg.LEFT_INIT, execute=True)
                    time.sleep(9.0)
                    # make sure robot is actually at this configuration
                    yumi_ar.arm.set_jpos(grasp_cfg.RIGHT_INIT + grasp_cfg.LEFT_INIT, wait=True)
                    for k, subplan in enumerate(full_plan[i]):
                        time.sleep(1.0)
                        action_planner.playback_dual_arm('grasp', subplan, k)
                        time.sleep(1.0)
                    action_planner.dual_arm_retract(repeat=2)
                    _, _ = yumi_gs.move_to_joint_target_mp(grasp_cfg.RIGHT_INIT, grasp_cfg.LEFT_INIT, execute=True)                    
                    time.sleep(3.0)
        except ValueError as e:
            print(e)
            continue

        done_str = '***Done with execution, press enter to proceed to beginning and reset robot to home***'
        raw_input(done_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--num_trials',
        type=int
    )

    parser.add_argument(
        '--sim_step_repeat',
        type=int,
        default=10
    )

    parser.add_argument(
        '--save_data',
        action='store_true'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/'
    )

    parser.add_argument(
        '--experiment_name',
        type=str,
        default='sample_experiment'
    )

    parser.add_argument(
        '--config_package_path',
        type=str,
        default='catkin_ws/src/config/')

    parser.add_argument(
        '--example_config_path',
        type=str,
        default='catkin_ws/src/primitives/config')

    parser.add_argument(
        '--primitive',
        type=str,
        default='pull',
        help='which primitive to plan')

    parser.add_argument(
        '--simulate',
        type=int,
        default=1)

    parser.add_argument(
        '-o',
        '--object',
        action='store_true')

    parser.add_argument(
        '-re',
        '--replan',
        action='store_true',
        default=False)

    parser.add_argument(
        '--object_name',
        type=str,
        default='realsense_box')

    parser.add_argument(
        '-ex',
        '--execute_thread',
        action='store_true'
    )

    parser.add_argument(
        '--debug', action='store_true'
    )

    parser.add_argument(
        '-r', '--rolling',
        type=float, default=0.0,
        help='rolling friction value for pybullet sim'
    )

    parser.add_argument(
        '-v', '--visualize',
        action='store_true'
    )

    parser.add_argument(
        '--np_seed', type=int,
        default=0
    )

    parser.add_argument(
        '--multi', action='store_true'
    )

    parser.add_argument(
        '--num_obj_samples', type=int, default=10
    )

    parser.add_argument(
        '--num_blocks', type=int, default=20
    )

    parser.add_argument(
        '--nice_pull_release', action='store_true'
    )

    parser.add_argument(
        '--goal_viz', action='store_true'
    )

    parser.add_argument(
        '--trimesh_viz', action='store_true'
    )

    parser.add_argument(
        '--rviz_viz', action='store_true'
    )    

    parser.add_argument(
        '--skeleton', type=str, default='pg'
    )

    parser.add_argument(
        '--playback_num', type=int, default=1
    )

    parser.add_argument(
        '--exp_name', type=str, default='debug'
    )

    parser.add_argument(
        '--no_skeleton', action='store_true'
    )

    parser.add_argument(
        '--max_steps', type=int, default=5
    )

    parser.add_argument(
        '--ignore_physics', action='store_true'
    )

    parser.add_argument(
        '--baseline', action='store_true'
    )

    parser.add_argument(
        '--camera_inds', nargs='+', type=int
    )

    parser.add_argument(
        '--pcd_noise', action='store_true'
    )

    parser.add_argument(
        '--pcd_noise_std', type=float, default=0.0025
    )

    parser.add_argument(
        '--pcd_noise_rate', type=float, default=0.00025
    )

    parser.add_argument(
        '--failure_tracking', action='store_true'
    )

    parser.add_argument(
        '--demo', action='store_true'
    )

    parser.add_argument(
        '--bookshelf', action='store_true'
    )

    parser.add_argument(
        '--demo_type', type=str, default='cuboid_regular'
    )

    parser.add_argument(
        '--color_seg', action='store_true'
    )

    args = parser.parse_args()
    main(args)
