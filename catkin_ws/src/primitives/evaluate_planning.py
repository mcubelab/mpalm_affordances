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
import copy
from IPython import embed

from airobot import Robot
from airobot.utils import common
import pybullet as p

from helper import util
from macro_actions import ClosedLoopMacroActions
from closed_loop_eval import SingleArmPrimitives, DualArmPrimitives

# from closed_loop_experiments_cfg import get_cfg_defaults
from multistep_planning_eval_cfg import get_cfg_defaults
from data_tools.proc_gen_cuboids import CuboidSampler
from data_gen_utils import YumiCamsGS, DataManager, MultiBlockManager, GoalVisual
import simulation
from helper import registration as reg
from helper.pointcloud_planning import (
    PointCloudNode, PointCloudTree,
    GraspSamplerVAEPubSub, PullSamplerBasic,
    GraspSkill, PullRightSkill)
from planning import grasp_planning_wf, pulling_planning_wf
from eval_utils.visualization_tools import PCDVis, PalmVis


def signal_handler(sig, frame):
    """
    Capture exit signal from the keyboard
    """
    print('Exit')
    sys.exit(0)


def correct_grasp_pos(contact_positions, pcd_pts):
    contact_world_frame_pred_r = contact_positions['right']
    contact_world_frame_pred_l = contact_positions['left']
    contact_world_frame_corrected = {}

    r2l_vector = contact_world_frame_pred_r - contact_world_frame_pred_l
    right_endpoint = contact_world_frame_pred_r + r2l_vector
    left_endpoint = contact_world_frame_pred_l - r2l_vector
    midpoint = contact_world_frame_pred_l + r2l_vector/2.0

    r_points_along_r2l = np.linspace(right_endpoint, midpoint, 200)
    l_points_along_r2l = np.linspace(midpoint, left_endpoint, 200)
    points = {}
    points['right'] = r_points_along_r2l
    points['left'] = l_points_along_r2l

    dists = {}
    dists['right'] = []
    dists['left'] = []

    inds = {}
    inds['right'] = []
    inds['left'] = []

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.concatenate(pcd_pts))
    # pcd.points = open3d.utility.Vector3dVector(pcd_pts)

    kdtree = open3d.geometry.KDTreeFlann(pcd)
    for arm in ['right', 'left']:
        for i in range(points[arm].shape[0]):
            pos = points[arm][i, :]

            nearest_pt_ind = kdtree.search_knn_vector_3d(pos, 1)[1][0]

#             dist = (np.asarray(pcd.points)[nearest_pt_ind] - pos).dot(np.asarray(pcd.points)[nearest_pt_ind] - pos)
            dist = np.asarray(pcd.points)[nearest_pt_ind] - pos


            inds[arm].append((i, nearest_pt_ind))
            dists[arm].append(dist.dot(dist))

    for arm in ['right', 'left']:
        min_ind = np.argmin(dists[arm])
#         print(min_ind)
#         print(len(inds[arm]))
        min_point_ind = inds[arm][min_ind][0]
#         nearest_pt_world = np.asarray(pcd.points)[min_point_ind]
        nearest_pt_world = points[arm][min_point_ind]
        contact_world_frame_corrected[arm] = nearest_pt_world

    return contact_world_frame_corrected


def main(args):
    cfg_file = osp.join(args.example_config_path, args.primitive) + ".yaml"
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    cfg.freeze()

    rospy.init_node('MacroActions')
    signal.signal(signal.SIGINT, signal_handler)

    data_seed = args.np_seed
    primitive_name = args.primitive

    pickle_path = osp.join(
        args.data_dir,
        primitive_name,
        args.experiment_name
    )

    if args.save_data:
        suf_i = 0
        original_pickle_path = pickle_path
        while True:
            if osp.exists(pickle_path):
                suffix = '_%d' % suf_i
                pickle_path = original_pickle_path + suffix
                suf_i += 1
                data_seed += 1
            else:
                os.makedirs(pickle_path)
                break

        if not osp.exists(pickle_path):
            os.makedirs(pickle_path)

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

    alpha = cfg.ALPHA
    K = cfg.GEL_CONTACT_STIFFNESS
    restitution = cfg.GEL_RESTITUION

    p.changeDynamics(
        yumi_ar.arm.robot_id,
        r_gel_id,
        restitution=restitution,
        contactStiffness=K,
        contactDamping=alpha*K,
        rollingFriction=args.rolling
    )

    p.changeDynamics(
        yumi_ar.arm.robot_id,
        l_gel_id,
        restitution=restitution,
        contactStiffness=K,
        contactDamping=alpha*K,
        rollingFriction=args.rolling
    )

    yumi_gs = YumiCamsGS(
        yumi_ar,
        cfg,
        exec_thread=False,
        sim_step_repeat=args.sim_step_repeat
    )

    for _ in range(10):
        yumi_gs.update_joints(cfg.RIGHT_INIT + cfg.LEFT_INIT)

    cuboid_sampler = CuboidSampler(
        osp.join(os.environ['CODE_BASE'], 'catkin_ws/src/config/descriptions/meshes/objects/cuboids/nominal_cuboid.stl'),
        pb_client=yumi_ar.pb_client)
    cuboid_fname_template = osp.join(os.environ['CODE_BASE'], 'catkin_ws/src/config/descriptions/meshes/objects/cuboids/')

    cuboid_manager = MultiBlockManager(
        cuboid_fname_template,
        cuboid_sampler,
        robot_id=yumi_ar.arm.robot_id,
        table_id=27,
        r_gel_id=r_gel_id,
        l_gel_id=l_gel_id)

    if args.multi:
        cuboid_fname = cuboid_manager.get_cuboid_fname()
        # cuboid_fname = '/root/catkin_ws/src/config/descriptions/meshes/objects/cuboids/test_cuboid_smaller_4479.stl'
    else:
        cuboid_fname = args.config_package_path + 'descriptions/meshes/objects/' + \
            args.object_name + '_experiments.stl'
    mesh_file = cuboid_fname
    print("Cuboid file: " + cuboid_fname)

    if args.goal_viz:
        goal_visualization = True
    else:
        goal_visualization = False

    obj_id, sphere_ids, mesh, goal_obj_id = \
        cuboid_sampler.sample_cuboid_pybullet(
            cuboid_fname,
            goal=goal_visualization,
            keypoints=False)

    cuboid_manager.filter_collisions(obj_id, goal_obj_id)

    p.changeDynamics(
        obj_id,
        -1,
        lateralFriction=1.0
    )

    # goal_face = 0
    goal_faces = [0, 1, 2, 3, 4, 5]
    from random import shuffle
    shuffle(goal_faces)
    goal_face = goal_faces[0]

    exp_single = SingleArmPrimitives(
        cfg,
        yumi_ar.pb_client.get_client_id(),
        obj_id,
        cuboid_fname)
    k = 0
    while True:
        k += 1
        if k > 10:
            print('FAILED TO BUILD GRASPING GRAPH')
            return
        try:
            exp_double = DualArmPrimitives(
                cfg,
                yumi_ar.pb_client.get_client_id(),
                obj_id,
                cuboid_fname,
                goal_face=goal_face)
            break
        except ValueError as e:
            print(e)
            yumi_ar.pb_client.remove_body(obj_id)
            if goal_visualization:
                yumi_ar.pb_client.remove_body(goal_obj_id)
            cuboid_fname = cuboid_manager.get_cuboid_fname()
            print("Cuboid file: " + cuboid_fname)

            obj_id, sphere_ids, mesh, goal_obj_id = \
                cuboid_sampler.sample_cuboid_pybullet(
                    cuboid_fname,
                    goal=goal_visualization,
                    keypoints=False)

            cuboid_manager.filter_collisions(obj_id, goal_obj_id)

            p.changeDynamics(
                obj_id,
                -1,
                lateralFriction=0.4)
    if primitive_name == 'grasp':
        exp_running = exp_double
    else:
        exp_running = exp_single

    action_planner = ClosedLoopMacroActions(
        cfg,
        yumi_gs,
        obj_id,
        yumi_ar.pb_client.get_client_id(),
        args.config_package_path,
        replan=args.replan,
        object_mesh_file=mesh_file
    )

    if goal_visualization:
        trans_box_lock = threading.RLock()
        goal_viz = GoalVisual(
            trans_box_lock,
            goal_obj_id,
            action_planner.pb_client,
            cfg.OBJECT_POSE_3)

    action_planner.update_object(obj_id, mesh_file)
    exp_single.initialize_object(obj_id, cuboid_fname)

    dynamics_info = {}
    dynamics_info['contactDamping'] = alpha*K
    dynamics_info['contactStiffness'] = K
    dynamics_info['rollingFriction'] = args.rolling
    dynamics_info['restitution'] = restitution

    data = {}
    data['saved_data'] = []
    data['metadata'] = {}
    data['metadata']['mesh_file'] = mesh_file
    data['metadata']['cfg'] = cfg
    data['metadata']['dynamics'] = dynamics_info
    data['metadata']['cam_cfg'] = yumi_gs.cam_setup_cfg
    data['metadata']['step_repeat'] = args.sim_step_repeat
    data['metadata']['seed'] = data_seed
    data['metadata']['seed_original'] = args.np_seed

    metadata = data['metadata']

    data_manager = DataManager(pickle_path)
    pred_dir = osp.join(os.environ['CODE_BASE'], cfg.PREDICTION_DIR)
    obs_dir = osp.join(os.environ['CODE_BASE'], cfg.OBSERVATION_DIR)

    if args.save_data:
        with open(osp.join(pickle_path, 'metadata.pkl'), 'wb') as mdata_f:
            pickle.dump(metadata, mdata_f)

    total_trials = 0
    successes = 0

    # prep visualization tools
    palm_mesh_file = osp.join(os.environ['CODE_BASE'],
                              cfg.PALM_MESH_FILE)
    table_mesh_file = osp.join(os.environ['CODE_BASE'],
                               cfg.TABLE_MESH_FILE)
    viz_palms = PalmVis(palm_mesh_file, table_mesh_file, cfg)
    viz_pcd = PCDVis()

    if args.skeleton == 'pg':
        skeleton = ['pull', 'grasp']
    elif args.skeleton == 'gp':
        skeleton = ['grasp', 'pull']
    elif args.skeleton == 'pgp':
        skeleton = ['pull', 'grasp', 'pull']
    else:
        raise ValueError('Unrecognized plan skeleton!')

    pull_sampler = PullSamplerBasic()
    grasp_sampler = GraspSamplerVAEPubSub(
        default_target=None,
        obs_dir=obs_dir,
        pred_dir=pred_dir
    )
    pull_skill = PullRightSkill(pull_sampler, yumi_gs, pulling_planning_wf)
    pull_skill_no_mp = PullRightSkill(pull_sampler, yumi_gs,
                                      pulling_planning_wf, ignore_mp=True)
    grasp_skill = GraspSkill(grasp_sampler, yumi_gs, grasp_planning_wf)
    skills = {}
    # skills['pull'] = pull_skill_no_mp
    skills['pull'] = pull_skill
    skills['grasp'] = grasp_skill

    problems_file = '/root/catkin_ws/src/primitives/data/planning/test_problems_0/demo_0.pkl'
    with open(problems_file, 'rb') as f:
        problems_data = pickle.load(f)

    prob_inds = np.arange(len(problems_data), dtype=np.int64).tolist()
    data_inds = np.arange(len(problems_data[0]['problems']), dtype=np.int64).tolist()

    for _ in range(len(problems_data)):
        # prob_ind = 3

        # obj_fname = str(osp.join(
        #     '/root/catkin_ws/src/config/descriptions/meshes/objects/cuboids',
        #     problems_data[prob_ind]['object_name']))

        # print(obj_fname)
        # for j, problem_data in enumerate(problems_data[prob_ind]['problems']):
        for _ in range(len(problems_data[0]['problems'])):
            prob_ind = prob_inds[np.random.randint(len(prob_inds))]
            data_ind = data_inds[np.random.randint(len(data_inds))]
            problem_data = problems_data[prob_ind]['problems'][data_ind]

            obj_fname = str(osp.join(
                '/root/catkin_ws/src/config/descriptions/meshes/objects/cuboids',
                problems_data[prob_ind]['object_name']))
            print(obj_fname)

            # get these to be lists
            start_pose = problem_data['start_vis'].tolist()

            # put object into work at start_pose, with known obj_fname
            yumi_ar.pb_client.remove_body(obj_id)
            if goal_visualization:
                yumi_ar.pb_client.remove_body(goal_obj_id)

            obj_id, sphere_ids, mesh, goal_obj_id = \
                cuboid_sampler.sample_cuboid_pybullet(
                    obj_fname,
                    goal=goal_visualization,
                    keypoints=False)
            if goal_visualization:
                goal_viz.update_goal_obj(goal_obj_id)
            cuboid_manager.filter_collisions(obj_id, goal_obj_id)
            exp_single.initialize_object(obj_id, obj_fname)

            p.resetBasePositionAndOrientation(
                obj_id,
                start_pose[:3],
                start_pose[3:])

            p.changeDynamics(
                obj_id,
                -1,
                lateralFriction=1.0
            )

            yumi_ar.arm.set_jpos(cfg.RIGHT_OUT_OF_FRAME +
                                 cfg.LEFT_OUT_OF_FRAME,
                                 ignore_physics=True)
            time.sleep(0.2)

            real_start_pos = p.getBasePositionAndOrientation(obj_id)[0]
            real_start_ori = p.getBasePositionAndOrientation(obj_id)[1]
            real_start_pose = list(real_start_pos) + list(real_start_ori)

            transformation_des = util.matrix_from_pose(
                util.list2pose_stamped(problem_data['transformation'].tolist()))

            goal_pose = util.pose_stamped2list(util.transform_pose(
                util.list2pose_stamped(real_start_pose),
                util.list2pose_stamped(problem_data['transformation'])
            ))

            ###
            if args.skeleton == 'pgp':
                while True:
                    x, y, dq = exp_single.get_rand_trans_yaw()

                    goal_pose_2_list = copy.deepcopy(goal_pose)
                    goal_pose_2_list[0] = x
                    goal_pose_2_list[1] = y
                    goal_pose_2_list[3:] = common.quat_multiply(dq, np.asarray(goal_pose[3:]))

                    if goal_pose_2_list[0] > 0.2 and goal_pose_2_list[0] < 0.4 and \
                            goal_pose_2_list[1] > -0.3 and goal_pose_2_list[1] < 0.1:
                        rot = common.quat2rot(dq)
                        T_2 = np.eye(4)
                        T_2[:-1, :-1] = rot
                        T_2[:2, -1] = [x-goal_pose[0], y-goal_pose[1]]
                        break

                goal_pose = goal_pose_2_list
                transformation_des = np.matmul(T_2, transformation_des)

            ###

            # # if skeleton is 'grasp' first, invert the desired trans and flip everything
            if args.skeleton == 'gp':
                transformation_des = np.linalg.inv(transformation_des)
                start_tmp = copy.deepcopy(start_pose)
                start_pose = goal_pose
                goal_pose = start_tmp

                p.resetBasePositionAndOrientation(
                    obj_id,
                    start_pose[:3],
                    start_pose[3:])

                real_start_pos = p.getBasePositionAndOrientation(obj_id)[0]
                real_start_ori = p.getBasePositionAndOrientation(obj_id)[1]
                real_start_pose = list(real_start_pos) + list(real_start_ori)

                time.sleep(0.5)
            # ###

            if goal_visualization:
                goal_viz.update_goal_state(goal_pose)

            # get observation
            obs, pcd = yumi_gs.get_observation(
                obj_id=obj_id,
                robot_table_id=(yumi_ar.arm.robot_id, table_id))

            pointcloud_pts = np.asarray(obs['down_pcd_pts'][:100, :], dtype=np.float32)
            pointcloud_pts_full = np.concatenate(obs['pcd_pts'])

            grasp_sampler.update_default_target(
                np.concatenate(obs['table_pcd_pts'], axis=0)[::500, :])

            # plan!
            planner = PointCloudTree(
                pointcloud_pts,
                transformation_des,
                skeleton,
                skills,
                start_pcd_full=pointcloud_pts_full)
            plan = planner.plan()
            if plan is None:
                print('Could not find plan')
                continue

            if args.trimesh_viz:
                # from multistep_planning_eval_cfg import get_cfg_defaults
                # import os.path as osp
                # from eval_utils.visualization_tools import PCDVis, PalmVis
                # cfg = get_cfg_defaults()
                # palm_mesh_file = osp.join(os.environ['CODE_BASE'],
                #                         cfg.PALM_MESH_FILE)
                # table_mesh_file = osp.join(os.environ['CODE_BASE'],
                #                         cfg.TABLE_MESH_FILE)
                # viz_palms = PalmVis(palm_mesh_file, table_mesh_file, cfg)
                # viz_pcd = PCDVis()

                # pcd_data = {}
                # pcd_data['start'] = pcd_pts
                # pcd_data['object_pointcloud'] = pcd_pts
                # pcd_data['transformation'] = np.asarray(util.pose_stamped2list(util.pose_from_matrix(new_state.transformation)))
                # pcd_data['contact_world_frame_right'] = np.asarray(new_state.palms[:7])
                # pcd_data['contact_world_frame_left'] = np.asarray(new_state.palms[:7])
                # scene = viz_palms.vis_palms_pcd(pcd_data, world=True, centered=False, corr=False)
                # scene.show()

                pcd_data = copy.deepcopy(problem_data)
                pcd_data['start'] = plan[-2].pointcloud_full
                pcd_data['object_pointcloud'] = plan[-2].pointcloud_full
                pcd_data['transformation'] = np.asarray(util.pose_stamped2list(util.pose_from_matrix(plan[-1].transformation)))
                pcd_data['contact_world_frame_right'] = np.asarray(plan[-1].palms[:7])
                pcd_data['contact_world_frame_left'] = np.asarray(plan[-1].palms[:7])
                scene = viz_palms.vis_palms_pcd(pcd_data, world=True, centered=False, corr=False)
                scene.show()

                pcd_data = copy.deepcopy(problem_data)
                pcd_data['start'] = plan[0].pointcloud_full
                pcd_data['object_pointcloud'] = plan[0].pointcloud_full
                pcd_data['transformation'] = np.asarray(util.pose_stamped2list(util.pose_from_matrix(plan[1].transformation)))
                pcd_data['contact_world_frame_right'] = np.asarray(plan[1].palms[:7])
                pcd_data['contact_world_frame_left'] = np.asarray(plan[1].palms[7:])
                scene = viz_palms.vis_palms_pcd(pcd_data, world=True, centered=False, corr=False)
                scene.show()

                embed()
            # execute plan if one is found...

            # ***INTERMEDIATE STEP***
            # sample a pull palm pose from the mesh, and add to plan
            # from plan of relative transformations, get plan of world frame poses
            pose_plan = [(real_start_pose, util.list2pose_stamped(real_start_pose))]
            for i in range(1, len(plan)):
                pose = util.transform_pose(pose_plan[i-1][1], util.pose_from_matrix(plan[i].transformation))
                pose_list = util.pose_stamped2list(pose)
                pose_plan.append((pose_list, pose))


            # get a palm pose from the mesh for each step in the plan that doesn't have one
            palm_pose_plan = []
            for i, node in enumerate(plan):
                palm_pose_plan.append(node.palms)

            # observe results
            full_plan = []
            for i in range(1, len(plan)):
                if skeleton[i-1] == 'pull':
                    local_plan = pulling_planning_wf(
                        util.list2pose_stamped(palm_pose_plan[i]),
                        util.list2pose_stamped(palm_pose_plan[i]),
                        util.pose_from_matrix(plan[i].transformation)
                    )
                elif skeleton[i-1] == 'grasp':
                    local_plan = grasp_planning_wf(
                        util.list2pose_stamped(palm_pose_plan[i][7:]),
                        util.list2pose_stamped(palm_pose_plan[i][:7]),
                        util.pose_from_matrix(plan[i].transformation)
                    )
                full_plan.append(local_plan)

            action_planner.active_arm = 'right'
            action_planner.inactive_arm = 'left'

            yumi_ar.pb_client.reset_body(obj_id, pose_plan[0][0][:3], pose_plan[0][0][3:])
            for i, skill in enumerate(skeleton):
                if skill == 'pull':
                    yumi_ar.arm.set_jpos(cfg.RIGHT_INIT + cfg.LEFT_INIT, ignore_physics=True)
                    time.sleep(0.5)
                    action_planner.playback_single_arm('pull', full_plan[i][0])
                    time.sleep(0.5)
                    action_planner.single_arm_retract()

                elif skill == 'grasp':
                    yumi_ar.arm.set_jpos([0.9936, -2.1848, -0.9915, 0.8458, 3.7618,  1.5486,  0.1127,
                                         -1.0777, -2.1187, 0.995, 1.002,  -3.6834,  1.8132,  2.6405],
                                         ignore_physics=True)
                    time.sleep(0.5)
                    for k, subplan in enumerate(full_plan[i]):
                        try:
                            action_planner.playback_dual_arm('grasp', subplan, k)
                        except ValueError as e:
                            print(e)
                            break
                        time.sleep(1.0)
            time.sleep(3.0)


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
        default='/root/catkin_ws/src/config/')

    parser.add_argument(
        '--example_config_path',
        type=str,
        default='config')

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
        '--skeleton', type=str, default='pg'
    )

    args = parser.parse_args()
    main(args)
