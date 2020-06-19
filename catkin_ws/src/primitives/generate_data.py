import os
import sys
import time
import argparse
import numpy as np
import rospy
import signal
import threading
import pickle
import open3d
import random
from IPython import embed

from airobot import Robot
import pybullet as p

from helper import util
from macro_actions import ClosedLoopMacroActions
from closed_loop_eval import SingleArmPrimitives, DualArmPrimitives

from closed_loop_experiments_cfg import get_cfg_defaults
from data_tools.proc_gen_cuboids import CuboidSampler
from data_gen_utils import YumiCamsGS, DataManager, MultiBlockManager, GoalVisual


def signal_handler(sig, frame):
    """
    Capture exit signal from the keyboard
    """
    print('Exit')
    sys.exit(0)


def main(args):
    cfg_file = os.path.join(args.example_config_path, args.primitive) + ".yaml"
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    cfg.freeze()

    rospy.init_node('MacroActions')
    signal.signal(signal.SIGINT, signal_handler)

    data_seed = args.np_seed
    primitive_name = args.primitive

    pickle_path = os.path.join(
        args.data_dir,
        primitive_name,
        args.experiment_name
    )

    # update save folder and seed if save folder already exists
    if args.save_data:
        suf_i = 0
        original_pickle_path = pickle_path
        while True:
            if os.path.exists(pickle_path):
                suffix = '_%d' % suf_i
                pickle_path = original_pickle_path + suffix
                suf_i += 1
                data_seed += 1
                time.sleep(1.0)
            else:
                os.makedirs(pickle_path)
                print('Saving in directory: ' + pickle_path)
                break

        if not os.path.exists(pickle_path):
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

    # set up cuboid sampler with directly that has .stl files
    cuboid_sampler = CuboidSampler(
        os.path.join(os.environ['CODE_BASE'], 'catkin_ws/src/config/descriptions/meshes/objects/cuboids/nominal_cuboid.stl'),
        pb_client=yumi_ar.pb_client)
    cuboid_fname_template = os.path.join(os.environ['CODE_BASE'], 'catkin_ws/src/config/descriptions/meshes/objects/cuboids/')

    # set up manager for sampling different blocks
    cuboid_manager = MultiBlockManager(
        cuboid_fname_template,
        cuboid_sampler,
        robot_id=yumi_ar.arm.robot_id,
        table_id=27,
        r_gel_id=r_gel_id,
        l_gel_id=l_gel_id)

    if args.multi:
        cuboid_fname = cuboid_manager.get_cuboid_fname()
    else:
        # cuboid_fname = args.config_package_path + 'descriptions/meshes/objects/' + \
        #     args.object_name + '_experiments.stl'
        cuboid_fname = args.config_package_path + 'descriptions/meshes/objects/cuboids/test_cuboid_smaller_300.stl'
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
        lateralFriction=0.4
    )

    # goal_face = 0
    if args.single_face:
        goal_faces = [args.goal_face]
    else:
        goal_faces = [0, 1, 2, 3, 4, 5]
    random.shuffle(goal_faces)
    goal_face = goal_faces[0]
    print('GOAL FACE ORDER: ')
    print(goal_faces)

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
    
    penetrate_delta = 7e-3

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
    data['metadata']['penetration_delta'] = penetrate_delta

    metadata = data['metadata']


    data_manager = DataManager(pickle_path)

    if args.save_data:
        with open(os.path.join(pickle_path, 'metadata.pkl'), 'wb') as mdata_f:
            pickle.dump(metadata, mdata_f)

    total_trials = 0
    for _ in range(args.num_blocks):
        for goal_face in goal_faces:
            try:
                exp_double.initialize_object(obj_id, cuboid_fname, goal_face)
            except ValueError as e:
                print(e)
                print('Goal face: ' + str(goal_face))
                continue
            for _ in range(args.num_obj_samples):
                total_trials += 1
                start_face = None
                current_state_success = 0
                current_state_trial = 0
                while current_state_success < args.start_success:
                    if current_state_trial > args.start_trials:
                        print('Moving to new start state')
                        break
                    if primitive_name == 'grasp':
                        if current_state_trial == 0 or start_face is None:
                            start_pose = None
                            start_face = exp_double.get_valid_ind()
                            if start_face is None:
                                print('Could not find valid start face')
                                continue
                        plan_args = exp_double.get_random_primitive_args(ind=start_face,
                                                                         random_goal=True,
                                                                         execute=True,
                                                                         start_pose=start_pose,
                                                                         penetration_delta=penetrate_delta)
                    elif primitive_name == 'pull':
                        if current_state_trial == 0:
                            start_pose = None
                        plan_args = exp_single.get_random_primitive_args(ind=goal_face,
                                                                         random_goal=True,
                                                                         execute=True,
                                                                         start_pose=start_pose)

                    start_pose = plan_args['object_pose1_world']
                    goal_pose = plan_args['object_pose2_world']
                    current_state_trial += 1

                    if goal_visualization:
                        goal_viz.update_goal_state(util.pose_stamped2list(goal_pose))
                    if args.debug:
                        import simulation

                        plan = action_planner.get_primitive_plan(primitive_name, plan_args, 'right')
                        print('total trials: ' + str(total_trials))
                        for i in range(10):
                            simulation.visualize_object(
                                start_pose,
                                filepath="package://config/descriptions/meshes/objects/cuboids/" +
                                    cuboid_fname.split('objects/cuboids')[1],
                                name="/object_initial",
                                color=(1., 0., 0., 1.),
                                frame_id="/yumi_body",
                                scale=(1., 1., 1.))
                            simulation.visualize_object(
                                goal_pose,
                                filepath="package://config/descriptions/meshes/objects/cuboids/" +
                                    cuboid_fname.split('objects/cuboids')[1],
                                name="/object_final",
                                color=(0., 0., 1., 1.),
                                frame_id="/yumi_body",
                                scale=(1., 1., 1.))
                            rospy.sleep(.1)
                        simulation.simulate(plan, cuboid_fname.split('objects/cuboids')[1])
                    else:
                        success = False
                        try:
                            obs, pcd = yumi_gs.get_observation(
                                obj_id=obj_id,
                                robot_table_id=(yumi_ar.arm.robot_id, table_id))

                            obj_pose_world = start_pose
                            obj_pose_final = goal_pose

                            contact_obj_frame, contact_world_frame = {}, {}
                            contact_obj_frame_2, contact_world_frame_2 = {}, {}

                            contact_obj_frame['right'] = plan_args['palm_pose_r_object']
                            contact_world_frame['right'] = plan_args['palm_pose_r_world']
                            contact_obj_frame['left'] = plan_args['palm_pose_l_object']
                            contact_world_frame['left'] = plan_args['palm_pose_l_world']

                            contact_world_frame_2['right'] = util.convert_reference_frame(
                                util.list2pose_stamped(cfg.TIP_TIP2_TF),
                                util.unit_pose(),
                                contact_world_frame['right']
                            )
                            contact_world_frame_2['left'] = util.convert_reference_frame(
                                util.list2pose_stamped(cfg.TIP_TIP2_TF),
                                util.unit_pose(),
                                contact_world_frame['left']
                            )

                            contact_obj_frame_2['right'] = util.convert_reference_frame(
                                contact_world_frame_2['right'],
                                obj_pose_world,
                                util.unit_pose()
                            )
                            contact_obj_frame_2['left'] = util.convert_reference_frame(
                                contact_world_frame_2['left'],
                                obj_pose_world,
                                util.unit_pose()
                            )

                            start = util.pose_stamped2list(obj_pose_world)
                            goal = util.pose_stamped2list(obj_pose_final)

                            keypoints_start = np.array(exp_running.mesh_world.vertices.tolist())
                            keypoints_start_homog = np.hstack(
                                (keypoints_start, np.ones((keypoints_start.shape[0], 1)))
                            )

                            start_mat = util.matrix_from_pose(obj_pose_world)
                            goal_mat = util.matrix_from_pose(obj_pose_final)

                            T_mat = np.matmul(goal_mat, np.linalg.inv(start_mat))

                            contact_obj_frame_dict, contact_world_frame_dict = {}, {}
                            contact_obj_frame_2_dict, contact_world_frame_2_dict = {}, {}
                            nearest_pt_world_dict = {}
                            corner_norms = {}
                            down_pcd_norms = {}

                            if primitive_name == 'pull':
                                # active_arm, inactive_arm = action_planner.get_active_arm(
                                #     util.pose_stamped2list(obj_pose_world)
                                # )
                                active_arm = action_planner.active_arm
                                inactive_arm = action_planner.inactive_arm

                                contact_obj_frame_dict[active_arm] = util.pose_stamped2list(contact_obj_frame[active_arm])
                                contact_world_frame_dict[active_arm] = util.pose_stamped2list(contact_world_frame[active_arm])
                                contact_obj_frame_2_dict[active_arm] = util.pose_stamped2list(contact_obj_frame_2[active_arm])
                                contact_world_frame_2_dict[active_arm] = util.pose_stamped2list(contact_world_frame_2[active_arm])
                                # contact_pos = open3d.utility.DoubleVector(np.array(contact_world_frame_dict[active_arm][:3]))
                                # kdtree = open3d.geometry.KDTreeFlann(pcd)
                                # nearest_pt_ind = kdtree.search_knn_vector_3d(contact_pos, 1)[1][0]
                                # nearest_pt_world_dict[active_arm] = np.asarray(pcd.points)[nearest_pt_ind]
                                contact_pos = np.array(contact_world_frame_dict[active_arm][:3])

                                corner_dists = (np.asarray(keypoints_start) - contact_pos)
                                corner_norms[active_arm] = np.linalg.norm(corner_dists, axis=1)

                                down_pcd_dists = (obs['down_pcd_pts'] - contact_pos)
                                down_pcd_norms[active_arm] = np.linalg.norm(down_pcd_dists, axis=1)

                                contact_obj_frame_dict[inactive_arm] = None
                                contact_world_frame_dict[inactive_arm] = None
                                contact_obj_frame_2_dict[inactive_arm] = None
                                contact_world_frame_2_dict[inactive_arm] = None
                                nearest_pt_world_dict[inactive_arm] = None
                                corner_norms[inactive_arm] = None
                                down_pcd_norms[inactive_arm] = None
                            elif primitive_name == 'grasp':
                                for key in contact_obj_frame.keys():
                                    contact_obj_frame_dict[key] = util.pose_stamped2list(contact_obj_frame[key])
                                    contact_world_frame_dict[key] = util.pose_stamped2list(contact_world_frame[key])
                                    contact_obj_frame_2_dict[key] = util.pose_stamped2list(contact_obj_frame_2[key])
                                    contact_world_frame_2_dict[key] = util.pose_stamped2list(contact_world_frame_2[key])

                                    contact_pos = np.array(contact_world_frame_dict[key][:3])

                                    corner_dists = (np.asarray(keypoints_start) - contact_pos)
                                    corner_norms[key] = np.linalg.norm(corner_dists, axis=1)

                                    down_pcd_dists = (obs['down_pcd_pts'] - contact_pos)
                                    down_pcd_norms[key] = np.linalg.norm(down_pcd_dists, axis=1)

                            reached_start_pose = exp_running.get_obj_pose()[0]
                            reached_start_mat = util.matrix_from_pose(reached_start_pose)
                            result = action_planner.execute(primitive_name, plan_args)
                            if result is None:
                                continue
                            # print('Result: ' + str(result[0]) +
                            #       ' Pos Error: ' + str(result[1]) +
                            #       ' Ori Error: ' + str(result[2]))
                            if result[0] is True:

                                reached_goal_pose = exp_running.get_obj_pose()[0]
                                reached_goal_mat = util.matrix_from_pose(reached_goal_pose)
                                T_reached_mat = np.matmul(reached_goal_mat, np.linalg.inv(reached_start_mat))
                                keypoints_goal = np.matmul(T_reached_mat, keypoints_start_homog.T).T[:, :3]

                                # object-table interaction masks and other info
                                table_height_z = np.mean(np.concatenate(obs['table_pcd_pts'], axis=0)[:, 2])*1.5
                                pcd_pts = np.concatenate(obs['pcd_pts'], axis=0)
                                pcd_start_homog = np.hstack(
                                    (pcd_pts, np.ones((pcd_pts.shape[0], 1)))
                                )
                                down_pcd_start_homog = np.hstack(
                                    (obs['down_pcd_pts'], np.ones((obs['down_pcd_pts'].shape[0], 1)))
                                )
                                pcd_goal = np.matmul(T_reached_mat, pcd_start_homog.T).T[:, :3]
                                down_pcd_goal = np.matmul(T_reached_mat, down_pcd_start_homog.T).T[:, :3]

                                average_min_z = np.mean(pcd_goal[np.argsort(pcd_goal[:, 2])[:3500], 2])

                                pcd_table_inds = np.where(pcd_goal[:, 2] < average_min_z)[0]
                                pcd_table_mask = pcd_goal[:, 2] < average_min_z
                                down_pcd_table_inds = np.where(down_pcd_goal[:, 2] < average_min_z)[0]
                                down_pcd_table_mask = down_pcd_goal[:, 2] < average_min_z

                                pcd_table = open3d.geometry.PointCloud()
                                pcd_table.points = open3d.utility.Vector3dVector(np.concatenate(obs['table_pcd_pts'], axis=0))

                                table_kdtree = open3d.geometry.KDTreeFlann(pcd_table)

                                table_contact_inds = []
                                table_contact_pts = []
                                table_contact_mask = np.zeros(np.asarray(pcd_table.points).shape[0], dtype=np.bool)

                                for i, table_contact_pos in enumerate(pcd_goal[pcd_table_inds]):
                                    #nearest_pt_ind = table_kdtree.search_knn_vector_3d(table_contact_pos, 1)[1][0]
                                    nearest_pt_ind = table_kdtree.search_knn_vector_3d(table_contact_pos, 20)[1]
                                    nearest_pt = np.asarray(pcd_table.points)[nearest_pt_ind]
                                    table_contact_pts.append(nearest_pt)
                                    table_contact_inds.append(nearest_pt_ind)
                                    table_contact_mask[nearest_pt_ind] = True

                                print('Success: ' + str(result[0]) +
                                    ' Trial Number: ' + str(total_trials) + 
                                    ' State Success Number: ' + str(current_state_success))
                                sample = {}
                                sample['obs'] = obs
                                sample['start'] = start
                                sample['goal'] = goal
                                sample['keypoints_start'] = keypoints_start
                                sample['keypoints_goal'] = keypoints_goal
                                sample['keypoint_dists'] = corner_norms
                                sample['down_pcd_pts'] = obs['down_pcd_pts']
                                sample['down_pcd_normals'] = obs['down_pcd_normals']
                                sample['down_pcd_dists'] = down_pcd_norms
                                sample['transformation'] = util.pose_stamped2list(util.pose_from_matrix(T_mat))
                                sample['center_of_mass'] = obs['center_of_mass']
                                sample['down_pcd_mask'] = down_pcd_table_mask
                                sample['pcd_mask'] = pcd_table_mask
                                # sample['table_pcd_pts'] = np.asarray(pcd_table.points)
                                sample['table_contact_mask'] = table_contact_mask
                                sample['contact_obj_frame'] = contact_obj_frame_dict
                                sample['contact_world_frame'] = contact_world_frame_dict
                                sample['contact_obj_frame_2'] = contact_obj_frame_2_dict
                                sample['contact_world_frame_2'] = contact_world_frame_2_dict
                                # sample['contact_pcd'] = nearest_pt_world_dict
                                sample['result'] = result
                                sample['mesh_file'] = cuboid_fname
                                sample['goal_face'] = goal_face
                                sample['start_face'] = start_face

                                if primitive_name == 'grasp':
                                    sample['goal_face'] = exp_double.goal_face

                                current_state_success += 1
                                if args.save_data:
                                    fname = '%d_%d' % (total_trials, current_state_success)
                                    data_manager.save_observation(
                                        sample, 
                                        fname)

                        except ValueError as e:
                            print("Value error: ")
                            print(e)

                        if args.nice_pull_release:
                            time.sleep(1.0)

                            pose = util.pose_stamped2list(yumi_gs.compute_fk(yumi_gs.get_jpos(arm='right')))
                            pos, ori = pose[:3], pose[3:]

                            pos[2] += 0.001
                            r_jnts = yumi_gs.compute_ik(pos, ori, yumi_gs.get_jpos(arm='right'))
                            l_jnts = yumi_gs.get_jpos(arm='left')

                            if r_jnts is not None:
                                for _ in range(10):
                                    pos[2] += 0.001
                                    r_jnts = yumi_gs.compute_ik(pos, ori, yumi_gs.get_jpos(arm='right'))
                                    l_jnts = yumi_gs.get_jpos(arm='left')

                                    if r_jnts is not None:
                                        yumi_gs.update_joints(list(r_jnts) + l_jnts)
                                    time.sleep(0.1)

                        time.sleep(0.1)
                        for _ in range(10):
                            yumi_gs.update_joints(cfg.RIGHT_INIT + cfg.LEFT_INIT)

        while True:
            try:
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
                    lateralFriction=0.4
                )
                action_planner.update_object(obj_id, mesh_file)
                exp_single.initialize_object(obj_id, cuboid_fname)
                # exp_double.initialize_object(obj_id, cuboid_fname, goal_face)
                if goal_visualization:
                    goal_viz.update_goal_obj(goal_obj_id)
                break
            except ValueError as e:
                print(e)



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
        '--single_face', action='store_true'
    )

    parser.add_argument(
        '--goal_face', type=int, default=0
    )

    parser.add_argument(
        '--start_trials', type=int, default=15
    )

    parser.add_argument(
        '--start_success', type=int, default=20
    )

    args = parser.parse_args()
    main(args)
