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
import copy
from IPython import embed

from airobot import Robot
import pybullet as p

from helper import util
from macro_actions import ClosedLoopMacroActions, YumiGelslimPybulet
from closed_loop_eval import SingleArmPrimitives, DualArmPrimitives

from closed_loop_experiments_cfg import get_cfg_defaults
from data_tools.proc_gen_cuboids import CuboidSampler
from data_gen_utils import YumiCamsGS, DataManager, MultiBlockManager, GoalVisual
import simulation
from helper import registration as reg
from helper.pointcloud_planning import (
    PointCloudNode, PointCloudTree,
    GraspSamplerVAEPubSub, PullSamplerBasic,
    GraspSkill, PullSkill)
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

    if args.save_data:
        suf_i = 0
        original_pickle_path = pickle_path
        while True:
            if os.path.exists(pickle_path):
                suffix = '_%d' % suf_i
                pickle_path = original_pickle_path + suffix
                suf_i += 1
                data_seed += 1
            else:
                os.makedirs(pickle_path)
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

    cuboid_sampler = CuboidSampler(
        os.path.join(os.environ['CODE_BASE'], 'catkin_ws/src/config/descriptions/meshes/objects/cuboids/nominal_cuboid.stl'),
        pb_client=yumi_ar.pb_client)
    cuboid_fname_template = os.path.join(os.environ['CODE_BASE'], 'catkin_ws/src/config/descriptions/meshes/objects/cuboids/')

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
        lateralFriction=0.4
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
    pred_dir = os.path.join(os.getcwd(), 'predictions')
    obs_dir = os.path.join(os.getcwd(), 'observations')

    if args.save_data:
        with open(os.path.join(pickle_path, 'metadata.pkl'), 'wb') as mdata_f:
            pickle.dump(metadata, mdata_f)

    total_trials = 0
    successes = 0

    # prep visualization tools
    palm_mesh_file = '/root/catkin_ws/src/config/descriptions/meshes/mpalm/mpalms_all_coarse.stl'
    table_mesh_file = '/root/catkin_ws/src/config/descriptions/meshes/table/table_top.stl'
    viz_palms = PalmVis(palm_mesh_file, table_mesh_file, cfg)
    viz_pcd = PCDVis()

    # problem_folder = '/root/catkin_ws/src/primitives/data/planning/test_problems_0/demo_0'
    # for fname in os.listdir(problem_folder):
    #     problems = os.listdir(os.path.join(problem_folder, fname))
    #     # problem_number = problems[np.random.randint(len(problems))]
    #     for p_number in problems:
    #         problem_fname = os.path.join(problem_folder, fname, p_number)
    #         with open(problem_fname, 'rb') as f:
    #             problem_data = pickle.load(f)
    #         start_pose = problem_data['start_vis']
    #         goal_pose = problem_data['goal_vis']
    #         transformation_des = problem_data['transformation']

    #         # set simulator state

    skeleton = ['pull', 'grasp']
    pull_sampler = PullSamplerBasic()
    grasp_sampler = GraspSamplerVAEPubSub(
        default_target=None,
        obs_dir='/root/catkin_ws/src/primitives/observations',
        pred_dir='/root/catkin_ws/src/primitives/predictions'
    )
    pull_skill = PullSkill(pull_sampler, yumi_gs)
    grasp_skill = GraspSkill(grasp_sampler, yumi_gs, grasp_planning_wf)
    skills = {}
    skills['pull'] = pull_skill
    skills['grasp'] = grasp_skill

    problems_file = '/root/catkin_ws/src/primitives/data/planning/test_problems_0/demo_0.pkl'
    with open(problems_file, 'rb') as f:
        problems_data = pickle.load(f)

    # import random
    # random.shuffle(problems_data)

    for i in range(len(problems_data)):
        prob_ind = 1

        obj_fname = str(os.path.join(
            '/root/catkin_ws/src/config/descriptions/meshes/objects/cuboids',
            problems_data[prob_ind]['object_name']))

        print(obj_fname)
        for j, problem_data in enumerate(problems_data[prob_ind]['problems']):
            # get these to be lists
            start_pose = problem_data['start_vis'].tolist()
            goal_pose = problem_data['goal_vis'].tolist()

            # get this to be transformation matrix
            transformation_des = util.matrix_from_pose(
                util.list2pose_stamped(problem_data['transformation'].tolist()))

            # put object into work at start_pose, with known obj_fname
            yumi_ar.pb_client.remove_body(obj_id)
            obj_id, sphere_ids, mesh, goal_obj_id = \
                cuboid_sampler.sample_cuboid_pybullet(
                    obj_fname,
                    goal=goal_visualization,
                    keypoints=False)
            exp_single.initialize_object(obj_id, obj_fname)

            p.resetBasePositionAndOrientation(
                obj_id,
                start_pose[:3],
                start_pose[3:])

            # get observation
            obs, pcd = yumi_gs.get_observation(
                obj_id=obj_id,
                robot_table_id=(yumi_ar.arm.robot_id, table_id))

            pointcloud_pts = np.asarray(obs['down_pcd_pts'][:100, :], dtype=np.float32)
            pointcloud_pts_full = np.concatenate(obs['pcd_pts'])

            grasp_sampler.update_default_target(np.concatenate(obs['table_pcd_pts'], axis=0)[::500, :])

            # plan!
            planner = PointCloudTree(
                pointcloud_pts,
                transformation_des,
                skeleton,
                skills)
            plan = planner.plan()

            print(len(plan))
            pcd_data = copy.deepcopy(problem_data)
            pcd_data['start'] = plan[-2].pointcloud
            pcd_data['object_pointcloud'] = plan[-2].pointcloud
            pcd_data['transformation'] = np.asarray(util.pose_stamped2list(util.pose_from_matrix(plan[-1].transformation)))
            pcd_data['contact_world_frame_right'] = np.asarray(plan[-1].palms[:7])
            pcd_data['contact_world_frame_left'] = np.asarray(plan[-1].palms[7:])
            scene = viz_palms.vis_palms_pcd(pcd_data, world=True, centered=False, corr=False)
            scene.show()
            # embed()
            # execute plan if one is found...

            # ***INTERMEDIATE STEP***
            # sample a pull palm pose from the mesh, and add to plan
            # from plan of relative transformations, get plan of world frame poses
            pose_plan = [(start_pose, util.list2pose_stamped(start_pose))]
            for i in range(1, len(plan)):
                pose = util.transform_pose(pose_plan[i-1][1], util.pose_from_matrix(plan[i].transformation))
                pose_list = util.pose_stamped2list(pose)
                pose_plan.append((pose_list, pose))

            # get a palm pose from the mesh for each step in the plan that doesn't have one
            palm_pose_plan = []
            for i, node in enumerate(plan):
                if node.palms is None and i > 0:
                    # sample a palm pose from the pose_plan
                    if skeleton[i-1] == 'pull':
                        # exp_single.transform_mesh_world(new_pose=pose_plan[i-1][0])
                        # contact, normal, facet = exp_single.sample_contact()                        
                        contact, normal, facet = exp_single.sample_contact(new_pose=pose_plan[i-1][0])
                        palm_pose_world = exp_single.get_palm_poses_world_frame(contact, normal)
                        palm_pose_plan.insert(i, util.pose_stamped2list(palm_pose_world))
                    elif skeleton[i-1] == 'grasp':
                        pass
                else:
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
            # embed()
                
            action_planner.active_arm = 'right'
            action_planner.inactive_arm = 'left'             

            start_pose = pose_plan[0][1]
            goal_pose = pose_plan[1][1]
            cuboid_fname = obj_fname
            import simulation
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
            simulation.simulate_palms(full_plan[0], cuboid_fname.split('objects/cuboids')[1])
            # simulation.simulate(full_plan[0], cuboid_fname.split('objects/cuboids')[1])            

            time.sleep(1.0)

            start_pose = pose_plan[1][1]
            goal_pose = pose_plan[2][1]
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
            simulation.simulate_palms(full_plan[1], cuboid_fname.split('objects/cuboids')[1])
            # simulation.simulate(full_plan[1], cuboid_fname.split('objects/cuboids')[1])                

            embed()

            yumi_ar.pb_client.reset_body(obj_id, pose_plan[0][0][:3], pose_plan[0][0][3:])
            yumi_ar.arm.set_jpos(cfg.RIGHT_INIT + cfg.LEFT_INIT, ignore_physics=True)                                            

            # execute first step
            action_planner.execute_single_arm('pull', full_plan[0][0], pose_plan[1][0], 0)
            time.sleep(0.5)
            
            # nice pulling release
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
            time.sleep(0.5)
            
            # reset arms for grasping preparation (should do motion planning here)
            yumi_ar.arm.set_jpos([0.9936, -2.1848, -0.9915, 0.8458, 3.7618,  1.5486,  0.1127, -1.0777, -2.1187, 0.995, 1.002 ,  -3.6834,  1.8132,  2.6405], ignore_physics=True)                                            
            time.sleep(0.5)

            # execute each of the grasping steps, with a small delay in between
            action_planner.execute_two_arm('grasp', full_plan[1][0], pose_plan[2][0], 0)
            time.sleep(1.0)

            action_planner.execute_two_arm('grasp', full_plan[1][1], pose_plan[2][0], 1)
            time.sleep(1.0)

            action_planner.execute_two_arm('grasp', full_plan[1][2], pose_plan[2][0], 2)

            embed()


    # break

    for _ in range(args.num_blocks):
        # for goal_face in goal_faces:
        for _ in range(1):
            goal_face = np.random.randint(6)
            try:
                print('New object!')
                exp_double.initialize_object(obj_id, cuboid_fname, goal_face)
            except ValueError as e:
                print(e)
                print('Goal face: ' + str(goal_face))
                continue
            for _ in range(args.num_obj_samples):
                total_trials += 1
                if primitive_name == 'grasp':
                    start_face = exp_double.get_valid_ind()
                    if start_face is None:
                        print('Could not find valid start face')
                        continue
                    plan_args = exp_double.get_random_primitive_args(ind=start_face,
                                                                     random_goal=True,
                                                                     execute=True)
                elif primitive_name == 'pull':
                    plan_args = exp_single.get_random_primitive_args(ind=goal_face,
                                                                     random_goal=True,
                                                                     execute=True)

                start_pose = plan_args['object_pose1_world']
                goal_pose = plan_args['object_pose2_world']

                if goal_visualization:
                    goal_viz.update_goal_state(util.pose_stamped2list(goal_pose))
                if args.debug:
                    import simulation

                    plan = action_planner.get_primitive_plan(primitive_name, plan_args, 'right')

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
                    attempts = 0

                    while True:
                        attempts += 1
                        time.sleep(0.1)
                        for _ in range(10):
                            yumi_gs.update_joints(cfg.RIGHT_INIT + cfg.LEFT_INIT)

                        p.resetBasePositionAndOrientation(
                            obj_id,
                            util.pose_stamped2list(start_pose)[:3],
                            util.pose_stamped2list(start_pose)[3:])

                        if attempts > 15:
                            break
                        print('attempts: ' + str(attempts))
                        try:
                            obs, pcd = yumi_gs.get_observation(
                                obj_id=obj_id,
                                robot_table_id=(yumi_ar.arm.robot_id, table_id))

                            obj_pose_world = start_pose
                            obj_pose_final = goal_pose

                            start = util.pose_stamped2list(obj_pose_world)
                            goal = util.pose_stamped2list(obj_pose_final)

                            start_mat = util.matrix_from_pose(obj_pose_world)
                            goal_mat = util.matrix_from_pose(obj_pose_final)

                            T_mat = np.matmul(goal_mat, np.linalg.inv(start_mat))

                            transformation = np.asarray(util.pose_stamped2list(util.pose_from_matrix(T_mat)), dtype=np.float32)
                            # model takes in observation, and predicts:
                            pointcloud_pts = np.asarray(obs['down_pcd_pts'][:100, :], dtype=np.float32)
                            pointcloud_pts_full = np.concatenate(obs['pcd_pts'])
                            obs_fname = os.path.join(obs_dir, str(total_trials) + '.npz')
                            np.savez(
                                obs_fname,
                                pointcloud_pts=pointcloud_pts,
                                transformation=transformation
                            )

                            # embed()

                            got_file = False
                            pred_fname = os.path.join(pred_dir, str(total_trials) + '.npz')
                            start = time.time()
                            while True:
                                try:
                                    prediction = np.load(pred_fname)
                                    got_file = True
                                except:
                                    pass
                                if got_file or (time.time() - start > 300):
                                    break
                                time.sleep(0.01)
                            if not got_file:
                                wait = raw_input('waiting for predictions to come back online')
                                continue
                            os.remove(pred_fname)
                            # embed()

                            ind = np.random.randint(10)
                            ind_contact = np.random.randint(5)
                            # contact_obj_frame_r = prediction['prediction'][ind, :7]
                            # contact_obj_frame_l = prediction['prediction'][ind, 7:]

                            # get mask prediction from NN
                            mask_prediction = prediction['mask_predictions'][ind, :]

                            # use top scoring points as mask prediction
                            top_inds = np.argsort(mask_prediction)[::-1]
                            pred_mask = np.zeros_like(mask_prediction, dtype=bool)
                            pred_mask[top_inds[:15]] = True

                            # get masked points and run local registration
                            source = pointcloud_pts[np.where(pred_mask)[0], :]  # source is masked points
                            source_obj = np.concatenate(obs['pcd_pts'])  # source object is full pointcloud
                            # source_obj = new_data['start']
                            target = np.concatenate(obs['table_pcd_pts'])[::10]  # target is table pointcloud

                            # get initial guess based on grasping trans prior
                            init_trans_fwd = reg.init_grasp_trans(
                                source, fwd=True)
                            init_trans_bwd = reg.init_grasp_trans(
                                source, fwd=False)

                            # run registration
                            init_trans = init_trans_fwd
                            transform = copy.deepcopy(reg.full_registration_np(source, target, init_trans))
                            source_obj_trans = reg.apply_transformation_np(source_obj, transform)

                            # check if pointcloud after transform is below the table, and correct if needed
                            # if np.where(source_obj_trans[:, 2] < 0.01)[0].shape[0] > 100:
                            if np.mean(source_obj_trans, axis=0)[2] < 0.005:
                                print('Switching')
                                init_trans = init_trans_bwd
                                transform = copy.deepcopy(reg.full_registration_np(source, target, init_trans))

                            goal_mat_pred = np.matmul(transform, start_mat)
                            goal_pose_pred = util.pose_from_matrix(goal_mat_pred)

                            contact_prediction_r = prediction['palm_predictions'][ind, ind_contact, :7]
                            contact_prediction_l = prediction['palm_predictions'][ind, ind_contact, 7:]
                            contact_world_pos_r = contact_prediction_r[:3] + np.mean(pointcloud_pts, axis=0)
                            contact_world_pos_l = contact_prediction_l[:3] + np.mean(pointcloud_pts, axis=0)

                            contact_world_pos_pred = {}
                            contact_world_pos_pred['right'] = contact_world_pos_r
                            contact_world_pos_pred['left'] = contact_world_pos_l

                            contact_world_pos_corr = correct_grasp_pos(
                                contact_world_pos_pred,
                                obs['pcd_pts'])

                            contact_world_pos_r = contact_world_pos_corr['right']
                            contact_world_pos_l = contact_world_pos_corr['left']

                            contact_world_r = contact_world_pos_r.tolist() + contact_prediction_r[3:].tolist()
                            contact_world_l = contact_world_pos_l.tolist() + contact_prediction_l[3:].tolist()

                            palm_poses_world = {}
                            # palm_poses_world['right'] = util.convert_reference_frame(
                            #     util.list2pose_stamped(contact_obj_frame_r),
                            #     util.unit_pose(),
                            #     obj_pose_world)
                            # palm_poses_world['left'] = util.convert_reference_frame(
                            #     util.list2pose_stamped(contact_obj_frame_l),
                            #     util.unit_pose(),
                            #     obj_pose_world)
                            palm_poses_world['right'] = util.list2pose_stamped(contact_world_r)
                            palm_poses_world['left'] = util.list2pose_stamped(contact_world_l)

                            # obj_pose_final = self.goal_pose_world_frame_mod``
                            palm_poses_obj_frame = {}
                            # delta = 10e-3
                            penetration_delta = 7.5e-3
                            # delta = np.random.random_sample() * \
                            #     (penetration_delta - 0.5*penetration_delta) + \
                            #     penetration_delta
                            delta = penetration_delta
                            y_normals = action_planner.get_palm_y_normals(palm_poses_world)
                            for key in palm_poses_world.keys():
                                # try to penetrate the object a small amount
                                palm_poses_world[key].pose.position.x -= delta*y_normals[key].pose.position.x
                                palm_poses_world[key].pose.position.y -= delta*y_normals[key].pose.position.y
                                palm_poses_world[key].pose.position.z -= delta*y_normals[key].pose.position.z

                                palm_poses_obj_frame[key] = util.convert_reference_frame(
                                    palm_poses_world[key], obj_pose_world, util.unit_pose())

                            # plan_args['palm_pose_r_object'] = util.list2pose_stamped(contact_obj_frame_r)
                            # plan_args['palm_pose_l_object'] = util.list2pose_stamped(contact_obj_frame_l)
                            plan_args['palm_pose_r_object'] = palm_poses_obj_frame['right']
                            plan_args['palm_pose_l_object'] = palm_poses_obj_frame['left']
                            plan_args['object_pose2_world'] = goal_pose_pred


                            plan = action_planner.get_primitive_plan(primitive_name, plan_args, 'right')

                            # import simulation
                            # for i in range(10):
                            #     simulation.visualize_object(
                            #         start_pose,
                            #         filepath="package://config/descriptions/meshes/objects/cuboids/" +
                            #             cuboid_fname.split('objects/cuboids')[1],
                            #         name="/object_initial",
                            #         color=(1., 0., 0., 1.),
                            #         frame_id="/yumi_body",
                            #         scale=(1., 1., 1.))
                            #     simulation.visualize_object(
                            #         goal_pose,
                            #         filepath="package://config/descriptions/meshes/objects/cuboids/" +
                            #             cuboid_fname.split('objects/cuboids')[1],
                            #         name="/object_final",
                            #         color=(0., 0., 1., 1.),
                            #         frame_id="/yumi_body",
                            #         scale=(1., 1., 1.))
                            #     rospy.sleep(.1)
                            # simulation.simulate(plan, cuboid_fname.split('objects/cuboids')[1])
                            # continue

                            result = action_planner.execute(primitive_name, plan_args)
                            if result is None:
                                continue
                            print('Result: ' + str(result[0]) +
                                  ' Pos Error: ' + str(result[1]) +
                                  ' Ori Error: ' + str(result[2]))
                            if result[0]:
                                successes += 1
                                print('Success rate: ' + str(successes * 100.0 / total_trials))
                            break

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

    args = parser.parse_args()
    main(args)
