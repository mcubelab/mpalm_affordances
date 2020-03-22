import os
import sys
import time
import argparse
import numpy as np
import rospy
import signal
import threading
from IPython import embed

from airobot import Robot
from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet
import pybullet as p

from helper import util
from macro_actions import ClosedLoopMacroActions, YumiGelslimPybulet
from closed_loop_eval import SingleArmPrimitives, DualArmPrimitives

from closed_loop_experiments_cfg import get_cfg_defaults
from data_tools.proc_gen_cuboids import CuboidSampler
from primitive_data_gen import MultiBlockManager, GoalVisual
import simulation


def signal_handler(sig, frame):
    """
    Capture exit signal from the keyboard
    """
    print('Exit')
    sys.exit(0)


class MultiStepManager(object):
    def __init__(self, cfg, grasp_cfg, pull_cfg):
        self.grasp_cfg = grasp_cfg
        self.pull_cfg = pull_cfg
        self.cfg = cfg

        # mappings between the stable orientation indices for grasping/pulling
        self.grasp2pull = cfg.GRASP_TO_PULL
        self.pull2grasp = cfg.PULL_TO_GRASP

    def get_args(self, exp_single, exp_double, goal_face, 
                 start_face=None, execute=False):
        """
        Get the primitive arguments for a multi step plan
        """
        if start_face is None:
            start_face_index = np.random.randint(
                len(self.grasp_cfg.VALID_GRASP_PAIRS[goal_face]))
            start_face = self.grasp_cfg.VALID_GRASP_PAIRS[goal_face][start_face_index]

        grasp_args = exp_double.get_random_primitive_args(
            ind=start_face,
            random_goal=True,
            execute=execute)
        pull_args_start = exp_single.get_random_primitive_args(
            ind=self.grasp2pull[start_face],
            random_goal=True,
            execute=execute)
        pull_args_goal = exp_single.get_random_primitive_args(
            ind=self.grasp2pull[goal_face],
            random_goal=True,
            execute=execute)

        pull_args_start['object_pose2_world'] = grasp_args['object_pose1_world']
        pull_args_goal['object_pose1_world'] = grasp_args['object_pose2_world']

        full_args = [pull_args_start, grasp_args, pull_args_goal]
        return full_args


def main(args):
    pull_cfg_file = os.path.join(args.example_config_path, 'pull') + ".yaml"
    pull_cfg = get_cfg_defaults()
    pull_cfg.merge_from_file(pull_cfg_file)
    pull_cfg.freeze()

    grasp_cfg_file = os.path.join(args.example_config_path, 'grasp') + ".yaml"
    grasp_cfg = get_cfg_defaults()
    grasp_cfg.merge_from_file(grasp_cfg_file)
    grasp_cfg.freeze()

    # cfg = grasp_cfg
    cfg = pull_cfg

    rospy.init_node('MultiStep')
    signal.signal(signal.SIGINT, signal_handler)

    data_seed = args.np_seed
    np.random.seed(data_seed)

    yumi_ar = Robot('yumi_palms',
                    pb=True,
                    pb_cfg={'gui': args.visualize},
                    arm_cfg={'self_collision': False,
                             'seed': data_seed})

    r_gel_id = cfg.RIGHT_GEL_ID
    l_gel_id = cfg.LEFT_GEL_ID

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

    yumi_gs = YumiGelslimPybulet(
        yumi_ar,
        cfg,
        exec_thread=False
    )

    for _ in range(10):
        yumi_gs.update_joints(cfg.RIGHT_INIT + cfg.LEFT_INIT)

    obj_id = yumi_ar.pb_client.load_urdf(
        args.config_package_path +
        'descriptions/urdf/'+args.object_name+'.urdf',
        cfg.OBJECT_POSE_3[0:3],
        cfg.OBJECT_POSE_3[3:]
    )

    goal_obj_id = yumi_ar.pb_client.load_urdf(
        args.config_package_path +
        'descriptions/urdf/'+args.object_name+'_trans.urdf',
        cfg.OBJECT_POSE_3[0:3],
        cfg.OBJECT_POSE_3[3:]
    )
    p.setCollisionFilterPair(
        yumi_ar.arm.robot_id, goal_obj_id, r_gel_id, -1, enableCollision=False)
    p.setCollisionFilterPair(
        obj_id, goal_obj_id, -1, -1, enableCollision=False)
    p.setCollisionFilterPair(
        yumi_ar.arm.robot_id, obj_id, r_gel_id, -1, enableCollision=True)
    p.setCollisionFilterPair(
        yumi_ar.arm.robot_id, obj_id, 27, -1, enableCollision=True)

    yumi_ar.pb_client.reset_body(
        obj_id,
        cfg.OBJECT_POSE_3[:3],
        cfg.OBJECT_POSE_3[3:])

    yumi_ar.pb_client.reset_body(
        goal_obj_id,
        cfg.OBJECT_POSE_3[:3],
        cfg.OBJECT_POSE_3[3:])

    # manipulated_object = None
    # object_pose1_world = util.list2pose_stamped(cfg.OBJECT_INIT)
    # object_pose2_world = util.list2pose_stamped(cfg.OBJECT_FINAL)
    # palm_pose_l_object = util.list2pose_stamped(cfg.PALM_LEFT)
    # palm_pose_r_object = util.list2pose_stamped(cfg.PALM_RIGHT)

    # example_args = {}
    # example_args['object_pose1_world'] = object_pose1_world
    # example_args['object_pose2_world'] = object_pose2_world
    # example_args['palm_pose_l_object'] = palm_pose_l_object
    # example_args['palm_pose_r_object'] = palm_pose_r_object
    # example_args['object'] = manipulated_object
    # example_args['N'] = 60
    # example_args['init'] = True
    # example_args['table_face'] = 0

    primitive_name = args.primitive
    face = np.random.randint(6)
    # face = 3

    mesh_file = args.config_package_path + 'descriptions/meshes/objects/cuboids/' + \
        args.object_name + '_experiments.stl'
    cuboid_fname = mesh_file
    exp_single = SingleArmPrimitives(
        pull_cfg,
        yumi_ar.pb_client.get_client_id(),
        obj_id,
        mesh_file)

    exp_double = DualArmPrimitives(
        grasp_cfg,
        yumi_ar.pb_client.get_client_id(),
        obj_id,
        mesh_file,
        goal_face=face)

    action_planner = ClosedLoopMacroActions(
        cfg,
        yumi_gs,
        obj_id,
        yumi_ar.pb_client.get_client_id(),
        args.config_package_path,
        replan=args.replan,
        object_mesh_file=mesh_file
    )

    trans_box_lock = threading.RLock()
    goal_viz = GoalVisual(
        trans_box_lock,
        goal_obj_id,
        action_planner.pb_client,
        cfg.OBJECT_POSE_3)

    multistep_planner = MultiStepManager(cfg, grasp_cfg, pull_cfg)

    action_planner.update_object(obj_id, mesh_file)
    exp_single.initialize_object(obj_id, cuboid_fname)
    exp_double.initialize_object(obj_id, cuboid_fname, face)
    print('Reset multi block!')

    for _ in range(args.num_obj_samples):
        # get grasp sample
        # start_face_index = np.random.randint(len(grasp_cfg.VALID_GRASP_PAIRS[face]))
        # start_face = grasp_cfg.VALID_GRASP_PAIRS[face][start_face_index]

        # grasp_args = exp_double.get_random_primitive_args(ind=start_face, 
        #                                                   random_goal=True,
        #                                                   execute=False)
        # # pull_args_start = exp_single.get_random_primitive_args(ind=cfg.GRASP_TO_PULL[start_face], 
        #                                                        random_goal=True)
        # pull_args_goal = exp_single.get_random_primitive_args(ind=cfg.GRASP_TO_PULL[face], 
        #                                                       random_goal=True)

        # pull_args_start['object_pose2_world'] = grasp_args['object_pose1_world']
        # pull_args_goal['object_pose1_world'] = grasp_args['object_pose2_world']

        # full_args = [pull_args_start, grasp_args, pull_args_goal]
        # full_args = [grasp_args]

        full_args = multistep_planner.get_args(
            exp_single,
            exp_double,
            face,
            execute=False
        )

        # obj_pose_final = exp_running.goal_pose_world_frame_mod
        # palm_poses_obj_frame = {}
        # y_normals = action_planner.get_palm_y_normals(palm_poses_world)
        # delta = 2e-3
        # for key in palm_poses_world.keys():
        #     palm_pose_world = palm_poses_world[key]
            
        #     # try to penetrate the object a small amount
        #     palm_pose_world.pose.position.x -= delta*y_normals[key].pose.position.x
        #     palm_pose_world.pose.position.y -= delta*y_normals[key].pose.position.y
        #     palm_pose_world.pose.position.z -= delta*y_normals[key].pose.position.z
            
        #     palm_poses_obj_frame[key] = util.convert_reference_frame(
        #         palm_pose_world, obj_pose_world, util.unit_pose())
            
        valid_subplans = 0
        valid_plans = []
        for plan_args in full_args:
            plan = action_planner.get_primitive_plan(
                plan_args['name'], plan_args, 'right')
            goal_viz.update_goal_state(
                util.pose_stamped2list(plan_args['object_pose2_world'])
            )
            start_pose = plan_args['object_pose1_world']
            goal_pose = plan_args['object_pose2_world']

            if args.debug:
                for _ in range(10):
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
                primitive_name = plan_args['name']
                subplan_valid = action_planner.full_mp_check(plan, primitive_name)
                if subplan_valid:
                    print("subplan valid!")
                    valid_subplans += 1
                    valid_plans.append(plan)

        if valid_subplans == len(full_args) and not args.debug:
            yumi_ar.pb_client.reset_body(
                obj_id, 
                util.pose_stamped2list(full_args[0]['object_pose1_world'])[:3],
                util.pose_stamped2list(full_args[0]['object_pose1_world'])[3:])                
            for plan_args in full_args:
                primitive_name = plan_args['name']
                time.sleep(0.1)
                for _ in range(10):
                    if primitive_name == 'pull':
                        yumi_gs.update_joints(pull_cfg.RIGHT_INIT + pull_cfg.LEFT_INIT)
                    elif primitive_name == 'grasp':
                        yumi_gs.update_joints(grasp_cfg.RIGHT_INIT + grasp_cfg.LEFT_INIT)                                        
                try:
                    result = action_planner.execute(primitive_name, plan_args)
                    if result is not None:
                        print(str(result[0]))

                except ValueError as e:
                    print("Value error: ")
                    print(e)

                time.sleep(1.0)

                if primitive_name == 'pull':
                    pose = util.pose_stamped2list(yumi_gs.compute_fk(yumi_gs.get_jpos(arm='right')))
                    pos, ori = pose[:3], pose[3:]

                    # pose = yumi_gs.get_ee_pose()
                    # pos, ori = pose[0], pose[1]
                    # pos[2] -= 0.0714
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
                # here should add box to scene, and plan a collision free motion to the init state

                # for _ in range(10):
                #     j_pos = cfg.RIGHT_INIT + cfg.LEFT_INIT
                #     for ind, jnt_id in enumerate(yumi_ar.arm.arm_jnt_ids):
                #         p.resetJointState(
                #             yumi_ar.arm.robot_id,
                #             jnt_id,
                #             targetValue=j_pos[ind]
                #         )

                # yumi_gs.update_joints(cfg.RIGHT_INIT + cfg.LEFT_INIT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--num_trials',
        type=int
    )

    parser.add_argument(
        '--step_repeat',
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
        '--primitive',
        type=str,
        default='pull',
        help='which primitive to plan')

    args = parser.parse_args()
    main(args)
