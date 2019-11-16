from planning import pushing_planning, grasp_planning, levering_planning, pulling_planning
from helper import util, planning_helper, collisions

import os
from example_config import get_cfg_defaults
import copy

from airobot import Robot
from airobot.utils import pb_util
import time
import argparse
import numpy as np

import pickle

data = {}
data['palm_pose_world'] = []
data['object_pose_palm'] = []
data['contact_bool'] = []


def get_joint_poses(tip_poses, robot, cfg, nullspace=True):
    tip_to_wrist = util.list2pose_stamped(cfg.TIP_TO_WRIST_TF, '')
    world_to_world = util.unit_pose()

    r_joints, l_joints = None, None

    wrist_left = util.convert_reference_frame(
        tip_to_wrist,
        world_to_world,
        tip_poses[0],
        "yumi_body")
    wrist_right = util.convert_reference_frame(
        tip_to_wrist,
        world_to_world,
        tip_poses[1],
        "yumi_body")

    wrist_left = util.pose_stamped2list(wrist_left)
    wrist_right = util.pose_stamped2list(wrist_right)

    # r_joints = robot._accurate_ik(
    # 	wrist_right[0:3],
    # 	wrist_right[3:],
    # 	arm='right',
    # 	nullspace=nullspace)[:7]

    # l_joints = robot._accurate_ik(
    # 	wrist_left[0:3],
    # 	wrist_left[3:],
    # 	arm='left',
    # 	nullspace=nullspace)[7:]

    #     wrist_right[3:])

    # print("r_joints:")
    r_joints = robot.arm.compute_ik(
        wrist_right[0:3],
        wrist_right[3:],
        arm='right',
        ns=nullspace)

    # print("l_joints:")
    l_joints = robot.arm.compute_ik(
        wrist_left[0:3],
        wrist_left[3:],
        arm='left',
        ns=nullspace)

    # l_joints = robot.arm.left_arm.compute_ik(
    #     wrist_left[0:3],
    #     wrist_left[3:])

    # r_joints = robot.arm.right_arm.compute_ik(
    #     wrist_right[0:3],


    return r_joints, l_joints, wrist_right, wrist_left


def main(args):
    print(args)

    yumi = Robot('yumi',
                 pb=True,
                 arm_cfg={'render': True, 'self_collision': False})
    yumi.arm.go_home()

    cfg_file = os.path.join(args.example_config_path, args.primitive) + ".yaml"
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    cfg.freeze()
    print(cfg)

    manipulated_object = None
    object_pose1_world = util.list2pose_stamped(cfg.OBJECT_INIT)
    object_pose2_world = util.list2pose_stamped(cfg.OBJECT_FINAL)
    palm_pose_l_object = util.list2pose_stamped(cfg.PALM_LEFT)
    palm_pose_r_object = util.list2pose_stamped(cfg.PALM_RIGHT)

    if args.primitive == 'push':
        plan = pushing_planning(
            object=manipulated_object,
            object_pose1_world=object_pose1_world,
            object_pose2_world=object_pose2_world,
            palm_pose_l_object=palm_pose_l_object,
            palm_pose_r_object=palm_pose_r_object)

    elif args.primitive == 'grasp':
        plan = grasp_planning(
            object=manipulated_object,
            object_pose1_world=object_pose1_world,
            object_pose2_world=object_pose2_world,
            palm_pose_l_object=palm_pose_l_object,
            palm_pose_r_object=palm_pose_r_object)

    elif args.primitive == 'pivot':
        gripper_name = args.config_package_path + \
            'descriptions/meshes/mpalm/mpalms_all_coarse.stl'
        table_name = args.config_package_path + \
            'descriptions/meshes/table/table_top.stl'

        manipulated_object = collisions.CollisionBody(
            args.config_package_path + 'descriptions/meshes/objects/realsense_box_experiments.stl')

        plan = levering_planning(
            object=manipulated_object,
            object_pose1_world=object_pose1_world,
            object_pose2_world=object_pose2_world,
            palm_pose_l_object=palm_pose_l_object,
            palm_pose_r_object=palm_pose_r_object,
            gripper_name=gripper_name,
            table_name=table_name)

    elif args.primitive == 'pull':
        plan = pulling_planning(
            object=manipulated_object,
            object_pose1_world=object_pose1_world,
            object_pose2_world=object_pose2_world,
            palm_pose_l_object=palm_pose_l_object,
            palm_pose_r_object=palm_pose_r_object,
            arm='r')

    else:
        raise NotImplementedError

    object_loaded = False
    box_id = None

    from IPython import embed
    embed()

    for plan_dict in plan:
        for i, t in enumerate(plan_dict['t']):
            if i == 0 and not object_loaded and args.object:
                time.sleep(2.0)
                box_id = pb_util.load_urdf(
                    args.config_package_path+'descriptions/urdf/realsense_box.urdf',
                    cfg.OBJECT_INIT[0:3],
                    cfg.OBJECT_INIT[3:]
                )
                # box_id = pb_util.load_urdf(
                #     args.config_package_path+'descriptions/urdf/realsense_box.urdf',
                #     cfg.OBJECT_INIT[0:3],
                #     [0, 0, 0, 1]
                # )

                time.sleep(2.0)
                object_loaded = True

            tip_poses = plan_dict['palm_poses_world'][i]

            r_joints, l_joints, wrist_right, wrist_left = get_joint_poses(
                tip_poses,
                yumi,
                cfg,
                nullspace=True)

            # from IPython import embed
            # embed()

            loop_time = 0.125
            sleep_time = 0.005
            start = time.time()
            if args.primitive != 'push':
                # yumi.arm.set_ee_pose(
                #     wrist_right[0:3],
                #     wrist_right[3:],
                #     arm='right',
                #     wait=True)
                while (time.time() - start < loop_time):
                    # success = yumi.arm.set_jpos(
                    #     r_joints, 
                    #     arm='right', 
                    #     wait=True)
                    # time.sleep(sleep_time)
                    # yumi.arm.set_jpos(
                    #     l_joints, 
                    #     arm='left', 
                    #     wait=True)

                    yumi.arm.set_jpos(r_joints+l_joints, wait=False)
                    time.sleep(sleep_time)

                    # compliant_states = yumi.arm.p.getJointStates(
                    #     yumi.arm.robot_id, 
                    #     yumi.arm.right_arm.comp_jnt_ids)

                    # if box_id is not None:
                    #     pts = yumi.arm.p.getContactPoints(
                    #         bodyA=yumi.arm.robot_id, bodyB=box_id, linkIndexA=12)
                    #     contact_bool = 0 if len(pts) == 0 else 1
                    #     if not contact_bool:
                    #         print("not in contact!")
                    #     else:
                    #         print("---")

                    # object_pose_world = list(yumi.arm.p.getBasePositionAndOrientation(box_id, 0)[
                    #     0]) + list(yumi.arm.p.getBasePositionAndOrientation(box_id, 0)[1])
                    # object_pose_world = util.list2pose_stamped(object_pose_world)

                    # palm_frame_world = list(yumi.arm.p.getLinkState(yumi.arm.robot_id, 13)[
                    #                         0]) + list(yumi.arm.p.getLinkState(yumi.arm.robot_id, 13)[1])
                    # palm_frame_world = util.list2pose_stamped(palm_frame_world)

                    # object_pose_palm = util.convert_reference_frame(
                    #     object_pose_world, palm_frame_world, util.unit_pose())

                    # data['palm_pose_world'].append(
                    #     util.pose_stamped2list(palm_frame_world))
                    # data['object_pose_palm'].append(
                    #     util.pose_stamped2list(object_pose_palm))
                    # data['contact_bool'].append(contact_bool)
            else:
                while (time.time() - start < loop_time):
                    yumi.arm.set_jpos(
                        r_joints, 
                        arm='right', 
                        wait=False)
                    time.sleep(sleep_time)

                    # pts = yumi.arm.p.getContactPoints(
                    #     bodyA=yumi.arm.robot_id, bodyB=box_id, linkIndexA=12)
                    # contact_bool = 0 if len(pts) == 0 else 1

                    # object_pose_world = list(yumi.arm.p.getBasePositionAndOrientation(box_id, 0)[
                    #     0]) + list(yumi.arm.p.getBasePositionAndOrientation(box_id, 0)[1])
                    # object_pose_world = util.list2pose_stamped(
                    #     object_pose_world)

                    # palm_frame_world = list(yumi.arm.p.getLinkState(yumi.arm.robot_id, 13)[
                    #                         0]) + list(yumi.arm.p.getLinkState(yumi.arm.robot_id, 13)[1])
                    # palm_frame_world = util.list2pose_stamped(palm_frame_world)

                    # object_pose_palm = util.convert_reference_frame(
                    #     object_pose_world, palm_frame_world, util.unit_pose())

                    # data['palm_pose_world'].append(
                    #     util.pose_stamped2list(palm_frame_world))
                    # data['object_pose_palm'].append(
                    #     util.pose_stamped2list(object_pose_palm))
                    # data['contact_bool'].append(contact_bool)

            # print(i, contact_bool)
            # wait = raw_input('press enter to continue\n')

            # r_success = yumi.set_jpos(r_joints, arm='right', wait=True)
            # l_success = yumi.set_jpos(l_joints, arm='left', wait=True)

            # r_success = yumi.set_ee_pose(wrist_right[0:3], wrist_right[3:], arm='right')
            # l_success = yumi.set_ee_pose(wrist_left[0:3], wrist_left[3:], arm='left')
            # from IPython import embed
            # embed()
            # if i > 20:
                # from IPython import embed
                # embed()

    # with open('./data/' + args.primitive+'_object_poses_tip_2.pkl', 'wb') as f:
        # pickle.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_package_path',
                        type=str,
                        default='/home/anthony/repos/research/mpalm_affordances/catkin_ws/src/config/')
    parser.add_argument('--example_config_path', type=str, default='config')
    parser.add_argument('--primitive', type=str, default='push', help='which primitive to plan')
    parser.add_argument('--simulate', type=int, default=1)
    parser.add_argument('--object', type=int, default=0)
    args = parser.parse_args()
    main(args)
