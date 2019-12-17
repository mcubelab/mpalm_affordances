from planning import pushing_planning, grasp_planning, levering_planning, pulling_planning
from helper import util, planning_helper, collisions

import os
# from example_config import get_cfg_defaults
from closed_loop_experiments import get_cfg_defaults

from airobot import Robot
from airobot.utils import pb_util, common, arm_util
import pybullet as p
import time
import argparse
import numpy as np

import pickle
import rospy
from IPython import embed

import trimesh
import copy

from macro_actions import ClosedLoopMacroActions, YumiGelslimPybulet


class EvalPrimitives():
    def __init__(self, cfg, object_id, mesh_file):
        self.cfg = cfg
        self.object_id = object_id

        self.init_poses = [
            self.cfg.OBJECT_POSE_1,
            self.cfg.OBJECT_POSE_2,
            self.cfg.OBJECT_POSE_3
        ]

        self.init_oris = []
        for i, pose in enumerate(self.init_poses):
            self.init_oris.append(pose[3:])

        self.pb_client = pb_util.PB_CLIENT

        self.x_bounds = [0.2, 0.55]
        self.y_bounds = [-0.3, -0.01]
        self.default_z = 0.1

        self.mesh_file = mesh_file
        self.mesh = trimesh.load(self.mesh_file)
        self.mesh_world = copy.deepcopy(self.mesh)

    def transform_mesh_world(self):
        self.mesh_world = copy.deepcopy(self.mesh)
        obj_pos_world = list(p.getBasePositionAndOrientation(self.object_id, self.pb_client)[0])
        obj_ori_world = list(p.getBasePositionAndOrientation(self.object_id, self.pb_client)[1])
        obj_ori_mat = common.quat2rot(obj_ori_world)
        h_trans = np.zeros((4, 4))
        h_trans[:3, :3] = obj_ori_mat
        h_trans[:-1, -1] = obj_pos_world
        h_trans[-1, -1] = 1
        self.mesh_world.apply_transform(h_trans)

    def get_obj_pose(self):
        obj_pos_world = list(p.getBasePositionAndOrientation(self.object_id, self.pb_client)[0])
        obj_ori_world = list(p.getBasePositionAndOrientation(self.object_id, self.pb_client)[1])

        obj_pose_world = util.list2pose_stamped(obj_pos_world + obj_ori_world)
        return obj_pose_world, obj_pos_world + obj_ori_world

    def get_rand_init(self, execute=True, ind=None):
        rand_yaw = (np.pi/4)*np.random.random_sample() - np.pi/8
        dq = common.euler2quat([0, 0, rand_yaw]).tolist()
        if ind is None:
            init_ind = np.random.randint(len(self.init_oris))
        else:
            init_ind = ind
        q = common.quat_multiply(
            dq,
            self.init_oris[init_ind])
        x = self.x_bounds[0] + (self.x_bounds[1] - self.x_bounds[0]) * np.random.random_sample()
        y = self.y_bounds[0] + (self.y_bounds[1] - self.y_bounds[0]) * np.random.random_sample()
        if execute:
            p.resetBasePositionAndOrientation(
                self.object_id,
                [x, y, self.default_z],
                q,
                self.pb_client)

        time.sleep(1.0)
        self.transform_mesh_world()
        return x, y, q, init_ind

    def get_init(self, ind, execute=True):
        if execute:
            p.resetBasePositionAndOrientation(
                self.object_id,
                self.init_poses[ind][:3],
                self.init_poses[ind][3:],
                self.pb_client
            )
        return self.init_poses[ind]

    def sample_contact(self, primitive_name='pull', N=1):
        valid = False
        timeout = 10
        start = time.time()
        while not valid:
            sampled_contact, sampled_facet = self.mesh_world.sample(N, True)
            sampled_normal = self.mesh_world.face_normals[sampled_facet[0]]
            if primitive_name == 'push':
                in_xy = np.abs(np.dot(sampled_normal, [0, 0, 1])) < 0.0001

                if in_xy:
                    valid = True

            elif primitive_name == 'pull':
                parallel_z = np.abs(np.dot(sampled_normal, [1, 0, 0])) < 0.0001 and \
                    np.abs(np.dot(sampled_normal, [0, 1, 0])) < 0.0001

                above_com = sampled_contact[0][-1] > self.mesh_world.center_mass[-1]

                if parallel_z and above_com:
                    valid = True
            else:
                raise ValueError('Primitive name not recognized')

            if time.time() - start > timeout:
                print("Contact point sample timed out! Exiting")
                return None, None, None

        return sampled_contact, sampled_normal, sampled_facet

    def get_palm_pose_body_frame(self, point, normal, primitive_name='pull'):
        if primitive_name == 'pull':
            # rand_pull_yaw = (np.pi/2)*np.random.random_sample() + np.pi/4
            rand_pull_yaw = 3*np.pi/4
            tip_ori = common.euler2quat([np.pi/2, 0, rand_pull_yaw])
            ori_list = tip_ori.tolist()
        elif primitive_name == 'push':
            y_vec = normal
            z_vec = np.array([0, 0, -1])
            x_vec = np.cross(y_vec, z_vec)

            tip_ori = util.pose_from_vectors(x_vec, y_vec, z_vec, point[0])
            ori_list = util.pose_stamped2list(tip_ori)[3:]

        point_list = point[0].tolist()

        world_pose_list = point_list + ori_list
        world_pose = util.list2pose_stamped(world_pose_list)
        return world_pose


def main(args):
    cfg_file = os.path.join(args.example_config_path, args.primitive) + ".yaml"
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    cfg.freeze()

    rospy.init_node('MacroActions')

    # setup yumi
    yumi_ar = Robot('yumi',
                    pb=True,
                    arm_cfg={'render': True, 'self_collision': False})
    yumi_ar.arm.set_jpos(cfg.RIGHT_INIT + cfg.LEFT_INIT)

    # setup yumi_gs
    yumi_gs = YumiGelslimPybulet(yumi_ar, cfg, exec_thread=args.execute_thread)

    # embed()

    if args.object:
        # box_id = pb_util.load_urdf(
        #     args.config_package_path +
        #     'descriptions/urdf/'+args.object_name+'.urdf',
        #     cfg.OBJECT_INIT[0:3],
        #     cfg.OBJECT_INIT[3:]
        # )
        box_id = pb_util.load_urdf(
            args.config_package_path +
            'descriptions/urdf/'+args.object_name+'_trans.urdf',
            cfg.OBJECT_POSE_3[0:3],
            cfg.OBJECT_POSE_3[3:]
        )

    # setup macro_planner
    action_planner = ClosedLoopMacroActions(
        cfg,
        yumi_gs,
        box_id,
        pb_util.PB_CLIENT,
        args.config_package_path,
        replan=args.replan
    )

    manipulated_object = None
    object_pose1_world = util.list2pose_stamped(cfg.OBJECT_INIT)
    object_pose2_world = util.list2pose_stamped(cfg.OBJECT_FINAL)
    palm_pose_l_object = util.list2pose_stamped(cfg.PALM_LEFT)
    palm_pose_r_object = util.list2pose_stamped(cfg.PALM_RIGHT)

    example_args = {}
    example_args['object_pose1_world'] = object_pose1_world
    example_args['object_pose2_world'] = object_pose2_world
    example_args['palm_pose_l_object'] = palm_pose_l_object
    example_args['palm_pose_r_object'] = palm_pose_r_object
    example_args['object'] = manipulated_object
    example_args['N'] = 60  # 60
    example_args['init'] = True

    primitive_name = args.primitive

    mesh_file = args.config_package_path + 'descriptions/meshes/objects/' + args.object_name + '_experiments.stl'
    exp = EvalPrimitives(cfg, box_id, mesh_file)

    if args.debug:
        init_id = exp.get_rand_init()[-1]
        obj_pose_final = util.list2pose_stamped(exp.init_poses[init_id])
        point, normal, face = exp.sample_contact(primitive_name)

        world_pose = exp.get_palm_pose_body_frame(
            point,
            normal,
            primitive_name=primitive_name)

        obj_pos_world = list(p.getBasePositionAndOrientation(box_id, pb_util.PB_CLIENT)[0])
        obj_ori_world = list(p.getBasePositionAndOrientation(box_id, pb_util.PB_CLIENT)[1])

        obj_pose_world = util.list2pose_stamped(obj_pos_world + obj_ori_world)
        contact_obj_frame = util.convert_reference_frame(world_pose, obj_pose_world, util.unit_pose())

        example_args['palm_pose_r_object'] = contact_obj_frame
        example_args['object_pose1_world'] = obj_pose_world

        obj_pose_final = util.list2pose_stamped(exp.init_poses[init_id])
        obj_pose_final.pose.position.z = obj_pose_world.pose.position.z/1.175
        print("init: ")
        print(util.pose_stamped2list(object_pose1_world))
        print("final: ")
        print(util.pose_stamped2list(obj_pose_final))
        example_args['object_pose2_world'] = obj_pose_final

        plan = action_planner.get_primitive_plan(primitive_name, example_args, 'right')

        embed()

        import simulation

        for i in range(10):
            simulation.visualize_object(
                object_pose1_world,
                filepath="package://config/descriptions/meshes/objects/realsense_box_experiments.stl",
                name="/object_initial",
                color=(1., 0., 0., 1.),
                frame_id="/yumi_body",
                scale=(1., 1., 1.))
            simulation.visualize_object(
                object_pose2_world,
                filepath="package://config/descriptions/meshes/objects/realsense_box_experiments.stl",
                name="/object_final",
                color=(0., 0., 1., 1.),
                frame_id="/yumi_body",
                scale=(1., 1., 1.))
            rospy.sleep(.1)
        simulation.simulate(plan)    
    else:
        for _ in range(20):
            k = 0
            while True:
                k += 1
                # init_id = exp.get_rand_init()[-1]
                init_id = exp.get_rand_init(ind=0)[-1]                

                point, normal, face = exp.sample_contact(primitive_name=primitive_name)
                if point is not None:
                    break
                if k >= 10:
                    print("FAILED")
                    return

            world_pose = exp.get_palm_pose_body_frame(
                point,
                normal,
                primitive_name=primitive_name)

            obj_pos_world = list(p.getBasePositionAndOrientation(box_id, pb_util.PB_CLIENT)[0])
            obj_ori_world = list(p.getBasePositionAndOrientation(box_id, pb_util.PB_CLIENT)[1])

            obj_pose_world = util.list2pose_stamped(obj_pos_world + obj_ori_world)
            contact_obj_frame = util.convert_reference_frame(world_pose, obj_pose_world, util.unit_pose())

            example_args['palm_pose_r_object'] = contact_obj_frame
            example_args['object_pose1_world'] = obj_pose_world

            obj_pose_final = util.list2pose_stamped(exp.init_poses[init_id])
            obj_pose_final.pose.position.z = obj_pose_world.pose.position.z/1.175
            print("init: ")
            print(util.pose_stamped2list(object_pose1_world))
            print("final: ")
            print(util.pose_stamped2list(obj_pose_final))
            example_args['object_pose2_world'] = obj_pose_final
            try:
                result = action_planner.execute(primitive_name, example_args)

                print("reached final: " + str(result[0]))
            except ValueError:
                print("moveit failed!")

            time.sleep(1.0)
            yumi_gs.update_joints(cfg.RIGHT_INIT + cfg.LEFT_INIT)
            time.sleep(1.0)

    embed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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

    args = parser.parse_args()
    main(args)