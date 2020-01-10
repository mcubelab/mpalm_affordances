import task_planning.sampling as sampling
import task_planning.grasp_sampling as grasp_sampling
from task_planning.objects import Object, CollisionBody
import tf

from helper import util

import os
# from example_config import get_cfg_defaults
from closed_loop_experiments import get_cfg_defaults

from airobot import Robot
from airobot.utils import pb_util, common, arm_util
import pybullet as p
import time
import argparse
import numpy as np
import threading

import pickle
import rospy
from IPython import embed

import trimesh
import copy

from macro_actions import ClosedLoopMacroActions, YumiGelslimPybulet


class EvalPrimitives(object):
    """
    Base class for evaluating manipulation primitives
    """
    def __init__(self, cfg, object_id, mesh_file):
        self.cfg = cfg
        self.object_id = object_id

        self.pb_client = pb_util.PB_CLIENT

        self.x_bounds = [0.2, 0.55]
        self.y_bounds = [-0.3, -0.01]
        self.yaw_bounds = [-np.pi/8, np.pi/8]
        self.default_z = 0.1

        self.mesh_file = mesh_file
        self.mesh = trimesh.load(self.mesh_file)
        self.mesh_world = copy.deepcopy(self.mesh)

    def transform_mesh_world(self):
        """
        Interal method to transform the object mesh coordinates
        to the world frame, based on where it is in the environment
        """
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
        """
        Method to get the pose of the object in the world

        Returns:
            PoseStamped: World frame object pose
            list: World frame object pose
        """
        obj_pos_world = list(p.getBasePositionAndOrientation(self.object_id, self.pb_client)[0])
        obj_ori_world = list(p.getBasePositionAndOrientation(self.object_id, self.pb_client)[1])

        obj_pose_world = util.list2pose_stamped(obj_pos_world + obj_ori_world)
        return obj_pose_world, obj_pos_world + obj_ori_world

    def get_rand_trans_yaw(self):
        """
        Get a random x, y, and theta (yaw) value, depending on
        internally defined bounds

        Returns:
            float: x value (meters)
            float: y value (meters)
            float: yaw value (radians)
        """
        rand_yaw = (max(self.yaw_bounds) - min(self.yaw_bounds)) * \
            np.random.random_sample() - max(self.yaw_bounds)
        dq = common.euler2quat([0, 0, rand_yaw]).tolist()
        x = self.x_bounds[0] + (self.x_bounds[1] - self.x_bounds[0]) * np.random.random_sample()
        y = self.y_bounds[0] + (self.y_bounds[1] - self.y_bounds[0]) * np.random.random_sample()
        return x, y, dq

    def get_rand_init(self, *args, **kwargs):
        raise NotImplementedError

    def sample_contact(self, *args, **kwargs):
        raise NotImplementedError

    def get_palm_poses_world_frame(self, *args, **kwargs):
        raise NotImplementedError


class SingleArmPrimitives(EvalPrimitives):
    """
    Helper class for evaluating the closed loop performance of
    push and pull manipulation primitives
    """
    def __init__(self, cfg, object_id, mesh_file):
        """
        Constructor, sets up samplers for primitive problem
        instances using the 3D model of the object being manipulated.
        Sets up an internal set of valid stable object orientations,
        specified from config file, and internal mesh of the object
        in the environment

        Args:
            cfg ([type]): [description]
            object_id ([type]): [description]
            mesh_file ([type]): [description]
        """
        super(SingleArmPrimitives, self).__init__(
            cfg=cfg,
            object_id=object_id,
            mesh_file=mesh_file
        )
        # self.cfg = cfg
        # self.object_id = object_id

        self.init_poses = [
            self.cfg.OBJECT_POSE_1,
            self.cfg.OBJECT_POSE_2,
            self.cfg.OBJECT_POSE_3
        ]

        self.init_oris = []
        for i, pose in enumerate(self.init_poses):
            self.init_oris.append(pose[3:])

        # self.pb_client = pb_util.PB_CLIENT

        self.x_bounds = [0.2, 0.55]
        self.y_bounds = [-0.3, -0.01]
        self.default_z = 0.1

        # self.mesh_file = mesh_file
        # self.mesh = trimesh.load(self.mesh_file)
        # self.mesh_world = copy.deepcopy(self.mesh)

    # def transform_mesh_world(self):
    #     """
    #     Interal method to transform the object mesh coordinates
    #     to the world frame, based on where it is in the environment
    #     """
    #     self.mesh_world = copy.deepcopy(self.mesh)
    #     obj_pos_world = list(p.getBasePositionAndOrientation(self.object_id, self.pb_client)[0])
    #     obj_ori_world = list(p.getBasePositionAndOrientation(self.object_id, self.pb_client)[1])
    #     obj_ori_mat = common.quat2rot(obj_ori_world)
    #     h_trans = np.zeros((4, 4))
    #     h_trans[:3, :3] = obj_ori_mat
    #     h_trans[:-1, -1] = obj_pos_world
    #     h_trans[-1, -1] = 1
    #     self.mesh_world.apply_transform(h_trans)

    # def get_obj_pose(self):
    #     """
    #     Method to get the pose of the object in the world

    #     Returns:
    #         [type]: [description]
    #     """
    #     obj_pos_world = list(p.getBasePositionAndOrientation(self.object_id, self.pb_client)[0])
    #     obj_ori_world = list(p.getBasePositionAndOrientation(self.object_id, self.pb_client)[1])

    #     obj_pose_world = util.list2pose_stamped(obj_pos_world + obj_ori_world)
    #     return obj_pose_world, obj_pos_world + obj_ori_world

    def get_rand_init(self, execute=True, ind=None):
        """
        Getter function to get a random initial pose of the object,
        corresponding to some stable orientation and a random yaw and
        translation on the table.

        Args:
            execute (bool, optional): [description]. Defaults to True.
            ind ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        rand_yaw = (np.pi/4)*np.random.random_sample() - np.pi/8
        # dq = common.euler2quat([0, 0, rand_yaw]).tolist()
        # x = self.x_bounds[0] + (self.x_bounds[1] - self.x_bounds[0]) * np.random.random_sample()
        # y = self.y_bounds[0] + (self.y_bounds[1] - self.y_bounds[0]) * np.random.random_sample()
        x, y, dq = self.get_rand_trans_yaw()

        if ind is None:
            init_ind = np.random.randint(len(self.init_oris))
        else:
            init_ind = ind
        q = common.quat_multiply(
            dq,
            self.init_oris[init_ind])

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
        """
        Gets one of the valid stable initial poses of the object,
        as specified in the configuration file

        Args:
            ind (int): Index of the pose
            execute (bool, optional): If true, updates the pose of the
                object in the environment. Defaults to True.

        Returns:
            [type]: [description]
        """
        if execute:
            p.resetBasePositionAndOrientation(
                self.object_id,
                self.init_poses[ind][:3],
                self.init_poses[ind][3:],
                self.pb_client
            )
        return self.init_poses[ind]

    def sample_contact(self, primitive_name='pull', N=1):
        """
        Function to sample a contact point and orientation on the object,
        for a particular type of primitive action.

        Args:
            primitive_name (str, optional): [description]. Defaults to 'pull'.
            N (int, optional): [description]. Defaults to 1.

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """
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

                above_com = (sampled_contact[0][-1] >
                             self.mesh_world.center_mass[-1])

                if parallel_z and above_com:
                    valid = True
            else:
                raise ValueError('Primitive name not recognized')

            if time.time() - start > timeout:
                print("Contact point sample timed out! Exiting")
                return None, None, None

        return sampled_contact, sampled_normal, sampled_facet

    def get_palm_poses_world_frame(self, point, normal, primitive_name='pull'):
        """
        Function to get a valid orientation of the palm in the world,
        specific to a particular primitive action type and contact location.

        Args:
            point ([type]): [description]
            normal ([type]): [description]
            primitive_name (str, optional): [description]. Defaults to 'pull'.

        Returns:
            [type]: [description]
        """
        # default to only using right arm right now
        active_arm = 'right'
        inactive_arm = 'left'
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

        palm_poses_world = {}
        palm_poses_world[active_arm] = world_pose
        palm_poses_world[inactive_arm] = None # should be whatever it currently is, set to current value if returned as None
        return world_pose


class DualArmPrimitives(EvalPrimitives):
    """
    Helper class for evaluating the closed loop performance of
    push and pull manipulation primitives
    """
    def __init__(self, cfg, object_id, mesh_file, goal_face=0):
        super(DualArmPrimitives, self).__init__(
            cfg=cfg,
            object_id=object_id,
            mesh_file=mesh_file
        )

        self.goal_face = goal_face

        self._setup_graph()
        self._build_and_sample_graph()

        self.goal_pose_prop_frame = self.grasp_samples.collision_free_samples['object_pose'][self.goal_face][0]
        self.goal_pose_world_frame = util.convert_reference_frame(
            pose_source=self.goal_pose_prop_frame,
            pose_frame_target=util.unit_pose(),
            pose_frame_source=self.proposals_base_frame
        )

    def _setup_graph(self):
        self.listener = tf.TransformListener()
        self.br = tf.TransformBroadcaster()

        self.gripper_name = 'mpalms_all_coarse.stl'
        self.table_name = 'table_top_collision.stl'
        self._object = Object(
            mesh_name=self.mesh_file
        )
        self.table = CollisionBody(
            mesh_name=os.path.join(os.environ["CODE_BASE"],
            'catkin_ws/src/config/descriptions/meshes/table/' + self.table_name)
        )
        trans, quat = self.listener.lookupTransform('yumi_body', 'table_top', rospy.Time(0))
        table_pose = trans + quat
        self.table.setCollisionPose(
            self.table.collision_object,
            util.list2pose_stamped(table_pose)
        )

        self.gripper_left = CollisionBody(
            mesh_name=os.path.join(os.environ["CODE_BASE"], 
            'catkin_ws/src/config/descriptions/meshes/mpalm/' + self.gripper_name)
        )
        self.gripper_right = CollisionBody(
            mesh_name=os.path.join(os.environ["CODE_BASE"], 
            'catkin_ws/src/config/descriptions/meshes/mpalm/' + self.gripper_name)
        )

        self.proposals_base_frame = util.list2pose_stamped(
            [0.45, 0, 0, 0, 0, 0, 1]
        )

    def _build_and_sample_graph(self):
        self.sampler = sampling.Sampling(
            self.proposals_base_frame,
            self._object,
            self.table,
            self.gripper_left,
            self.gripper_right,
            self.listener,
            self.br
        )

        self.grasp_samples = grasp_sampling.GraspSampling(
            self.sampler,
            num_samples=3,
            is_visualize=True
        )

        self.node_seq_dict = {}
        self.intersection_dict_grasp_dict = {}
        self.placement_seq_dict = {}
        self.sample_seq_dict = {}
        self.primitive_seq_dict =  {}
        for i in range(6):
            self.node_seq_dict[i] = None
            self.intersection_dict_grasp_dict[i] = None
            self.placement_seq_dict[i] = None
            self.sample_seq_dict[i] = None
            self.primitive_seq_dict[i] = None

        # hard coding sampling over 6 faces for now, should be parameterized based on faces on the mesh?
        for i in range(6):
            node_seq, intersection_dict_grasp = sampling.search_placement_graph(
                grasp_samples=self.grasp_samples,
                placement_list=[self.goal_face, i]
            )

            placement_seq, sample_seq, primitive_seq = sampling.search_primitive_graph(
                _node_sequence=node_seq,
                intersection_dict_grasp=intersection_dict_grasp
            )

            self.node_seq_dict[i] = node_seq
            self.intersection_dict_grasp_dict[i] = intersection_dict_grasp
            self.placement_seq_dict[i] = placement_seq
            self.sample_seq_dict[i] = sample_seq
            self.primitive_seq_dict[i] = primitive_seq

        # just get the first object pose on the goal face (should all be the same)
        self.goal_pose = self.grasp_samples.collision_free_samples['object_pose'][self.goal_face][0]

    def get_nominal_init(self, ind, sample=0):
        """
        Get the nominal object initial object pose corresponding
        to a particular stable placement, indicated by argument "ind"

        Args:
            ind (int): Index of the stable placement/face
            sample (int): Which sample id in the grasp samples dict,
                Defaults to 0

        Raises:
            ValueError: If placement graph has not been built yet

        Returns:
            PoseStamped: Initial object pose in the world frame
        """
        # embed()
        # if self.sample_seq_dict[ind] is None:
        #     raise ValueError('Samples do not exist yet')

        init_object_pose = self.grasp_samples.collision_free_samples['object_pose'][ind][sample]
        init_object_pose_world = util.convert_reference_frame(
            pose_source=init_object_pose,
            pose_frame_target=util.unit_pose(),
            pose_frame_source=self.proposals_base_frame
        )
        return init_object_pose_world

    def get_rand_init(self, execute=True, ind=None):
        # sample from the sample_sequence list to get a face and pose
        # that connects to the goal
        if ind is None:
            ind = np.random.randint(6)

        # perturb that initial pose with a translation and a yaw
        rand_yaw = (np.pi/4)*np.random.random_sample() - np.pi/8
        dq = common.euler2quat([0, 0, rand_yaw]).tolist()

        x = self.x_bounds[0] + (self.x_bounds[1] - self.x_bounds[0]) * np.random.random_sample()
        y = self.y_bounds[0] + (self.y_bounds[1] - self.y_bounds[0]) * np.random.random_sample()

        nominal_init_pose = self.get_nominal_init(ind)
        nominal_init_q = np.array(util.pose_stamped2list(nominal_init_pose)[3:])
        q = common.quat_multiply(dq, nominal_init_q)

        if execute:
            p.resetBasePositionAndOrientation(
                self.object_id,
                [x, y, self.default_z],
                q,
                self.pb_client)

        time.sleep(1.0)
        self.transform_mesh_world()
        return x, y, q, ind

    def get_palm_poses_world_frame(self, ind, obj_world, sample=None):
        if len(self.sample_seq_dict[ind]) > 1:
            raise ValueError('Only sampling one step reachable goals right now')
        if sample is None:
            # TODO handle cases where its two steps away
            sample_ind = np.random.randint(
                len(self.sample_seq_dict[ind][0])
            )
            sample_id = self.sample_seq_dict[ind][0][sample_ind]
        
        sample_index = self.grasp_samples.collision_free_samples['sample_ids'][ind].index(sample_id)
        right_prop_frame = self.grasp_samples.collision_free_samples['gripper_poses'][ind][sample_index][1]
        left_prop_frame = self.grasp_samples.collision_free_samples['gripper_poses'][ind][sample_index][0]

        right_nom_world_frame = util.convert_reference_frame(
            pose_source=right_prop_frame,
            pose_frame_target=util.unit_pose(),
            pose_frame_source=self.proposals_base_frame
        )

        left_nom_world_frame = util.convert_reference_frame(
            pose_source=left_prop_frame,
            pose_frame_target=util.unit_pose(),
            pose_frame_source=self.proposals_base_frame
        )

        right_obj_transform = util.convert_reference_frame(
            pose_source=obj_world,
            pose_frame_target=right_nom_world_frame,
            pose_frame_source=util.unit_pose()
        )
        # right_world_frame = util.transform_pose(
        #     pose_source=right_nom_world_frame,
        #     pose_transform=right_obj_transform)
        right_world_frame = util.transform_pose(
            pose_source=right_obj_transform,
            pose_transform=right_nom_world_frame)        

        left_obj_transform = util.convert_reference_frame(
            pose_source=obj_world,
            pose_frame_target=left_nom_world_frame,
            pose_frame_source=util.unit_pose()
        )
        # left_world_frame = util.transform_pose(
        #     pose_source=left_nom_world_frame,
        #     pose_transform=left_obj_transform)  
        left_world_frame = util.transform_pose(
            pose_source=left_obj_transform,
            pose_transform=left_nom_world_frame)              

        palm_poses_world = {}
        palm_poses_world['right'] = right_world_frame
        palm_poses_world['left'] = left_world_frame
        # palm_poses_world['right'] = right_nom_world_frame
        # palm_poses_world['left'] = left_nom_world_frame  
        return palm_poses_world


class GoalVisual():
    def __init__(self, trans_box_lock, object_id, pb_client, goal_init):
        self.trans_box_lock = trans_box_lock
        self.pb_client = pb_client
        self.object_id = object_id

        self.update_goal_state(goal_init)

    def visualize_goal_state(self):
        """
        [summary]

        Args:
            object_id ([type]): [description]
            goal_pose ([type]): [description]
            pb_client ([type]): [description]
        """
        while True:
            self.trans_box_lock.acquire()
            p.resetBasePositionAndOrientation(
                self.object_id,
                [self.goal_pose[0], self.goal_pose[1], self.goal_pose[2]],
                self.goal_pose[3:],
                physicsClientId=self.pb_client)
            self.trans_box_lock.release()
            time.sleep(0.01)

    def update_goal_state(self, goal):
        self.trans_box_lock.acquire()
        self.goal_pose = goal
        self.trans_box_lock.release()


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

    gel_id = 12

    alpha = 0.01
    K = 500

    p.changeDynamics(
        yumi_ar.arm.robot_id,
        gel_id,
        restitution=0.99,
        contactStiffness=K,
        contactDamping=alpha*K,
        rollingFriction=args.rolling
    )

    # setup yumi_gs
    yumi_gs = YumiGelslimPybulet(yumi_ar, cfg, exec_thread=args.execute_thread)

    if args.object:
        box_id = pb_util.load_urdf(
            args.config_package_path +
            'descriptions/urdf/'+args.object_name+'.urdf',
            cfg.OBJECT_POSE_3[0:3],
            cfg.OBJECT_POSE_3[3:]
        )
        trans_box_id = pb_util.load_urdf(
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
    example_args['table_face'] = 0

    primitive_name = args.primitive

    mesh_file = args.config_package_path + 'descriptions/meshes/objects/' + args.object_name + '_experiments.stl'
    exp_single = SingleArmPrimitives(cfg, box_id, mesh_file)
    exp_double = DualArmPrimitives(cfg, box_id, mesh_file)

    trans_box_lock = threading.RLock()
    goal_viz = GoalVisual(
        trans_box_lock,
        trans_box_id,
        action_planner.pb_client,
        cfg.OBJECT_POSE_3)

    visualize_goal_thread = threading.Thread(
        target=goal_viz.visualize_goal_state)
    visualize_goal_thread.daemon = True
    visualize_goal_thread.start()

    if args.debug:
        # embed()
        # init_id = exp.get_rand_init(ind=2)[-1]
        # obj_pose_final = util.list2pose_stamped(exp.init_poses[init_id])
        # point, normal, face = exp.sample_contact(primitive_name)

        # # embed()

        # world_pose = exp.get_palm_pose_world_frame(
        #     point,
        #     normal,
        #     primitive_name=primitive_name)

        # obj_pos_world = list(p.getBasePositionAndOrientation(box_id, pb_util.PB_CLIENT)[0])
        # obj_ori_world = list(p.getBasePositionAndOrientation(box_id, pb_util.PB_CLIENT)[1])

        # obj_pose_world = util.list2pose_stamped(obj_pos_world + obj_ori_world)
        # contact_obj_frame = util.convert_reference_frame(world_pose, obj_pose_world, util.unit_pose())

        # example_args['palm_pose_r_object'] = contact_obj_frame
        # example_args['object_pose1_world'] = obj_pose_world

        # obj_pose_final = util.list2pose_stamped(exp.init_poses[init_id])

        init_id = exp_double.get_rand_init(ind=1)[-1]
        obj_pose_world = exp_double.get_obj_pose()[0]
        palm_poses_world = exp_double.get_palm_poses_world_frame(init_id, obj_pose_world)
        obj_pose_final = exp_double.goal_pose_world_frame
        palm_poses_obj_frame = {}
        for key in palm_poses_world.keys():
            palm_poses_obj_frame[key] = util.convert_reference_frame(palm_poses_world[key], obj_pose_world, util.unit_pose())        

        example_args['palm_pose_r_object'] = palm_poses_obj_frame['right']
        example_args['palm_pose_l_object'] = palm_poses_obj_frame['left']        
        example_args['object_pose1_world'] = obj_pose_world

        # obj_pose_final.pose.position.z = obj_pose_world.pose.position.z/1.175
        print("init: ")
        print(util.pose_stamped2list(object_pose1_world))
        print("final: ")
        print(util.pose_stamped2list(obj_pose_final))
        example_args['object_pose2_world'] = obj_pose_final
        example_args['table_face'] = init_id

        plan = action_planner.get_primitive_plan(primitive_name, example_args, 'right')

        embed()

        import simulation

        for i in range(10):
            simulation.visualize_object(
                example_args['object_pose1_world'],
                filepath="package://config/descriptions/meshes/objects/realsense_box_experiments.stl",
                name="/object_initial",
                color=(1., 0., 0., 1.),
                frame_id="/yumi_body",
                scale=(1., 1., 1.))
            simulation.visualize_object(
                example_args['object_pose2_world'],
                filepath="package://config/descriptions/meshes/objects/realsense_box_experiments.stl",
                name="/object_final",
                color=(0., 0., 1., 1.),
                frame_id="/yumi_body",
                scale=(1., 1., 1.))
            rospy.sleep(.1)
        simulation.simulate(plan)
    else:
        for trial in range(20):
            k = 0
            while True:
                # sample a random stable pose, and get the corresponding
                # stable orientation index
                k += 1
                # init_id = exp.get_rand_init()[-1]
                init_id = exp.get_rand_init(ind=0)[-1]

                # sample a point on the object that is valid
                # for the primitive action being executed
                point, normal, face = exp.sample_contact(
                    primitive_name=primitive_name)
                if point is not None:
                    break
                if k >= 10:
                    print("FAILED")
                    return
            # get the full 6D pose palm in world, at contact location
            world_pose = exp.get_palm_pose_world_frame(
                point,
                normal,
                primitive_name=primitive_name)

            # get the object pose in the world frame
            obj_pos_world = list(p.getBasePositionAndOrientation(
                box_id,
                pb_util.PB_CLIENT)[0])
            obj_ori_world = list(p.getBasePositionAndOrientation(
                box_id,
                pb_util.PB_CLIENT)[1])

            obj_pose_world = util.list2pose_stamped(
                obj_pos_world + obj_ori_world)

            # transform the palm pose from the world frame to the object frame
            contact_obj_frame = util.convert_reference_frame(
                world_pose, obj_pose_world, util.unit_pose())

            # set up inputs to the primitive planner, based on task
            # including sampled initial object pose and contacts,
            # and final object pose
            example_args['palm_pose_r_object'] = contact_obj_frame
            example_args['object_pose1_world'] = obj_pose_world

            obj_pose_final = util.list2pose_stamped(exp.init_poses[init_id])
            obj_pose_final.pose.position.z /= 1.155
            print("init: ")
            print(util.pose_stamped2list(object_pose1_world))
            print("final: ")
            print(util.pose_stamped2list(obj_pose_final))
            example_args['object_pose2_world'] = obj_pose_final
            example_args['table_face'] = init_id
            if trial == 0:
                goal_viz.update_goal_state(exp.init_poses[init_id])
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

    parser.add_argument(
        '-r', '--rolling',
        type=float, default=0.0,
        help='rolling friction value for pybullet sim'
    )

    args = parser.parse_args()
    main(args)