import os
import time
import argparse
import numpy as np
import threading
import pickle
import rospy
import trimesh
import copy
import tf
import random
from airobot import Robot
# from airobot.utils import pb_util, common
from airobot.utils import common
import pybullet as p
from IPython import embed

# from example_config import get_cfg_defaults
from closed_loop_experiments_cfg import get_cfg_defaults
from macro_actions import ClosedLoopMacroActions  # YumiGelslimPybulet
from yumi_pybullet_ros import YumiGelslimPybullet
import task_planning.sampling as sampling
import task_planning.grasp_sampling as grasp_sampling
import task_planning.lever_sampling as lever_sampling
from task_planning.objects import Object, CollisionBody
from helper import util, planning_helper


class EvalPrimitives(object):
    """
    Base class for evaluating manipulation primitives
    """
    def __init__(self, cfg, pb_client, object_id, mesh_file):
        self.cfg = cfg
        self.pb_client = pb_client

        self.x_bounds = self.cfg.X_BOUNDS
        self.y_bounds = self.cfg.Y_BOUNDS
        self.default_xy_position = self.cfg.DEFAULT_XY_POS
        self.yaw_bounds = []
        for val in self.cfg.YAW_BOUNDS:
            if isinstance(val, str):
                self.yaw_bounds.append(eval(val))
            else:
                self.yaw_bounds.append(val)
        self.default_z = self.cfg.DEFAULT_Z
        self.mesh_file = None
        self.mesh = None
        self.object_id = None
        self.goal_face = None

    def initialize_object_stable_poses(self, object_id, mesh_file):
        """
        Set up the internal variables that keep track of where the mesh
        is in the world so that contacts and random poses can be computed

        Args:
            object_id (int): PyBullet unique object id of the object
            mesh_file (str): Path to the .stl file with the object geometry
        """
        self.mesh_file = mesh_file
        self.mesh = trimesh.load(self.mesh_file)
        self.mesh_world = copy.deepcopy(self.mesh)

        self.stable_poses_mat = self.mesh_world.compute_stable_poses()[0]
        self.stable_poses_list = []
        for i, mat in enumerate(self.stable_poses_mat):
            pose = util.pose_stamped2list(util.pose_from_matrix(mat))
            pose[0] = self.cfg.OBJECT_WORLD_XY[0]
            pose[1] = self.cfg.OBJECT_WORLD_XY[1]

            self.stable_poses_list.append(pose)

        self.object_id = object_id

    def transform_mesh_world(self, new_pose=None):
        """
        Interal method to transform the object mesh coordinates
        to the world frame, based on where it is in the environment

        Args:
            new_pose (list): World frame pose to update the mesh to reflect,
                [x, y, z, qx, qy, qz, qw]. If this is not provided,
                function will default to using the current world pose in
                pybullet to update the mesh
        """
        if new_pose is None:
            obj_pos_world = list(p.getBasePositionAndOrientation(
                self.object_id, self.pb_client)[0])
            obj_ori_world = list(p.getBasePositionAndOrientation(
                self.object_id, self.pb_client)[1])
        else:
            obj_pos_world = new_pose[:3]
            obj_ori_world = new_pose[3:]
        self.mesh_world = copy.deepcopy(self.mesh)

        obj_ori_mat = common.quat2rot(obj_ori_world)
        h_trans = np.eye(4)
        h_trans[:3, :3] = obj_ori_mat
        h_trans[:-1, -1] = obj_pos_world
        h_trans[-1, -1] = 1

        self.mesh_world.apply_transform(h_trans)

    def get_obj_pose(self):
        """
        Method to get the pose of the object in the world, from pybullet

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
            np.random.random_sample() + min(self.yaw_bounds)
        dq = common.euler2quat([0, 0, rand_yaw]).tolist()
        x = self.x_bounds[0] + (self.x_bounds[1] - self.x_bounds[0]) * np.random.random_sample()
        y = self.y_bounds[0] + (self.y_bounds[1] - self.y_bounds[0]) * np.random.random_sample()
        return x, y, dq

    def get_rand_trans_yaw_T(self, pos=None):
        """Get a random transform that translates the object in [x, y],
        and applied an in place yaw about the positive z-axis. Assume that
        the yaw angle happens first, followed by the translation

        Args:
            pos (np.ndarray, optional): Current position of the object (needed to compute the in-place yaw).
                If None, will get the current object pose for this angle

        Returns:
            np.ndarray: 4 X 4 homogeneous transform that can be applied to the specified position
        """
        if pos is None:
            pos = np.asarray(self.get_obj_pose()[1][:3])
        # get in place yaw transform
        rand_yaw_T = util.rand_body_yaw_transform(pos, min_theta=min(self.yaw_bounds), max_theta=max(self.yaw_bounds))

        # get final translation
        # sample positions in the world frame
        x = self.x_bounds[0] + (self.x_bounds[1] - self.x_bounds[0]) * np.random.random_sample()
        y = self.y_bounds[0] + (self.y_bounds[1] - self.y_bounds[0]) * np.random.random_sample()

        # use current position to obtain dx and dy
        dx, dy = x - pos[0], y - pos[1]
        rand_trans_T = np.eye(4)
        rand_trans_T[:2, -1] = [dx, dy]

        # compose transforms
        rand_T = np.matmul(rand_trans_T, rand_yaw_T)
        return rand_T

    def get_rand_init(self, *args, **kwargs):
        raise NotImplementedError

    def sample_contact(self, *args, **kwargs):
        raise NotImplementedError

    def get_palm_poses_world_frame(self, *args, **kwargs):
        raise NotImplementedError

    def calc_n(self, start, goal, scale=125):
        """
        Calculate the number of waypoints to include in the
        primitive plan based on distance to the goal

        Args:
            start (PoseStamped): Start pose of object
            goal (PoseStamped): Goal pose of object
            scale (int, optional): Integer value to scale
                depending on distance to goal

        Returns:
            int: Number of waypoints to be included in primitive plan
        """
        dist = np.sqrt(
            (start.pose.position.x - goal.pose.position.x)**2 +
            (start.pose.position.y - goal.pose.position.y)**2
        )
        N = max(2, int(dist*scale))
        return N

    def get_palm_y_normals(self, palm_poses, current=False):
        """
        Gets the updated world frame normal direction of the palms
        """
        normal_y = util.list2pose_stamped([0, 1, 0, 0, 0, 0, 1])
        normal_y_poses_world = {}

        normal_y_poses_world['right'] = util.transform_pose(normal_y, palm_poses['right'])
        normal_y_poses_world['left'] = util.transform_pose(normal_y, palm_poses['left'])

        return normal_y_poses_world        


class SingleArmPrimitives(EvalPrimitives):
    """
    Helper class for evaluating the closed loop performance of
    push and pull manipulation primitives
    """
    def __init__(self, cfg, pb_client, object_id, mesh_file):
        """
        Constructor, sets up samplers for primitive problem
        instances using the 3D model of the object being manipulated.
        Sets up an internal set of valid stable object orientations,
        specified from config file, and internal mesh of the object
        in the environment

        Args:
            cfg (YACS CfgNode): Configuration parameters
            object_id (int): PyBullet object id of object
                being manipulated
            mesh_file (str): Absolute path of the .stl file
                with the manipulated object
        """
        super(SingleArmPrimitives, self).__init__(
            cfg=cfg,
            pb_client=pb_client,
            object_id=object_id,
            mesh_file=mesh_file
        )

        self.initialize_object(object_id, mesh_file)

    def initialize_object(self, object_id, mesh_file, *args):
        """
        Set up the internal variables that keep track of where the mesh
        is in the world so that contacts and random poses can be computed

        Args:
            object_id (int): PyBullet unique object id of the object
            mesh_file (str): Path to the .stl file with the object geometry
        """
        self.initialize_object_stable_poses(object_id, mesh_file)
        self.init_poses = self.stable_poses_list
        self.init_oris = []
        for i, pose in enumerate(self.init_poses):
            self.init_oris.append(pose[3:])

    def get_valid_goal_faces(self):
        """Function to get a list with values indicating valid goal faces
        """
        # valid_faces = []
        # for i in range(len(self.init_poses)):
        #     valid_faces.append(i)
        # return valid_faces
        return range(len(self.init_poses))

    def get_rand_init(self, execute=True, ind=None):
        """
        Getter function to get a random initial pose of the object,
        corresponding to some stable orientation and a random yaw and
        translation on the table.

        Args:
            execute (bool, optional): Whether or not to actually
                updated the pose of the object in the world.
                True if random initial pose should be applied to
                object in the environment, else False. Defaults to True.
            ind (int, optional): Desired face index of the face touching
                the table in the initial pose. If none, a random one will
                be sampled. Defaults to None.

        Returns:
            5-element tuple containing:
            - float: x position
            - float: y position
            - float: quaternion, orientation
            - int: ID of the stable placement
            - PoseStamped: world frame pose that was sampled
        """
        if ind is None:
            init_ind = np.random.randint(len(self.init_oris))
        else:
            init_ind = ind

        nominal_pose_list = self.init_poses[init_ind]

        pos = nominal_pose_list[:3]
        rand_T = self.get_rand_trans_yaw_T(pos)

        rand_T_pose = util.pose_from_matrix(rand_T)
        nominal_pose = util.list2pose_stamped(nominal_pose_list)
        rand_pose = util.transform_pose(nominal_pose, rand_T_pose)
        rand_pose_list = util.pose_stamped2list(rand_pose)

        x, y, q = rand_pose_list[0], rand_pose_list[1], rand_pose_list[3:]

        if execute:
            default_z = nominal_pose_list[2]
            p.resetBasePositionAndOrientation(
                self.object_id,
                [x, y, default_z],
                q,
                self.pb_client)
            time.sleep(1.0)
            world_pose = self.get_obj_pose()[0]
            time.sleep(1.0)
            self.transform_mesh_world()
        else:
            world_pose = rand_pose
            self.transform_mesh_world(new_pose=rand_pose_list)

        return x, y, q, init_ind, world_pose

    def get_init(self, ind, execute=True):
        """
        Gets one of the valid stable initial poses of the object,
        as specified in the configuration file

        Args:
            ind (int): Index of the pose
            execute (bool, optional): If true, updates the pose of the
                object in the environment. Defaults to True.

        Returns:
            list: Initial pose in form [x, y, z, x, y, z, w]
                from the default initial poses (unperturbed)
        """
        if execute:
            p.resetBasePositionAndOrientation(
                self.object_id,
                self.init_poses[ind][:3],
                self.init_poses[ind][3:],
                self.pb_client
            )
        return self.init_poses[ind]

    def sample_contact(self, primitive_name='pull', new_pose=None, N=300):
        """
        Function to sample a contact point and orientation on the object,
        for a particular type of primitive action.

        For pulling, mesh is
        used to sample contacts on each face, and the normals of that face
        are checked to be orthogonal to the global x-y plane. The z-value of
        the point is also checked to ensure it's above the center of mass.

        For pushing, mesh is used to sample contacts corresponding to faces
        where normal is parallel to the global x-y plane.

        Args:
            primitive_name (str, optional): Which primitve is being used.
                Defaults to 'pull'.
            new_pose (list, optional): Specified pose to use as start state
                to sample mesh from
            N (int, optional): Number of contacts to be returned.
                Defaults to 1.

        Returns:
            3-element tupe containing:
            - list: The world frame position of the contact point
            - list: The surface normal of the object at that point
            - int: The index of the face on the mesh
        """
        valid = False
        timeout = 10
        start = time.time()
        # update mesh to reflect simulator state
        self.transform_mesh_world(new_pose=new_pose)
        time.sleep(1.0)

        # transform original normals (shouldn't need to do this, but was being weird to depend on the transformed mesh normals)
        obj_pose = self.get_obj_pose()[0] if new_pose is None else util.list2pose_stamped(new_pose)
        obj_pos = util.pose_stamped2np(obj_pose)[:3]
        face_normals = util.transform_vectors(self.mesh.face_normals, obj_pose)
        face_normals = face_normals - obj_pos

        # different rules for valid faces based on normals
        valid_faces = []
        if primitive_name == 'pull':
            for i, normal in enumerate(face_normals):
                parallel_z = np.abs(np.dot(normal, [1, 0, 0])) < 0.3 and \
                    np.abs(np.dot(normal, [0, 1, 0])) < 0.3
                if parallel_z:
                    valid_faces.append(i)
        elif primitive_name == 'push':
            for i, normal in enumerate(face_normals):
                in_xy = np.abs(np.dot(normal, [0, 0, 1])) < 0.1
                if in_xy:
                    valid_faces.append(i)
        else:
            raise NotImplementedError('Please select a different primitive')

        while not valid:
            sampled_contacts, sampled_faces = self.mesh_world.sample(N, True)

            # only check sampled contacts that are in valid_faces
            valid_contacts = []
            for i, face in enumerate(sampled_faces):
                if face in valid_faces:
                    valid_contacts.append((sampled_contacts[i], face))

            for i, valid_samples in enumerate(valid_contacts):
                sampled_contact = valid_samples[0]
                sampled_face = valid_samples[1]
                sampled_normal = face_normals[sampled_face]
                if primitive_name == 'pull':
                    if sampled_contact[-1] > self.mesh_world.center_mass[-1]:
                        # sampled_contact[2] -= 0.001
                        sampled_contact[2] -= 0.003 + np.random.random_sample()*1.0e-3
                        valid = True
                        break
                elif primitive_name == 'push':
                    valid = True
                    break
                else:
                    raise ValueError('Primitive name not recognized')

            if valid:
                break

            if time.time() - start > timeout:
                print("Contact point sample timed out! Exiting")
                return None, None, None

        return sampled_contact, sampled_normal, sampled_face

    def get_palm_poses_world_frame(self, point, normal, primitive_name='pull'):
        """
        Function to get a valid orientation of the palm in the world,
        specific to a particular primitive action type and contact location.

        Positive y-axis in the palm frame is set to be aligned with the surface
        normal at the contact point, and a random yaw is set between some
        defined range

        Args:
            point (list): [x, y, z] world frame position of palms at contact
            normal (list): [x, y, z] surface normal of the object at the
                contact point
            primitive_name (str, optional): Which primitive. Defaults to 'pull'.

        Returns:
            PoseStamped: World frame palm pose at the contact point
        """
        # default to only using right arm right now
        active_arm = 'right'
        inactive_arm = 'left'
        if primitive_name == 'pull':
            # rand_pull_yaw = (3*np.pi/4)*np.random.random_sample() + np.pi/4
            # # rand_pull_yaw = 3*np.pi/4
            # tip_ori = common.euler2quat([np.pi/2, 0, rand_pull_yaw])
            # ori_list = tip_ori.tolist()

            y_vec = normal
            z_vec = util.sample_orthogonal_vector(y_vec)
            x_vec = np.cross(y_vec, z_vec)
            tip_ori = util.pose_from_vectors(x_vec, y_vec, z_vec, point)
            ori_list = util.pose_stamped2list(tip_ori)[3:]

        elif primitive_name == 'push':
            y_vec = normal
            z_vec = np.array([0, 0, -1])
            x_vec = np.cross(y_vec, z_vec)

            tip_ori = util.pose_from_vectors(x_vec, y_vec, z_vec, point)
            ori_list = util.pose_stamped2list(tip_ori)[3:]

        point_list = point.tolist()

        world_pose_list = point_list + ori_list
        world_pose = util.list2pose_stamped(world_pose_list)

        palm_poses_world = {}
        palm_poses_world[active_arm] = world_pose
        palm_poses_world[inactive_arm] = None # should be whatever it currently is, set to current value if returned as None
        # palm_poses_world[inactive_arm] = world_pose
        return world_pose

    def get_random_primitive_args(self, ind=None, primitive='pull',
                                  random_goal=False, execute=True,
                                  start_pose=None, penetration_delta=5e-3):
        """
        Function to abstract away all the setup for sampling a primitive
        instance

        Args:
            ind (ind): Index representing which stable orientation should be used. If
                None, a random orientation will be sampled
            primitive (str): Which primitive to sample arguments for
            random_goal (bool): If True, sample a random goal. If False, use the default
                stable orientation as the goal.
            execute (bool): If True, reset the object in the PyBullet
                simulation to the sampled start state
            start_pose (PoseStamped): If included, random start pose will not
                be sampled, and instead only a random grasp and goal pose
                (if random_goal==True) will be sampled, and the specified
                start pose will be used.
            penetration_delta (float): Minimum penetration noise to add to the
                normal direction of the palm pose, to increase likelihood of
                sticking contact

        Returns:
            dict: Inputs to primitive planner that can be directly executed
        """
        primitive_args = {}
        primitive_args['name'] = primitive
        primitive_args['object'] = None
        primitive_args['N'] = 50
        primitive_args['init'] = True
        primitive_args['table_face'] = 0
        primitive_args['palm_pose_l_object'] = util.list2pose_stamped(self.cfg.PALM_LEFT)
        primitive_args['palm_pose_l_world'] = util.list2pose_stamped(self.cfg.PALM_LEFT)

        k = 0
        while True:
            have_contact = False
            k += 1

            if start_pose is None or ind is None:
                x, y, q, init_id, obj_pose_initial = self.get_rand_init(
                    execute=execute, ind=ind)
            else:
                if execute:
                    p.resetBasePositionAndOrientation(
                        self.object_id,
                        util.pose_stamped2list(start_pose)[:3],
                        util.pose_stamped2list(start_pose)[3:]
                    )
                    time.sleep(0.5)

                    real_start = p.getBasePositionAndOrientation(self.object_id)
                    real_start_pos, real_start_ori = real_start[0], real_start[1]
                    start_pose = util.list2pose_stamped(
                        list(real_start_pos) + list(real_start_ori))
                obj_pose_initial = start_pose
                init_id = ind
                self.transform_mesh_world(new_pose=util.pose_stamped2list(start_pose))

            point, normal, face = self.sample_contact(
                primitive_name=primitive)
            if point is not None:
                have_contact = True
                break
            if k >= 10:
                print("Failed to sample valid contact")
                break
        if have_contact:
            palm_poses_world = self.get_palm_poses_world_frame(
                point,
                normal,
                primitive_name=primitive)

            contact_obj_frame = util.convert_reference_frame(
                palm_poses_world, obj_pose_initial, util.unit_pose()
            )

            if random_goal:
                if start_pose is not None:
                    start_pose_np = np.asarray(util.pose_stamped2list(start_pose))
                    start_pos = start_pose_np[:3]

                    rand_T = self.get_rand_trans_yaw_T(start_pos)
                    rand_T_pose = util.pose_from_matrix(rand_T)
                    obj_pose_final = util.transform_pose(start_pose, rand_T_pose)
                else:
                    x, y, q, _, obj_pose_final = self.get_rand_init(
                        execute=False, ind=init_id)
            else:
                obj_pose_final = util.list2pose_stamped(
                    self.init_poses[init_id])

            if primitive == 'push':
                # embed()
                obj_pose = self.get_obj_pose()[0] if start_pose is None else start_pose
                normal = -normal # TODO check why I need to do this
                new_pt, _ = util.project_point2plane(normal,
                                                     np.array([0, 0, 1]),
                                                     np.array([[0, 0, 0]]))
                face_normal_2d = (new_pt / np.linalg.norm(new_pt))[:-1]
                pose_2d = planning_helper.get_2d_pose(obj_pose)
                pose_vector_2d = np.array([np.cos(pose_2d[-1]), np.sin(pose_2d[-1])])
                angle = np.arccos(np.dot(face_normal_2d, pose_vector_2d))
                cross = np.cross(face_normal_2d, pose_vector_2d)
                if cross < 0.0:
                    angle = -angle
                    print('switching once!')

                palm_poses_world_dict = {}
                palm_poses_world_dict['right'] = palm_poses_world
                palm_poses_world_dict['left'] = palm_poses_world
                palm_y_normals = self.get_palm_y_normals(palm_poses=palm_poses_world_dict)

                right_palm_normal = util.pose_stamped2np(palm_y_normals['right'])[:3] - util.pose_stamped2np(palm_poses_world)[:3]
                if np.dot(right_palm_normal, normal) < 0.0:
                    angle = -angle
                    print('switching again!')

                primitive_args['pusher_angle'] = angle

            primitive_args['palm_pose_r_object'] = contact_obj_frame
            primitive_args['palm_pose_r_world'] = palm_poses_world
            primitive_args['object_pose1_world'] = obj_pose_initial
            primitive_args['object_pose2_world'] = obj_pose_final
            primitive_args['table_face'] = init_id
            primitive_args['N'] = self.calc_n(obj_pose_initial, obj_pose_final)

            return primitive_args
        else:
            return None


class DualArmPrimitives(EvalPrimitives):
    """
    Helper class for evaluating the closed loop performance of
    grasp and pivot manipulation primitives
    """
    def __init__(self, cfg, pb_client, object_id, mesh_file, goal_face=None, load_from_external=True):
        """
        Constructor, sets up samplers for primitive problem
        instances using the 3D model of the object being manipulated.
        Sets up an internal set of valid stable object orientations,
        specified from config file, and internal mesh of the object
        in the environment. Also sets up dual arm manipulation graph
        with external helper functions/classes.

        Args:
            cfg (YACS CfgNode): Configuration parameters
            object_id (int): PyBullet object id of object
                being manipulated
            mesh_file (str): Absolute path of the .stl file
                with the manipulated object
            goal_face (int, optional): Index of which face should be
                in contact with table in goal pose.
            load_from_external (bool, optional): If True, try to
                load sampled grasps from externally provided file
        """
        super(DualArmPrimitives, self).__init__(
            cfg=cfg,
            pb_client=pb_client,
            object_id=object_id,
            mesh_file=mesh_file
        )

        self.num_grasp_samples = self.cfg.NUM_GRASP_SAMPLES
        self.grasp_distance_tol = self.cfg.GRASP_DIST_TOLERANCE
        self._min_y_palm = self.cfg.GRASP_MIN_Y_PALM_DEG
        self._max_y_palm = self.cfg.GRASP_MAX_Y_PALM_DEG
        self.load_from_external = load_from_external

        self.initialize_graph_resources()
        self.initialize_object(object_id, mesh_file)

    def initialize_graph_resources(self):
        """Sets up all the internal resources for sampling and building
        the manipulation graphs that don't change
        """
        self.listener = tf.TransformListener()
        self.br = tf.TransformBroadcaster()
        self.gripper_name = 'mpalms_all_coarse.stl'
        self.table_name = 'table_top_collision.stl'

        self.table = CollisionBody(
            mesh_name=os.path.join(
                os.environ["CODE_BASE"],
                'catkin_ws/src/config/descriptions/meshes/table',
                self.table_name)
        )
        table_pose = self.cfg.BODY_TABLE_TF
        self.table.setCollisionPose(
            self.table.collision_object,
            util.list2pose_stamped(table_pose)
        )

        self.gripper_left = CollisionBody(
            mesh_name=os.path.join(
                os.environ["CODE_BASE"],
                'catkin_ws/src/config/descriptions/meshes/mpalm',
                self.gripper_name)
        )
        self.gripper_right = CollisionBody(
            mesh_name=os.path.join(
                os.environ["CODE_BASE"],
                'catkin_ws/src/config/descriptions/meshes/mpalm',
                self.gripper_name)
        )

        self.proposals_base_frame = util.list2pose_stamped(
            [0.45, 0, 0, 0, 0, 0, 1])

    def initialize_object(self, object_id, mesh_file):
        """
        Set up the internal variables that keep track of where the mesh
        is in the world so that contacts and random poses can be computed.
        Only samples new grasps if a new mesh file has been provided

        Args:
            object_id (int): PyBullet unique object id of the object
            mesh_file (str): Path to the .stl file with the object geometry
            goal_face (int): Index of which face of the object should be
                touching the table in the goal configuration
        """
        if mesh_file != self.mesh_file:
            self.initialize_object_stable_poses(object_id, mesh_file)
            self._setup_graph()
            self.goal_face = None

    def get_valid_goal_faces(self):
        """Function to get a list with values indicating valid goal faces
        """
        self.grasping_graph = sampling.build_placement_graph(grasp_samples=self.grasp_samples)
        valid_goal_faces_keys = self.grasping_graph.neighbours.keys()
        valid_grasp_goal_faces = []
        for key in valid_goal_faces_keys:
            face = int(key.split('_grasping')[0])
            valid_grasp_goal_faces.append(face)
        return valid_grasp_goal_faces

    def reset_graph(self, goal_face):
        """Resets the manipulation graph with the specified goal face. If
        the specified goal face is the same as what is currently used,
        nothing happens. If a new goal face is specified, the grasping graph
        is rebuilt using this as the goal face and the nominal goal pose
        is re-specified

        Args:
            goal_face (int): Index of which face should contact the table in the
                goal configuration
        """
        if goal_face != self.goal_face:
            self.goal_face = goal_face

            self._build_and_sample_graph()

            self.goal_pose_prop_frame_nominal = self.grasp_samples.collision_free_samples[
                'object_pose'][self.goal_face][0]
            self.goal_pose_world_frame_nominal = util.convert_reference_frame(
                pose_source=self.goal_pose_prop_frame_nominal,
                pose_frame_target=util.unit_pose(),
                pose_frame_source=self.proposals_base_frame
            )

    def _setup_graph(self):
        """
        Set up 3D mesh-based manipulation graph variables
        """
        self._object = {}
        self._object['file'] = self.mesh_file
        self._object['object'] = Object(
            mesh_name=self.mesh_file
        )

        self.sampler = sampling.Sampling(
            self.proposals_base_frame,
            self._object,
            self.table,
            self.gripper_left,
            self.gripper_right,
            self.listener,
            self.br
        )

        got_samples = False
        if self.load_from_external:
            try:
                obj_name = self.mesh_file.split('/meshes/objects/')[1].split('.stl')[0]
                fname = os.path.join(os.environ['CODE_BASE'], self.cfg.GRASP_SAMPLES_DIR, obj_name, 'collision_free_samples.pkl')
                print('TRYING TO LOAD GRASPS FROM: ' + str(fname))
                with open(fname, 'rb') as f:
                    collision_free_samples = pickle.load(f)
                self.grasp_samples = grasp_sampling.GraspSampling(
                    sampler=self.sampler,
                    num_samples=self.num_grasp_samples,
                    point_dist_tol=self.grasp_distance_tol,
                    is_visualize=True,
                    load_from_external=True
                )
                self.grasp_samples.collision_free_samples = collision_free_samples
                got_samples = True
            except:
                pass
        if not got_samples:
            # this is where we sample the grasps on the object
            self.grasp_samples = grasp_sampling.GraspSampling(
                sampler=self.sampler,
                num_samples=self.num_grasp_samples,
                point_dist_tol=self.grasp_distance_tol,
                is_visualize=True
            )

        # print("lever sampling: ")
        # self.lever_samples_global = lever_sampling.LeverSampling(
        #     self.sampler
        # )

    def _build_and_sample_graph(self):
        """
        Function to build the manipulation graph based on the 3D mesh of the
        manipulated object and the palm/table meshes. NOTE: this function
        compute the placement graph with respect to a particular goal face.
        To sample actions that reach a different goal face, have to reset the
        graph with a new goal face.
        """
        # hard coding loop over GRASPING neighbors for now, leaving out levering neighbors
        self.grasping_graph = sampling.build_placement_graph(grasp_samples=self.grasp_samples)
        self.grasping_graph.neighbours = self.grasping_graph.neighbours
        goal_key = str(self.goal_face) + '_grasping'
        try:
            self.goal_neighbors = list(self.grasping_graph.neighbours[goal_key])
        except KeyError as e:
            print('failed to build graph with provided goal face')
            print(e)
            raise KeyError(e)

        # create dictionaries keyed by the PRIMITIVE TYPES
        # subdictionaries are keyed by the START faces
        # must be rebuilt when new goal face is specified
        self.node_seq_dict = {}
        self.intersection_dict_grasp_dict = {}
        self.placement_seq_dict = {}
        self.sample_seq_dict = {}
        self.primitive_seq_dict = {}
        for j, key in enumerate(['grasp', 'lever']):
            self.node_seq_dict[key] = {}
            self.intersection_dict_grasp_dict[key] = {}
            self.placement_seq_dict[key] = {}
            self.sample_seq_dict[key] = {}
            self.primitive_seq_dict[key] = {}
            for i, end_face_name in enumerate(self.goal_neighbors):
                end_face = int(end_face_name[0].split('_grasping')[0])
                self.node_seq_dict[key][end_face] = None
                self.intersection_dict_grasp_dict[key][end_face] = None
                self.placement_seq_dict[key][end_face] = None
                self.sample_seq_dict[key][end_face] = None
                self.primitive_seq_dict[key][end_face] = None

        # hard coding sampling over 6 faces for now, should be parameterized based on faces on the mesh?
        for i, end_face_name in enumerate(self.goal_neighbors):
            # grasping
            end_face = int(end_face_name[0].split('_grasping')[0])
            try:
                node_seq, intersection_dict_grasp = sampling.search_placement_graph(
                    grasp_samples=self.grasp_samples,
                    placement_list=[self.goal_face, end_face]
                )
            except KeyError as e:
                print('Could not search placement graph')
                print(e)
                raise ValueError('Failed to build graph, exiting')

            placement_seq, sample_seq, primitive_seq = sampling.search_primitive_graph(
                _node_sequence=node_seq,
                intersection_dict_grasp=intersection_dict_grasp
            )

            self.node_seq_dict['grasp'][end_face] = node_seq
            self.intersection_dict_grasp_dict['grasp'][end_face] = intersection_dict_grasp
            self.placement_seq_dict['grasp'][end_face] = placement_seq
            self.sample_seq_dict['grasp'][end_face] = sample_seq
            self.primitive_seq_dict['grasp'][end_face] = primitive_seq

            # levering
            # node_seq, intersection_dict_lever = sampling.search_placement_graph(
            #     grasp_samples=None,
            #     lever_samples=self.lever_samples,
            #     placement_list=[i, self.goal_face]
            # )

            # placement_seq, sample_seq, primitive_seq = sampling.search_primitive_graph(
            #     _node_sequence=node_seq,
            #     intersection_dict_grasp=None,
            #     intersection_dict_lever=intersection_dict_lever
            # )

            # self.node_seq_dict['lever'][i] = node_seq
            # self.intersection_dict_grasp_dict['lever'][i] = intersection_dict_lever
            # self.placement_seq_dict['lever'][i] = placement_seq
            # self.sample_seq_dict['lever'][i] = sample_seq
            # self.primitive_seq_dict['lever'][i] = primitive_seq

        # just get the first object pose on the goal face (should all be the same)
        self.goal_pose = self.grasp_samples.collision_free_samples['object_pose'][self.goal_face][0]

    def get_nominal_init(self, ind, sample=0, primitive_name='grasp'):
        """
        Get the nominal object initial object pose corresponding
        to a particular stable placement, indicated by argument "ind"

        Args:
            ind (int): Index of the stable placement/face
            sample (int): Which sample id in the grasp samples dict,
                Defaults to 0

        Returns:
            PoseStamped: Initial object pose in the world frame
        """
        if primitive_name == 'grasp':
            init_object_pose = self.grasp_samples.collision_free_samples['object_pose'][ind][sample]
        # elif primitive_name == 'lever':
        #     init_object_pose = self.lever_samples.samples_dict['object_pose'][ind][sample]
        init_object_pose_world = util.convert_reference_frame(
            pose_source=init_object_pose,
            pose_frame_target=util.unit_pose(),
            pose_frame_source=self.proposals_base_frame
        )
        return init_object_pose_world

    def get_valid_ind(self):
        """Return a valid index corresponding to a stable configuration
        that is one step away in the sequence graph

        Returns:
            int: Index of stable config that's valid for the current
                goal face
        """
        valid = False
        invalid = []
        k = 0
        max_k = 20
        while not valid:
            # ind = np.random.randint(len(self.sample_seq_dict['grasp']))
            # key = self.sample_seq_dict['grasp'].keys()[ind]
            ind = random.sample(self.sample_seq_dict['grasp'].keys(), 1)[0]
            if len(self.sample_seq_dict['grasp'][ind]) == 1:
                return ind
            else:
                if ind not in invalid:
                    invalid.append(ind)
            if len(invalid) == len(self.sample_seq_dict['grasp'].keys()):
                break
        return None

    def get_rand_init(self, execute=True, ind=None):
        """
        Get a random initial object configuration, based on the precomputed
        stable poses of the object

        Args:
            execute (bool, optional): True if object state in environment
                should be updated to the sampled pose. Defaults to True.
            ind (int, optional): Index of object face to be in contact
                with table surface. Defaults to None.

        Returns:
            6-element tuple containing:
            float: Randomly sampled x position of object
            float: Randomly sampled y position of object
            float: Randomly sampled delta-orientation of object
            float: Randomly sampled absolute-orientation of object
            float: Index of object face that is in contact with table
            PoseStamped: World frame pose that was sampled
        """
        # sample from the sample_sequence list to get a face and pose
        # that connects to the goal
        if ind is None:
            valid = False
            print('getting random initial placement')
            k = 0
            while not valid:
                # ind = np.random.randint(low=0, high=6)
                ind = random.sample(self.sample_seq_dict['grasp'], 1)[0]
                k += 1
                print(k)
                if len(self.sample_seq_dict['grasp'][ind]) == 1:
                    valid = True

        # perturb that initial pose with a translation and a yaw
        nominal_init_pose = self.get_nominal_init(ind)

        nominal_pose_list = util.pose_stamped2list(nominal_init_pose)
        pos = nominal_pose_list[:3]
        rand_T = self.get_rand_trans_yaw_T(pos)

        rand_T_pose = util.pose_from_matrix(rand_T)
        rand_pose = util.transform_pose(nominal_init_pose, rand_T_pose)
        rand_pose_list = util.pose_stamped2list(rand_pose)

        x, y, q = rand_pose_list[0], rand_pose_list[1], rand_pose_list[3:]
        dq = common.rot2quat(rand_T[:-1, :-1])

        if execute:
            # NOTE: z value from nominal pose and z value when
            # object is placed in simulator will differ (simulator
            #  z will be larger)
            nom_z = nominal_init_pose.pose.position.z
            p.resetBasePositionAndOrientation(
                self.object_id,
                [x, y, nom_z],
                q,
                self.pb_client)
            time.sleep(1.0)
            world_pose = self.get_obj_pose()[0]
            world_pose.pose.position.z = nom_z
        else:
            world_pose = rand_pose

        time.sleep(0.5)
        self.transform_mesh_world(new_pose=util.pose_stamped2list(world_pose))
        return x, y, dq, q, ind, world_pose

    def palm_pose_prop_to_obj_frame(self, palm_poses_prop_frame,
                                    obj_pose_world_frame):
        """Helper to convert the palm pose reference frame from
        the proposals frame to the object frame, by first converting
        to the world frame and then using the current object pose
        to convert to the object frame

        Args:
            palm_poses_prop_frame (dict): Dictionary of palm poses, type
                PoseStamped, keyed by ['right', 'left'], in the proposals
                base frame
            obj_pose_world_frame (PoseStamped): World frame object pose

        Returns:
            dict: Dictionary of object frame palm poses, keyed by
                ['right', 'left']
        """
        palm_poses_obj_frame = {}
        for arm in palm_poses_prop_frame.keys():
            nom_world_frame = util.convert_reference_frame(
                palm_poses_prop_frame[arm],
                util.unit_pose(),
                self.proposals_base_frame
            )
            palm_poses_obj_frame[arm] = util.convert_reference_frame(
                nom_world_frame,
                obj_pose_world_frame,
                util.unit_pose()
            )
        return palm_poses_obj_frame

    def get_palm_poses_world_frame(self, ind, obj_world,
                                   sample_ind=None, primitive='grasp',
                                   execute=True):
        """
        Function to get the palm poses corresponding to some contact points
        on the object for grasping or pivoting. This function can modify the
        object pose based on heuristics that increase the likelihood of passing
        motion planning.

        Args:
            ind (int): Index/face id of the initial pose placement
            obj_world (PoseStamped): Object pose in world frame
            sample (int, optional): [description]. Defaults to None.
            execute (bool, optional): If True, execute the modifications to
                the object pose in the PyBullet simulation

        Returns:
            dict: Dictionary with 'right' and 'left' keys, each corresponding
                to the poses of the palms in the world frame
            PoseStamped: New world frame object pose
        """
        # ind_key = self.sample_seq_dict['grasp'].keys()[ind]
        if primitive == 'grasp':
            if len(self.sample_seq_dict['grasp'][ind]) != 1:
                # raise ValueError('Only sampling one step reachable goals right now')
                print('Only sampling one step reachable goals right now')
                print("length of sequence: " +
                      str(len(self.sample_seq_dict['grasp'][ind])))
                if ind == self.goal_face:
                    print('Start and Goal stable placements are the same! '\
                          ', please make sure they are different!')
                return None
            # if sample_ind is None:
                # TODO handle cases where its two steps away
                # number_samples = len(self.sample_seq_dict['grasp'][ind][0])
                # sample_ind = np.random.randint(low=0, high=number_samples)

            try:
                sample_id = random.sample(self.sample_seq_dict['grasp'][ind][0], 1)[0]
                sample_index = self.grasp_samples.collision_free_samples['sample_ids'][ind].index(sample_id)
            except:
                print('bad sample index')
                embed()

            right_prop_frame = self.grasp_samples.collision_free_samples[
                'gripper_poses'][ind][sample_index][1]
            left_prop_frame = self.grasp_samples.collision_free_samples[
                'gripper_poses'][ind][sample_index][0]

            nominal_obj_pose = self.get_nominal_init(ind=ind)

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

            right_obj_frame = util.convert_reference_frame(
                pose_source=right_nom_world_frame,
                pose_frame_target=nominal_obj_pose,
                pose_frame_source=util.unit_pose()
            )

            left_obj_frame = util.convert_reference_frame(
                pose_source=left_nom_world_frame,
                pose_frame_target=nominal_obj_pose,
                pose_frame_source=util.unit_pose()
            )

            right_world_frame = util.convert_reference_frame(
                pose_source=right_obj_frame,
                pose_frame_target=util.unit_pose(),
                pose_frame_source=obj_world
            )

            left_world_frame = util.convert_reference_frame(
                pose_source=left_obj_frame,
                pose_frame_target=util.unit_pose(),
                pose_frame_source=obj_world
            )

            ## HERE IS WHERE TO CONVERT EVERYTHING SUCH THAT WE GENERATE USEFUL DATA ###
            obj_pose_mod, right_world_frame_mod, left_world_frame_mod = \
                self.modify_init_goal(
                    ind,
                    sample_index,
                    sample_id,
                    right_world_frame,
                    left_world_frame,
                    obj_nominal=obj_world,
                    execute=execute)

            palm_poses_world = {}
            palm_poses_world['right'] = right_world_frame_mod
            palm_poses_world['left'] = left_world_frame_mod


        # elif (primitive == 'lever' or primitive == 'pivot'):
        #     # node_seq, intersection_dict_lever = sampling.search_placement_graph(
        #     #     grasp_samples=None,
        #     #     lever_samples=self.lever_samples,
        #     #     placement_list=[self.goal_face, ind]
        #     # )

        #     # placement_seq, sample_seq, _ = sampling.search_primitive_graph(
        #     #     _node_sequence=node_seq,
        #     #     intersection_dict_grasp=None,
        #     #     intersection_dict_lever=intersection_dict_lever
        #     # )

        #     # HACKY WAY TO CHECK IF THERE ARE SAMPLES
        #     # if isinstance(sample_seq[0][0], int):
        #     if isinstance(self.sample_seq_dict['lever'][ind][0][0], int):
        #         # sample_id = sample_seq[0][0]
        #         sample_id = self.sample_seq_dict['lever'][ind][0][0]
        #         sample_index = self.lever_samples.samples_dict['sample_ids'][ind].index(sample_id)
        #     else:
        #         return None

        #     right_prop_frame = self.lever_samples.samples_dict[
        #         'gripper_poses'][ind][sample_index][1]
        #     left_prop_frame = self.lever_samples.samples_dict[
        #         'gripper_poses'][ind][sample_index][0]

        #     right_nom_world_frame = util.convert_reference_frame(
        #         pose_source=right_prop_frame,
        #         pose_frame_target=util.unit_pose(),
        #         pose_frame_source=self.proposals_base_frame
        #     )

        #     left_nom_world_frame = util.convert_reference_frame(
        #         pose_source=left_prop_frame,
        #         pose_frame_target=util.unit_pose(),
        #         pose_frame_source=self.proposals_base_frame
        #     )

        #     palm_poses_world = {}
        #     palm_poses_world['right'] = right_nom_world_frame
        #     palm_poses_world['left'] = left_nom_world_frame

        #     # should be able to do same forward pointing check here
        #     self.goal_pose_world_frame_mod = copy.deepcopy(self.goal_pose_world_frame_nominal)

        return palm_poses_world, obj_pose_mod

    def modify_init_goal(self, ind, sample_index, sample_id,
                         right_world_frame, left_world_frame,
                         obj_nominal, execute=True):
        """
        Function to modify the initial object/gripper and final
        object poses if necessary, to increase likelihood of
        kinematic and motion planning feasbility. Currently
        just checks if the palm poses at start/goal fall within
        some orientation range known to be "nice", and if not in
        that range rotates everything (object and palm poses) so
        that they are within that range

        Args:
            ind (int): Init face index
            sample_index (int): Index in self.grasp_samples of
                the object pose to use for init
            sample_id (int): ID of sample to use for inititial pose
            right_world_frame (PoseStamped): Right palm world frame
                corresponding to unmodified initial pose at sample_id
            left_world_frame (PoseStamped): Left palm world frame
                corresponding to unmodified initial pose at sample_id


        Returns:
            3-element tuple containing:
            - PoseStamped: Modified object pose in world frame
            - PoseStamped: Modified right palm pose in world frame
            - PoseStamped: Modified left palm pose in world frame
        """
        normal_y = util.list2pose_stamped([0, -1, 0, 0, 0, 0, 1])

        normal_y_pose_right_world = util.transform_pose(
            normal_y, right_world_frame
        )

        theta_r = np.arccos(np.dot(util.pose_stamped2list(
            normal_y_pose_right_world)[:3], [0, -1, 0]))

        # obj_nominal = self.get_obj_pose()[0]
        flipped_hands = False

        if np.isnan(theta_r):
            theta_r = 0.0  # hack

        # if not (theta_r < np.deg2rad(30) or theta_r > np.deg2rad(150)):
        if not (theta_r < np.deg2rad(self._min_y_palm) or
                theta_r > np.deg2rad(self._max_y_palm)):
            # just yaw it by theta in the world frame
            palm_right_obj_frame = util.convert_reference_frame(
                pose_source=right_world_frame,
                pose_frame_target=obj_nominal,
                pose_frame_source=util.unit_pose()
            )
            palm_left_obj_frame = util.convert_reference_frame(
                pose_source=left_world_frame,
                pose_frame_target=obj_nominal,
                pose_frame_source=util.unit_pose()
            )

            sample_obj_q = common.quat_multiply(
                common.euler2quat([0, 0, theta_r]),
                util.pose_stamped2list(obj_nominal)[3:]
            ).tolist()

            sample_obj = util.list2pose_stamped(
                util.pose_stamped2list(obj_nominal)[:3] +
                sample_obj_q
            )

            if execute:
                p.resetBasePositionAndOrientation(
                    self.object_id,
                    util.pose_stamped2list(sample_obj)[:3],
                    util.pose_stamped2list(sample_obj)[3:],
                    self.pb_client)

                time.sleep(1.0)
                self.transform_mesh_world()

            sample_palm_right = util.convert_reference_frame(
                pose_source=palm_right_obj_frame,
                pose_frame_target=util.unit_pose(),
                pose_frame_source=sample_obj
            )
            sample_palm_left = util.convert_reference_frame(
                pose_source=palm_left_obj_frame,
                pose_frame_target=util.unit_pose(),
                pose_frame_source=sample_obj
            )

        else:
            sample_palm_right = right_world_frame
            sample_palm_left = left_world_frame
            # sample_obj = self.get_nominal_init(ind)
            sample_obj = obj_nominal

        new_normal_y_pose_prop = util.transform_pose(
            normal_y, sample_palm_right)

        y_sign_negative = (new_normal_y_pose_prop.pose.position.y -
                           sample_palm_right.pose.position.y) < 0

        if y_sign_negative:
            sample_palm_left_tmp = copy.deepcopy(sample_palm_left)
            sample_palm_left = copy.deepcopy(sample_palm_right)
            sample_palm_right = sample_palm_left_tmp
            # print("FLIPPING HANDS!")
            flipped_hands = True

        sample_index_goal = self.grasp_samples.collision_free_samples['sample_ids'][self.goal_face].index(
            sample_id)
        right_prop_frame_goal = self.grasp_samples.collision_free_samples['gripper_poses'][self.goal_face][sample_index_goal][0]
        left_prop_frame_goal = self.grasp_samples.collision_free_samples['gripper_poses'][self.goal_face][sample_index_goal][1]
        palm_poses_prop_frame_goal = {}
        palm_poses_prop_frame_goal['right'] = right_prop_frame_goal
        palm_poses_prop_frame_goal['left'] = left_prop_frame_goal

        normal_y_pose_right_prop_goal = util.transform_pose(
            normal_y, right_prop_frame_goal
        )

        theta_r_goal = np.arccos(np.dot(util.pose_stamped2list(
            normal_y_pose_right_prop_goal)[:3], [0, -1, 0]))

        if np.isnan(theta_r_goal):
            theta_r_goal = 0.0  # hack

        # if not (theta_r_goal < np.deg2rad(30) or theta_r_goal > np.deg2rad(150)):
        if not (theta_r_goal < np.deg2rad(self._min_y_palm) or
                theta_r_goal > np.deg2rad(self._max_y_palm)):

            palm_poses_obj_frame = self.palm_pose_prop_to_obj_frame(
                palm_poses_prop_frame_goal,
                self.goal_pose_world_frame_nominal)

            sample_obj_goal_q = common.quat_multiply(
                common.euler2quat([0, 0, theta_r_goal]),
                util.pose_stamped2list(self.goal_pose_world_frame_nominal)[3:]
            ).tolist()

            sample_obj_goal = util.list2pose_stamped(
                util.pose_stamped2list(self.goal_pose_world_frame_nominal)[:3] +
                sample_obj_goal_q
            )

            new_right_prop_frame_goal_q = common.quat_multiply(
                common.euler2quat([0, 0, theta_r_goal]),
                util.pose_stamped2list(right_prop_frame_goal)[3:]
            ).tolist()
            new_right_prop_frame_goal = util.list2pose_stamped(
                util.pose_stamped2list(right_prop_frame_goal)[:3] +
                new_right_prop_frame_goal_q
            )

            new_left_prop_frame_goal_q = common.quat_multiply(
                common.euler2quat([0, 0, theta_r_goal]),
                util.pose_stamped2list(left_prop_frame_goal)[3:]
            ).tolist()
            new_left_prop_frame_goal = util.list2pose_stamped(
                util.pose_stamped2list(left_prop_frame_goal)[:3] +
                new_left_prop_frame_goal_q
            )

            self.goal_pose_world_frame_mod = sample_obj_goal
        elif theta_r_goal > np.deg2rad(135) and not flipped_hands:
            sample_obj_goal_q = common.quat_multiply(
                common.euler2quat([0, 0, theta_r_goal]),
                util.pose_stamped2list(self.goal_pose_world_frame_nominal)[3:]
            ).tolist()

            sample_obj_goal = util.list2pose_stamped(
                util.pose_stamped2list(self.goal_pose_world_frame_nominal)[:3] +
                sample_obj_goal_q
            )

            new_right_prop_frame_goal_q = common.quat_multiply(
                common.euler2quat([0, 0, theta_r_goal]),
                util.pose_stamped2list(right_prop_frame_goal)[3:]
            ).tolist()
            new_right_prop_frame_goal = util.list2pose_stamped(
                util.pose_stamped2list(right_prop_frame_goal)[:3] +
                new_right_prop_frame_goal_q
            )

            new_left_prop_frame_goal_q = common.quat_multiply(
                common.euler2quat([0, 0, theta_r_goal]),
                util.pose_stamped2list(left_prop_frame_goal)[3:]
            ).tolist()
            new_left_prop_frame_goal = util.list2pose_stamped(
                util.pose_stamped2list(left_prop_frame_goal)[:3] +
                new_left_prop_frame_goal_q
            )

            self.goal_pose_world_frame_mod = sample_obj_goal
        else:
            self.goal_pose_world_frame_mod = copy.deepcopy(self.goal_pose_world_frame_nominal)

            new_right_prop_frame_goal = copy.deepcopy(right_prop_frame_goal)
            new_left_prop_frame_goal = copy.deepcopy(left_prop_frame_goal)

        if not flipped_hands:
            new_normal_y_pose_goal = util.transform_pose(
                normal_y, new_right_prop_frame_goal)
            goal_y_sign_negative = (new_normal_y_pose_goal.pose.position.y -
                                    new_right_prop_frame_goal.pose.position.y) < 0
        else:
            new_normal_y_pose_goal = util.transform_pose(
                normal_y, new_left_prop_frame_goal)
            goal_y_sign_negative = (new_normal_y_pose_goal.pose.position.y -
                                    new_left_prop_frame_goal.pose.position.y) < 0

        if not goal_y_sign_negative:
            sample_obj_goal_q = common.quat_multiply(
                common.euler2quat([0, 0, np.pi]),
                util.pose_stamped2list(self.goal_pose_world_frame_mod)[3:]
            ).tolist()

            sample_obj_goal = util.list2pose_stamped(
                util.pose_stamped2list(self.goal_pose_world_frame_mod)[:3] +
                sample_obj_goal_q
            )
            self.goal_pose_world_frame_mod = sample_obj_goal
        return sample_obj, sample_palm_right, sample_palm_left

    def get_random_primitive_args(self, ind=None, primitive='grasp',
                                  random_goal=False, execute=True,
                                  start_pose=None, penetration_delta=5e-3):
        """
        Function to abstract away all the setup for sampling a primitive
        instance

        Args:
            ind (int): Index of object face that should contact the ground in
                start configuration.
            primitive (str): Which primitive to sample
            random_goal (bool): If True, randomly perturb the nominal goal
                state with some [x, y, theta] in SE(2)
            execute (bool): If True, reset the object in the PyBullet
                simulation to the sampled start state
            start_pose (PoseStamped): If included, random start pose will not
                be sampled, and instead only a random grasp and goal pose
                (if random_goal==True) will be sampled, and the specified
                start pose will be used.
            penetration_delta (float): Minimum penetration noise to add to the
                normal direction of the palm pose, to increase likelihood of
                sticking contact

        Returns:
            dict: Inputs to primitive planner that can be directly executed
        """
        primitive_args = {}
        primitive_args['name'] = primitive
        primitive_args['object'] = None
        primitive_args['N'] = 50
        primitive_args['init'] = True
        primitive_args['table_face'] = 0

        k = 0
        have_contact = False
        while True:
            if start_pose is None or ind is None:
                # if we have not provided a starting pose, get a new one
                x, y, dq, q, init_id, obj_pose_world_nom = self.get_rand_init(execute=execute, ind=ind)
            else:
                # if we have provided a starting pose, reset the object and use that pose
                p.resetBasePositionAndOrientation(
                    self.object_id,
                    util.pose_stamped2list(start_pose)[:3],
                    util.pose_stamped2list(start_pose)[3:]
                )
                time.sleep(0.5)
                obj_pose_world_nom = self.get_obj_pose()[0]
                init_id = ind

            time.sleep(1.5)
            # get world frame palm poses, using the nominal start object pose
            # will modify the start object pose if necessary
            palm_poses_world, obj_pose_world = self.get_palm_poses_world_frame(
                init_id,
                obj_pose_world_nom,
                primitive=primitive,
                execute=execute)

            if palm_poses_world is not None:
                have_contact = True
                break
            k += 1
            if k >= 10:
                print("FAILED")
                return None
        if have_contact:
            if random_goal:
                # randomly perturb the goal pose the same as we do for the start pose
                final_nominal_pose = self.goal_pose_world_frame_mod
                final_nominal_pose_list = util.pose_stamped2list(self.goal_pose_world_frame_mod)

                rand_T = self.get_rand_trans_yaw_T(final_nominal_pose_list[:3])
                rand_T_pose = util.pose_from_matrix(rand_T)
                obj_pose_final = util.transform_pose(final_nominal_pose, rand_T_pose)
            else:
                obj_pose_final = self.goal_pose_world_frame_mod

            palm_poses_obj_frame = {}
            # delta = np.random.random_sample() * \
            #     (penetration_delta - 0.5*penetration_delta) + \
            #     penetration_delta
            delta = 0.005
            y_normals = self.get_palm_y_normals(palm_poses_world)
            for key in palm_poses_world.keys():
                # try to penetrate the object a small amount
                if key == 'right':
                    palm_poses_world[key].pose.position.x -= delta*y_normals[key].pose.position.x
                    palm_poses_world[key].pose.position.y -= delta*y_normals[key].pose.position.y
                    palm_poses_world[key].pose.position.z -= delta*y_normals[key].pose.position.z

                palm_poses_obj_frame[key] = util.convert_reference_frame(
                    palm_poses_world[key], obj_pose_world, util.unit_pose())

            primitive_args['palm_pose_r_object'] = palm_poses_obj_frame['right']
            primitive_args['palm_pose_l_object'] = palm_poses_obj_frame['left']
            primitive_args['palm_pose_r_world'] = palm_poses_world['right']
            primitive_args['palm_pose_l_world'] = palm_poses_world['left']
            primitive_args['object_pose1_world'] = obj_pose_world
            primitive_args['object_pose2_world'] = obj_pose_final
            primitive_args['table_face'] = init_id

            return primitive_args
        else:
            return None

    def get_palm_y_normals(self, palm_poses, current=False):
        """
        Gets the updated world frame normal direction of the palms
        """
        normal_y = util.list2pose_stamped([0, 1, 0, 0, 0, 0, 1])
        normal_y_poses_world = {}

        normal_y_poses_world['right'] = util.transform_pose(normal_y, palm_poses['right'])
        normal_y_poses_world['left'] = util.transform_pose(normal_y, palm_poses['left'])

        return normal_y_poses_world


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
    np_seed = args.np_seed
    np.random.seed(np_seed)

    # setup yumi
    yumi_ar = Robot('yumi_palms',
                    pb=True,
                    pb_cfg={'gui': True, 'realtime': True},
                    arm_cfg={'self_collision': False, 'seed': np_seed})
    yumi_ar.arm.set_jpos(cfg.RIGHT_INIT + cfg.LEFT_INIT, ignore_physics=True)

    r_gel_id = cfg.RIGHT_GEL_ID
    l_gel_id = cfg.LEFT_GEL_ID

    alpha = cfg.ALPHA
    K = cfg.GEL_CONTACT_STIFFNESS
    restitution = cfg.GEL_RESTITUION

    # p.changeDynamics(
    #     yumi_ar.arm.robot_id,
    #     r_gel_id,
    #     restitution=restitution,
    #     contactStiffness=K,
    #     contactDamping=alpha*K,
    #     rollingFriction=args.rolling
    # )

    # p.changeDynamics(
    #     yumi_ar.arm.robot_id,
    #     l_gel_id,
    #     restitution=restitution,
    #     contactStiffness=K,
    #     contactDamping=alpha*K,
    #     rollingFriction=args.rolling
    # )

    lateral = 0.5
    p.changeDynamics(
        yumi_ar.arm.robot_id,
        r_gel_id,
        lateralFriction=lateral
    )

    p.changeDynamics(
        yumi_ar.arm.robot_id,
        l_gel_id,
        lateralFriction=lateral
    )

    # p.changeDynamics(
    #     yumi_ar.arm.robot_id,
    #     cfg.TABLE_ID,
    #     lateralFriction=0.1)

    yumi_gs = YumiGelslimPybullet(
        yumi_ar,
        cfg,
        exec_thread=True)

    # yumi_gs.update_joints(cfg.RIGHT_INIT + cfg.LEFT_INIT)

    if args.object:
        # box_id = yumi_ar.pb_client.load_urdf(
        #     args.config_package_path +
        #     'descriptions/urdf/'+args.object_name+'.urdf',
        #     cfg.OBJECT_POSE_3[0:3],
        #     cfg.OBJECT_POSE_3[3:]
        # )
        stl_file = os.path.join(args.config_package_path, 'descriptions/meshes/objects', args.object_name+'.stl')
        # stl_file = os.path.join(args.config_package_path, 'descriptions/meshes/objects/cuboids', args.object_name+'.stl')
        # obj_name = 'test_cylinder_'+str(np.random.randint(4999))
        # stl_file = os.path.join(args.config_package_path, 'descriptions/meshes/objects/cylinders', obj_name + '.stl')
        # stl_file = os.path.join(args.config_package_path, 'descriptions/meshes/objects/ycb_objects', args.object_name+'.stl')
        tmesh = trimesh.load_mesh(stl_file)
        init_pose = tmesh.compute_stable_poses()[0][0]
        pos = init_pose[:-1, -1]
        ori = common.rot2quat(init_pose[:-1, :-1])
        box_id = yumi_ar.pb_client.load_geom(
            shape_type='mesh',
            visualfile=stl_file,
            collifile=stl_file,
            mesh_scale=[1.0, 1.0, 1.0],
            base_pos=[0.45, 0, pos[-1]],
            base_ori=ori,
            rgba=[0.7, 0.2, 0.2, 1.0],
            mass=0.03)

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

    mesh_file = args.config_package_path + \
                'descriptions/meshes/objects/' + \
                args.object_name + '.stl'

    # mesh_file = args.config_package_path + \
    #             'descriptions/meshes/objects/cuboids/' + \
    #             args.object_name + '.stl'

    # mesh_file = args.config_package_path + \
    #             'descriptions/meshes/objects/cylinders/' + \
    #             obj_name + '.stl'                    

    # mesh_file = args.config_package_path + \
    #             'descriptions/meshes/objects/ycb_objects/' + \
    #             args.object_name + '.stl'
    exp_single = SingleArmPrimitives(
        cfg,
        yumi_ar.pb_client.get_client_id(),
        box_id,
        mesh_file)

    if args.primitive == 'grasp' or args.primitive == 'pivot':
        exp_double = DualArmPrimitives(
            cfg,
            yumi_ar.pb_client.get_client_id(),
            box_id,
            mesh_file)
        valid_goal_faces_keys = exp_double.grasping_graph.neighbours.keys()
        valid_grasp_goal_faces = []
        for key in valid_goal_faces_keys:
            face = int(key.split('_grasping')[0])
            valid_grasp_goal_faces.append(face)
        print('valid grap goal faces: ', valid_grasp_goal_faces)
        exp_running = exp_double

        goal_pose = util.pose_stamped2list(
            exp_double.get_nominal_init(ind=exp_double.goal_face))
    else:
        exp_running = exp_single
        goal_pose = cfg.OBJECT_FINAL

    # trans_box_id = yumi_ar.pb_client.load_urdf(
    #     args.config_package_path +
    #     'descriptions/urdf/'+args.object_name+'_trans.urdf',
    #     goal_pose[:3],
    #     goal_pose[3:]
    # )
    trans_box_id = yumi_ar.pb_client.load_geom(
        shape_type='mesh',
        visualfile=stl_file,
        collifile=stl_file,
        mesh_scale=[1.0, 1.0, 1.0],
        base_pos=[0.45, 0, pos[-1]],
        base_ori=ori,
        rgba=[0.1, 1.0, 0.1, 0.25],
        mass=0.03)
    for jnt_id in range(p.getNumJoints(yumi_ar.arm.robot_id)):
        p.setCollisionFilterPair(yumi_ar.arm.robot_id, trans_box_id, jnt_id, -1, enableCollision=False)

    p.setCollisionFilterPair(box_id, trans_box_id, -1, -1, enableCollision=False)

    # setup macro_planner
    action_planner = ClosedLoopMacroActions(
        cfg,
        yumi_gs,
        box_id,
        yumi_ar.pb_client.get_client_id(),
        args.config_package_path,
        object_mesh_file=mesh_file,
        replan=args.replan
    )

    trans_box_lock = threading.RLock()
    goal_viz = GoalVisual(
        trans_box_lock,
        trans_box_id,
        action_planner.pb_client,
        goal_pose)

    visualize_goal_thread = threading.Thread(
        target=goal_viz.visualize_goal_state)
    visualize_goal_thread.daemon = True
    visualize_goal_thread.start()

    contact_face = None

    face_success = [0] * 6

    # for face in range(2, 13):
    for face in range(2, 10):
        print("-------\n\n\nGOAL FACE NUMBER: " + str(face) + "\n\n\n-----------")
        start_time = time.time()
        try:
            exp_running.initialize_object(box_id, mesh_file)
            if primitive_name in ['grasp']:
                exp_running.reset_graph(face)
        except ValueError as e:
            print(e)
            print('Goal face: ' + str(face))
            continue
        face_success.append(0)
        for trial in range(5):
            print("Trial number: " + str(trial))
            print("Time so far: " + str(time.time() - start_time))
            if primitive_name == 'grasp':
                start_face = exp_double.get_valid_ind()
                if start_face is None:
                    print('Could not find valid start face')
                    continue
                plan_args = exp_running.get_random_primitive_args(ind=start_face,
                                                                  random_goal=True,
                                                                  execute=True)
            elif primitive_name in ['pull', 'push']:
                plan_args = exp_running.get_random_primitive_args(ind=face,
                                                                  random_goal=True,
                                                                  execute=True,
                                                                  primitive=primitive_name)

            start_pose = plan_args['object_pose1_world']
            goal_pose = plan_args['object_pose2_world']

            goal_viz.update_goal_state(util.pose_stamped2list(goal_pose))
            if args.debug:
                import simulation

                plan = action_planner.get_primitive_plan(primitive_name, plan_args, 'right')

                for i in range(10):
                    # simulation.visualize_object(
                    #     start_pose,
                    #     filepath="package://config/descriptions/meshes/objects/cuboids/" +
                    #         cuboid_fname.split('objects/cuboids')[1],
                    #     name="/object_initial",
                    #     color=(1., 0., 0., 1.),
                    #     frame_id="/yumi_body",
                    #     scale=(1., 1., 1.))
                    # simulation.visualize_object(
                    #     goal_pose,
                    #     filepath="package://config/descriptions/meshes/objects/cuboids/" +
                    #         cuboid_fname.split('objects/cuboids')[1],
                    #     name="/object_final",
                    #     color=(0., 0., 1., 1.),
                    #     frame_id="/yumi_body",
                    #     scale=(1., 1., 1.))
                    simulation.visualize_object(
                        start_pose,
                        filepath="package://config/descriptions/meshes/objects/" + args.object_name + '.stl',
                        name="/object_initial",
                        color=(1., 0., 0., 1.),
                        frame_id="/yumi_body",
                        scale=(1., 1., 1.))
                    simulation.visualize_object(
                        goal_pose,
                        filepath="package://config/descriptions/meshes/objects/" + args.object_name + '.stl',
                        name="/object_final",
                        color=(0., 0., 1., 1.),
                        frame_id="/yumi_body",
                        scale=(1., 1., 1.))
                    rospy.sleep(.1)
                simulation.simulate(plan, args.object_name + '.stl')
                # simulation.simulate(plan, 'realsense_box_experiments.stl')
            else:
                # local_plan = action_planner.get_primitive_plan(primitive_name, plan_args, 'right')
                try:
                #     for k, subplan in enumerate(local_plan):
                #         time.sleep(1.0)
                #         action_planner.playback_dual_arm('grasp', subplan, k)
                    result = action_planner.execute(
                        primitive_name,
                        plan_args,
                        contact_face=contact_face)

                    # if result is not None:
                    #     # print("reached final: " + str(result[0]))
                    #     print(result)
                    #     face_success[face] += 1
                    #     print("Face successes: ", face_success)
                except ValueError as e:
                    print("Value error: ")
                    print(e)

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

    parser.add_argument(
        '--np_seed', type=int,
        default=0
    )

    parser.add_argument(
        '--perturb',
        action='store_true'
    )

    args = parser.parse_args()
    main(args)
