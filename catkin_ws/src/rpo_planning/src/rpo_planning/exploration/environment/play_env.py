import os
import numpy as np
import random
import copy

import pybullet as p

from rpo_planning.utils import common as util


class PlayObjects(object):
    def __init__(self):
        self.name = None
        self.pose = None
        self.mesh = None
        self.obj_id = None


class PlayEnvironment(object):
    def __init__(self, pb_client, robot, cuboid_manager, cuboid_sampler, target_surface_ids, goal_visualization=False):
        self.pb_client = pb_client
        self.robot = robot
        self.cams = self.robot.cams
        self.robot_id = robot.yumi_ar.arm.robot_id
        self.cuboid_manager = cuboid_manager
        self.cuboid_sampler = cuboid_sampler
        self.goal_visualization = goal_visualization

        self.camera_inds = [0, 1, 2, 3]
        self.target_surface_ids = target_surface_ids

        # assume first target surface is the default
        self.default_surface_id = self.target_surface_ids[0]

        # table boundaries are [[x_min, x_max], [y_min, y_max]]
        self.table_boundaries = {}
        self.table_boundaries['x'] = np.array([0.1, 0.5])
        self.table_boundaries['y'] = np.array([-0.4, 0.4])

        self.clear_current_objects()

    @staticmethod
    def object_fly_away(obj_id):
        obj_pos = p.getBasePositionAndOrientation(obj_id)[0]
        return obj_pos[2] < -0.1

    @staticmethod
    def object_table_contact(robot_id, obj_id, table_id, pb_cl):
        table_contacts = p.getContactPoints(
            robot_id,
            obj_id,
            table_id,
            -1,
            pb_cl)

        n_list = []
        for pt in table_contacts:
            n_list.append(pt[-5])
        return len(table_contacts) > 0, n_list   

    @staticmethod
    def object_in_boundaries(obj_id, x_boundaries, y_boundaries, z_boundaries=None):
        obj_pos = p.getBasePositionAndOrientation(obj_id)[0]
        in_x = obj_pos[0] >= min(x_boundaries) and obj_pos[0] <= max(x_boundaries)
        in_y = obj_pos[1] >= min(y_boundaries) and obj_pos[1] <= max(y_boundaries)
        in_z = True if z_boundaries is None else obj_pos[2] >= min(z_boundaries) and obj_pos[2] <= max(z_boundaries)
        return in_x and in_y and in_z

    def _random_table_xy(self):
        x = np.random.random() * (max(self.table_boundaries['x']) - min(self.table_boundaries['x'])) + min(self.table_boundaries['x'])
        y = np.random.random() * (max(self.table_boundaries['y']) - min(self.table_boundaries['y'])) + min(self.table_boundaries['y'])
        return x, y

    def get_random_pose_mesh(self, tmesh):
        """Sample a random pose in the table top environment with a particular mesh object.
        This method computes a stable pose of the mesh using the internal trimesh function,
        then samples a position that is in the valid object region.

        Args:
            tmesh (Trimesh.T): [description]

        Returns:
            list: Random pose [x, y, z, qx, qy, qz, qw] in the tabletop environment
        """
        # TODO: implement functionality to be able to directly sample initial states which are not at z=0
        stable_poses = tmesh.compute_stable_poses()[0]
        # pose = np.random.choice(stable_poses, 1)
        pose = random.sample(stable_poses, 1)[0]
        x, y = self._random_table_xy()
        pose[0] = x
        pose[1] = y
        return util.pose_stamped2list(util.pose_from_matrix(pose))

    def clear_current_objects(self):
        self._current_obj_list = []

    def get_current_obj_info(self):
        return copy.deepcopy(self._current_obj_list)

    def _sample_cuboid(self, obj_fname=None):
        if obj_fname is None:
            obj_fname = self.cuboid_manager.get_cuboid_fname()
        obj_id, _, mesh, goal_obj_id = self.cuboid_sampler.sample_cuboid_pybullet(
            obj_fname, goal=self.goal_visualization)
        return mesh, obj_fname, obj_id, goal_obj_id

    def sample_objects(self, n=1):
        self.clear_current_objects()
        for _ in range(n):
            mesh, cuboid_fname, obj_id, goal_obj_id = self._sample_cuboid()
            obj_dict = {}
            obj_dict['mesh'] = mesh
            obj_dict['fname'] = cuboid_fname
            obj_dict['obj_id'] = obj_id
            obj_dict['goal_obj_id'] = goal_obj_id
            self._current_obj_list.append(obj_dict)

    def initialize_object_states(self):
        # check if current set of objects is empty
        if not len(self._current_obj_list):
            raise ValueError('Must sample objects in the environment before '
                             'initializing states')
        for i, obj_dict in enumerate(self._current_obj_list):
            # get a random start pose and set to that pose
            start_pose = self.get_random_pose_mesh(obj_dict['mesh'])
            obj_id = obj_dict['obj_id']
            self.pb_client.reset_body(obj_id, start_pose[:3], start_pose[3:])

    def segment_object(self, obj_body_id, pts, seg):
        obj_inds = np.where(seg == obj_body_id)
        obj_pts = pts[obj_inds[0], :]
        return obj_pts

    def segment_surface(self, surface_link_id, pts, seg):
        # use PyBullet's weird way of converting the segmentation labels based on body/link id
        surface_val = self.robot_id + (surface_link_id + 1) << 24
        surface_inds = np.where(seg == surface_val)

        # filter
        surface_pts = pts[surface_inds[0], :]
        return surface_pts

    def get_observation(self):
        # TODO handle if we just want to get a point cloud of the table

        # check if current set of objects is empty
        if not len(self._current_obj_list):
            raise ValueError('Must sample objects in the environment before '
                             'initializing states')

        depths = []
        segs = []
        raw_pcd_pts = []
        # get the full, unsegmented point cloud from each camera
        for i, cam in enumerate(self.cams):
            rgb, depth, seg = cam.get_images(
                get_rgb=True,
                get_depth=True,
                get_seg=True
            )

            pts_raw, colors_raw = cam.get_pcd(
                in_world=True,
                filter_depth=False,
                depth_max=1.0
            )

            raw_pcd_pts.append(pts_raw)
            segs.append(seg.flatten())

        # combine all point clouds from each viewpoint together
        raw_pcd_pts = np.concatenate(np.asarray(raw_pcd_pts, dtype=np.float32), axis=0)
        segs = np.concatenate(np.asarray(segs, dtype=np.float32), axis=0)

        # go through random objects, and get the individual segmented object point clouds
        obj_pcd_pts = []

        for i, obj_dict in enumerate(self._current_obj_list):
            obj_id = obj_dict['obj_id']
            pcd_pts = self.segment_object(obj_id, raw_pcd_pts, segs)
            obj_pcd_pts.append(pcd_pts)

        # TODO: incorporate sampling a target surface
        # # and then go through static target surfaces, as if they were objects
        target_surface_pts = []
        for i, obj_id in enumerate(self.target_surface_ids):
            pcd_pts = self.segment_surface(obj_id, raw_pcd_pts, segs)
            target_surface_pts.append(pcd_pts)
        obs = {}
        obs['object_pcds'] = obj_pcd_pts
        obs['surface_pcds'] = target_surface_pts
        return obs

    def _valid_object_status(self, object_table_contact, object_fly_away, object_in_boundaries, *args):
        """Function to check if object state is valid, depending on a number of symbolic states we obtain
        from the simulator. All inputs are meant to be boolean values computed using separate methods

        Args:
            object_table_contact (bool): True if contact was detected between the object and the table
            object_fly_away (bool): True if object position was detected as below the nominal table surface position
            object_in_boundaries (bool): True if object is inside designated table boundaries

        Returns:
            bool: True if object state is valid
        """
        return object_table_contact and not object_fly_away and object_in_boundaries

    def check_environment_status(self):
        valid = True
        for i, obj_dict in enumerate(self._current_obj_list):
            # check if object is contacting table
            obj_table_contacting = self.object_table_contact(
                robot_id=self.robot.yumi_ar.arm.robot_id, 
                obj_id=obj_dict['obj_id'], 
                table_id=self.default_surface_id, 
                pb_cl=self.robot.yumi_ar.pb_client.get_client_id())

            # check if object position is below the table
            obj_below_table = self.object_fly_away(obj_dict['obj_id'])

            # check if object is within valid table boundaries (not stuck behind the robot)
            obj_in_boundaries = self.object_in_boundaries(
                obj_dict['obj_id'], self.table_boundaries['x'], self.table_boundaries['y'])
            
            valid = valid and self._valid_object_status(
                obj_table_contacting, 
                obj_below_table,
                obj_in_boundaries)
        return valid