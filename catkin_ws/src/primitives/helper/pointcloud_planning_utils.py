import os, sys
import os.path as osp
import pickle
import numpy as np

import trimesh
import open3d
import pcl
import pybullet as p

import copy
import time
from IPython import embed

from yacs.config import CfgNode as CN
from airobot.utils import common

sys.path.append('/root/catkin_ws/src/primitives/')
# from helper import util2 as util
# from helper import registration as reg
import util2 as util
import registration as reg
from closed_loop_experiments_cfg import get_cfg_defaults
from eval_utils.visualization_tools import correct_grasp_pos, correct_palm_pos_single, project_point2plane


class PointCloudNode(object):
    """Class for representing object configurations based on point clouds,
    nodes in a search tree where edges represent rigid transformations between
    point cloud configurations, and bi-manual end effector poses that make contact
    with the object to execute the transformation edges.

    Attributes:
        parent (int): Index used to refer to previous state visited in planning
        pointcloud (np.ndarray): N X 3 pointcloud AFTER being transformed (downsampled # of points)
        pointcloud_full (np.ndarray): N' X 3 pointcloud AFTER being transformed (full # of points)
        pointcloud_mask (np.ndarray): N X 1 array of binary labels indicating which points are part
            of subgoal mask (i.e. for grasping)
        transformation (np.ndarray): 4 X 4 homogenous transformation matrix that is applied to initial
            pointcloud and transforms it into self.pointcloud.
        transformation_to_go (np.ndarray): 4 X 4 homogenous transformation matrix representing what
            transformation must be executed to satisfy the global desired transformation task specification
        transformation_so_far (np.ndarray): 4 X 4 homogenous transformation matrix that tracks the composed
            sequence of transformations that have been used to lead up to this point cloud
        palms (np.ndarray): TODO
        palms_corrected (np.ndarray): TODO
        palms_raw (np.ndarray): TODO
    """
    def __init__(self):
        self.parent = None
        self.pointcloud = None
        self.pointcloud_full = None
        self.pointcloud_mask = None
        self.transformation = None
        self.transformation_to_go = np.eye(4)
        self.transformation_so_far = None
        self.palms = None
        self.palms_corrected = None
        self.palms_raw = None
        self.skill = None

    def set_pointcloud(self, pcd, pcd_full=None, pcd_mask=None):
        """Setter function to directly set the transformed point cloud

        Args:
            pcd (np.ndarray): N X 3 transformed point cloud
            pcd_full (np.ndarray, optional): N' X 3 transformed pointcloud (full # of points). 
                Defaults to None.
            pcd_mask (np.ndarray, optional): N X 1 array of binary subgoal mask labels. Defaults to None.
        """
        self.pointcloud = pcd
        self.pointcloud_full = pcd_full
        self.pointcloud_mask = pcd_mask

    def set_trans_to_go(self, trans):
        """Setter function to set the transformation to go from this node on,
        if planning to solve some task specified by a desired global transformation

        Args:
            trans ([type]): [description]
        """
        self.transformation_to_go = trans

    def init_state(self, state, transformation, *args, **kwargs):
        """Initialize the pointcloud and transformation attributes based on some 
        initial node (state) and sampled transformation.

        Args:
            state (PointCloudNode): Node containing the initial pointcloud, to be transformed
            transformation (np.ndarray): 4 X 4 homogenous transformation matrix, representing
                the transformation to be applied to the start pointcloud
        """
        # compute the pointcloud based on the previous pointcloud and specified trans
        pcd_homog = np.ones((state.pointcloud.shape[0], 4))
        pcd_homog[:, :-1] = state.pointcloud
        self.pointcloud = np.matmul(transformation, pcd_homog.T).T[:, :-1]

        # do the same for the full pointcloud, if it's there
        if state.pointcloud_full is not None:
            pcd_full_homog = np.ones((state.pointcloud_full.shape[0], 4))
            pcd_full_homog[:, :-1] = state.pointcloud_full
            self.pointcloud_full = np.matmul(transformation, pcd_full_homog.T).T[:, :-1]

        # node's one step transformation
        self.transformation = transformation

        # transformation to go based on previous transformation to go
        self.transformation_to_go = np.matmul(state.transformation_to_go, np.linalg.inv(transformation))

        # transformation so far, accounting for if this is the first step,
        # and parent node has no transformation so far
        if state.transformation is not None:
            self.transformation_so_far = np.matmul(transformation, state.transformation_so_far)
        else:
            self.transformation_so_far = transformation
        
        if 'skill' in kwargs.keys():
            self.skill = kwargs['skill']

    def init_palms(self, palms, correction=False, prev_pointcloud=None, dual=True):
        """Initilize the palm attributes based on some sampled palm poses. Also implements
        heuristic corrections of the palm samples based on the transformed pointcloud, to make it
        more likely the palms will correctly contact the surface of the object.

        Args:
            palms (np.ndarray): 1 x 14
            correction (bool, optional): Whether or not to apply heuristic correction. Correction is
                based on searching along the vector normal to the intially specified palm plane for the
                closest point in the pointcloud, so that the position component of the palm pose can be
                refined while maintaining the same orientation. Must have access to the initial pointcloud
                used to sample the palm poses. Defaults to False.
            prev_pointcloud (np.ndarray, optional): N X 3, intial pointcloud used to sample the palms. Used
                here for palm pose refinement. Defaults to None.
            dual (bool, optional): True if this node corresponds to a two-arm skill, else False.
                Defaults to True.
        """
        if correction and prev_pointcloud is not None:
            palms_raw = palms
            palms_positions = {}
            palms_positions['right'] = palms_raw[:3]
            palms_positions['left'] = palms_raw[7:7+3]
            pcd_pts = prev_pointcloud
            if dual:
                # palms_positions_corr = correct_grasp_pos(palms_positions,
                #                                         pcd_pts)
                # palm_right_corr = np.hstack([
                #     palms_positions_corr['right'],
                #     palms_raw[3:7]])
                # palm_left_corr = np.hstack([
                #     palms_positions_corr['left'],
                #     palms_raw[7+3:]
                # ])
                # self.palms_corrected = np.hstack([palm_right_corr, palm_left_corr])

                r_positions_corr = correct_palm_pos_single(palms_raw[:7], pcd_pts)[:3]
                l_positions_corr = correct_palm_pos_single(palms_raw[7:], pcd_pts)[:3]
                palm_right_corr = np.hstack([r_positions_corr, palms_raw[3:7]])
                palm_left_corr = np.hstack([l_positions_corr, palms_raw[7+3:]])
                self.palms_corrected = np.hstack([palm_right_corr, palm_left_corr])
            else:
                r_positions_corr = correct_palm_pos_single(palms_raw[:7], pcd_pts)[:3]
                # l_positions_corr = correct_palm_pos_single(palms_raw[:7], pcd_pts)[:3]
                # palms_positions_corr = {}
                # palms_positions_corr['right'] = r_positions_corr
                # palms_positions_corr['left'] = l_positions_corr
                # self.palms_corrected = np.hstack([palms_positions_corr['right'], palms_raw[3:7]])
                self.palms_corrected = np.hstack([r_positions_corr, palms_raw[3:7]])

            self.palms_raw = palms_raw
            self.palms = self.palms_corrected
        else:
            self.palms = palms
            self.palms_raw = palms

        # check if hands got flipped like a dummy by checking y coordinate in world frame
        if self.palms.shape[0] > 7:
            if self.palms[1] > self.palms[1+7]:
                tmp_l = copy.deepcopy(self.palms[7:])
                self.palms[7:] = copy.deepcopy(self.palms[:7])
                self.palms[:7] = tmp_l


class PointCloudNodeForward(PointCloudNode):
    def __init__(self):
        super(PointCloudNodeForward, self).__init__()
        # self.obs_dir = osp.join(os.environ['CODE_BASE'], cfg.PREDICTION_DIR)
        # self.pred_dir = osp.join(os.environ['CODE_BASE'], cfg.OBSERVATION_DIR)
        self.obs_dir = osp.join(os.environ['CODE_BASE'], '/tmp/observations')
        self.pred_dir = osp.join(os.environ['CODE_BASE'], '/tmp/predictions')
        self.sampler_prefix = 'forward_trans_'
        self.samples_count = 0

    def get_forward_prediction(self, state, transformation, palms):
        # implement pub/sub logic using the filesystem
        self.samples_count += 1
        pointcloud_pts = state.pointcloud[:100]
        transformation = util.pose_stamped2np(util.pose_from_matrix(transformation))

        obs_fname = osp.join(
            self.obs_dir,
            self.sampler_prefix + str(self.samples_count) + '.npz')
        np.savez(
            obs_fname,
            pointcloud_pts=pointcloud_pts,
            transformation=transformation,
            palms=palms
        )

        # wait for return
        got_file = False
        pred_fname = osp.join(
            self.pred_dir,
            self.sampler_prefix + str(self.samples_count) + '.npz')
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
        os.remove(pred_fname)
        return prediction

    def init_state(self, state, transformation, palms):
        # compute the pointcloud based on the previous pointcloud and specified trans
        pcd_homog = np.ones((state.pointcloud.shape[0], 4))
        pcd_homog[:, :-1] = state.pointcloud

        # get new pointcloud from NN forward prediction
        prediction = self.get_forward_prediction(state, transformation, palms)

        # from IPython import embed
        # embed()

        # unpack prediction and do stuff with it
        transformation = util.matrix_from_pose(util.list2pose_stamped(prediction['trans_predictions'][0]))

        self.pointcloud = np.matmul(transformation, pcd_homog.T).T[:, :-1]

        # do the same for the full pointcloud, if it's there
        if state.pointcloud_full is not None:
            pcd_full_homog = np.ones((state.pointcloud_full.shape[0], 4))
            pcd_full_homog[:, :-1] = state.pointcloud_full
            self.pointcloud_full = np.matmul(transformation, pcd_full_homog.T).T[:, :-1]

        # node's one step transformation
        self.transformation = transformation

        # transformation to go based on previous transformation to go
        self.transformation_to_go = np.matmul(state.transformation_to_go, np.linalg.inv(transformation))

        # transformation so far, accounting for if this is the first step,
        # and parent node has no transformation so far
        if state.transformation is not None:
            self.transformation_so_far = np.matmul(transformation, state.transformation_so_far)
        else:
            self.transformation_so_far = transformation


# class PointCloudNodeForwardLatent(PointCloudNode):
#     def __init__(self):
#         super(PointCloudNodeForward, self).__init__(self)

#     def get_forward_prediction(self, state, transformation, palms):
#         # implement pub/sub logic using the filesystem

#     def init_state(self, state, transformation, palms):
#         # compute the pointcloud based on the previous pointcloud and specified trans
#         pcd_homog = np.ones((state.pointcloud.shape[0], 4))
#         pcd_homog[:, :-1] = state.pointcloud

#         # get new pointcloud from NN forward prediction
#         prediction = self.get_forward_prediction(state, transformation, palms)

#         # unpack prediction and do stuff with it

#         self.pointcloud = np.matmul(transformation, pcd_homog.T).T[:, :-1]

#         # do the same for the full pointcloud, if it's there
#         if state.pointcloud_full is not None:
#             pcd_full_homog = np.ones((state.pointcloud_full.shape[0], 4))
#             pcd_full_homog[:, :-1] = state.pointcloud_full
#             self.pointcloud_full = np.matmul(transformation, pcd_full_homog.T).T[:, :-1]

#         # node's one step transformation
#         self.transformation = transformation

#         # transformation to go based on previous transformation to go
#         self.transformation_to_go = np.matmul(state.transformation_to_go, np.linalg.inv(transformation))

#         # transformation so far, accounting for if this is the first step,
#         # and parent node has no transformation so far
#         if state.transformation is not None:
#             self.transformation_so_far = np.matmul(transformation, state.transformation_so_far)
#         else:
#             self.transformation_so_far = transformation

from scipy.spatial import ConvexHull
from shapely import geometry
from shapely.geometry import Point
from scipy.spatial import Delaunay

class PointCloudCollisionChecker(object):
    """Class for collision checking between different segmented pointclouds,
    based on convex hull inclusion. 

    Args:
        collision_pcds (list): Each element in list is N X 3 np.ndarray, representing
            the pointcloud of the objects in the scene that should be considered as
            collision obstacles.
    """    
    def __init__(self, collision_pcds):
        # all the pointclouds that we consider to be "collision objects"
        self.collision_pcds = collision_pcds

    def open3d_pcd_init(self, points):
        """Helper function to initialize an open3d pointcloud object from a numpy array

        Args:
            points (np.ndarray): N X 3 pointcloud array

        Returns:
            open3d.geometry.PointCloud: open3d Pointcloud
        """
        pcd3d = open3d.geometry.PointCloud()
        pcd3d.points = open3d.utility.Vector3dVector(points)
        return pcd3d

    def check_2d(self, obj_pcd):
        """Check if obj_pcd is inside the 2D projection of our collision geometry

        Args:
            obj_pcd (np.ndarray): N X 3 pointcloud array, to be checked for collisions
                with respect to other collision object pointclouds
        """
        valid = True
        if self.collision_pcds is None:
            return True
        com_2d = np.mean(obj_pcd, axis=0)[:-1]
        for pcd in self.collision_pcds:
            # hull_2d = ConvexHull(pcd[:, :-1])
            # hull_poly_2d = geometry.Polygon(hull_2d.points[hull_2d.vertices])
            com = np.mean(pcd, axis=0)
            pcd = pcd - com
            pcd = pcd * 1.1
            pcd = pcd + com

            res = self.in_poly_hull_multi(pcd, obj_pcd)
            valid = valid and (np.asarray(res) == False).all()
            if not valid:
                return False

        return valid

    def in_poly_hull_multi(self, poly, points):
        """Function to check if any points in a 3D pointcloud fall inside of
        the convex hull of a set of 3D points. A convex hull for the collision
        geometry is computed, and each point in the manipulated object is checked
        to see if it is inside the convex hull or not. This is done by adding the
        point to be checked to the set of points, recomputing the convex hull,
        and checking to see if the convex hull has changed (if it has not changed, then
        the point is inside the convex hull).

        Args:
            poly (np.ndarray): N X 3 pointcloud array of points in collision object
            points (np.ndarray): N X 3 pointcloud array of points in manipulated object

        Returns:
            list: Each element is a boolean, for each of the points that were checked.
                The value is True if the point is in collision, and False if not. Check
                if all values are False to see if the full object is collision-free
        """
        hull = ConvexHull(poly)
        res = []
        for p in points:
            new_hull = ConvexHull(np.concatenate((poly, [p])))
            res.append(np.array_equal(new_hull.vertices, hull.vertices))
        return res

    def in_poly_hull_single(self, poly, point):
        """Function to check if a 3D point falls inside of
        the convex hull of a set of 3D points. A convex hull for the collision
        geometry is computed, and the point is checked
        to see if it is inside the convex hull or not. This is done by adding the
        point to be checked to the set of points, recomputing the convex hull,
        and checking to see if the convex hull has changed (if it has not changed, then
        the point is inside the convex hull).

        Args:
            poly (np.ndarray): N X 3 pointcloud array of points in collision object
            point (np.ndarray): 1 X 3 point to be checked

        Returns:
            bool: True if point is in collision, False if collision-free
        """        
        hull = ConvexHull(poly)
        new_hull = ConvexHull(np.concatenate((poly, [point])))
        return np.array_equal(new_hull.vertices, hull.vertices)

    def show_collision_pcds(self, extra_pcds=None):
        """Visualize the pointcloud collision geometry that is being used.

        Args:
            extra_pcds (np.ndarray, optional): Other pointclouds to include in the
                visualization that are not used as collision objects. Defaults to None.
        """
        if self.collision_pcds is None:
            raise ValueError('no collision pointclouds found!')
        pcds = []
        for pcd in self.collision_pcds:
            # pcd3d = self.open3d_pcd_init(pcd)
            pcd3d = open3d.geometry.PointCloud()
            pcd3d.points = open3d.utility.Vector3dVector(pcd)
            pcds.append(pcd3d)
        if extra_pcds is not None:
            for pcd in extra_pcds:
                # pcd3d = self.open3d_pcd_init(pcd)
                pcd3d = open3d.geometry.PointCloud()
                pcd3d.points = open3d.utility.Vector3dVector(pcd)
                pcds.append(pcd3d)
        mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.6, origin=[0, 0, 0])
        pcds.append(mesh_frame)
        open3d.visualization.draw_geometries(pcds)

    def show_collision_check_sample(self, obj_pcd):
        """Visualize the collision geometry with a queried object that is being
        checked for collision

        Args:
            obj_pcd (np.ndarray): Manipulated object pointcloud, to be checked for collisions
        """
        print('Valid: ' + str(self.check_2d(obj_pcd)))
        self.show_collision_pcds(extra_pcds=[obj_pcd])
