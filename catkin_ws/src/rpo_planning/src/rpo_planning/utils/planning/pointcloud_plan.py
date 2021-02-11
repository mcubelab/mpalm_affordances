import os, sys
import os.path as osp
import pickle
import numpy as np
import rospy
import trimesh
import open3d
import pcl
import pybullet as p
import copy
import time
from scipy.spatial import ConvexHull
from shapely import geometry
from shapely.geometry import Point
from scipy.spatial import Delaunay
import rospkg

from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest, GetStateValidityResponse

from airobot.utils import common

from rpo_planning.utils import common as util
from rpo_planning.utils.perception import registration as reg
from rpo_planning.utils.contact import correct_grasp_pos, correct_palm_pos_single
from rpo_planning.utils.planning.collision import CollisionBody, is_collision
# TODO: refactor the task_planning stuff
# from rpo_planning.task_planning.objects import Object, CollisionBody
# from rpo_planning.task_planning import objects


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
        planes (list): List of dictionaries, which have the following key value pairs:
            - 'planes' : np.ndarray of [x, y, z] points in the plane
            - 'normals' : np.ndarray of [x, y, z] normal vectors, for each point
            - 'mean_normal' : np.ndarray of [x, y, z] normal vector, which is the average of all the normals
            - 'antipodal_inds' : int, indicating the index in the list of the most likely oposite plane
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

        self.planes = None
        self.antipodal_thresh = 0.01

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

    def set_planes(self, planes):
        """Setter function to set the planes that correspond to a particular point cloud

        Args:
            planes (list): List of np.ndarrays of size N X 3 that contain set of points, where
                each array element in a list is a separately segmented plane 
        """
        # put planes and mean plane normals
        self.planes = []
        for i in range(len(planes)):
            plane_dict = {}
            plane_dict['plane'] = planes[i]['points']
            self.planes.append(plane_dict)

        # use mean plane normals to estimate pairs of antipodal planes, specified by index in the list
        for i in range(len(self.planes)):
            plane = self.planes[i]['plane']
            # estimate plane normals
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(plane)
            pcd.estimate_normals()
            normals = np.asarray(pcd.normals)
            self.planes[i]['normals'] = normals
            mean_normal = np.mean(normals, axis=0)
            self.planes[i]['mean_normal'] = mean_normal/np.linalg.norm(mean_normal)

        for i in range(len(self.planes)):
            # get average normal vector of this plane
            mean_normal = self.planes[i]['mean_normal']
            self.planes[i]['antipodal_inds'] = None
            for j in range(len(self.planes)):
                # don't check with self
                if i == j:
                    continue
                
                # get average normal vector with other planes
                mean_normal_check = self.planes[j]['mean_normal']

                # use dot product threshold to guess if it's an antipodal face, indicate based on index in list
                dot_prod = np.dot(mean_normal, mean_normal_check)
                if np.abs(1 - dot_prod) < self.antipodal_thresh:
                    self.planes[i]['antipodal_inds'] = j
                    break

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
        
        # save skill if it is specified
        if 'skill' in kwargs.keys():
            self.skill = kwargs['skill']

        # transform planes and plane normals, if we have them available
        if state.planes is not None:
            self.planes = []
            for i, plane_dict in enumerate(state.planes):
                new_plane_dict = {}

                new_plane = copy.deepcopy(plane_dict['plane'])
                new_normals = copy.deepcopy(plane_dict['normals'])

                # transform planes
                new_plane_homog = np.ones((new_plane.shape[0], 4))
                new_plane_homog[:, :-1] = new_plane
                new_plane = np.matmul(transformation, new_plane_homog.T).T[:, :-1]
                
                # transform normals
                new_normals = util.transform_vectors(new_normals, util.pose_from_matrix(transformation))
                
                # save new info
                new_plane_dict['plane'] = new_plane
                new_plane_dict['normals'] = new_normals
                new_plane_dict['mean_normal'] = np.mean(new_normals, axis=0)
                new_plane_dict['antipodal_inds'] = plane_dict['antipodal_inds']

                self.planes.append(new_plane_dict)

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

class PointCloudPlaneSegmentation(object):
    def __init__(self):
        self.ksearch = 50
        self.coeff_optimize = True
        self.normal_distance_weight = 0.05
        self.max_iterations = 100
        self.distance_threshold = 0.005

    def segment_pointcloud(self, pointcloud):
        p = pcl.PointCloud(np.asarray(pointcloud, dtype=np.float32))

        seg = p.make_segmenter_normals(ksearch=self.ksearch)
        seg.set_optimize_coefficients(self.coeff_optimize)
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_normal_distance_weight(self.normal_distance_weight)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_max_iterations(self.max_iterations)
        seg.set_distance_threshold(self.distance_threshold)
        inliers, _ = seg.segment()

        # plane_pts = p.to_array()[inliers]
        # return plane_pts
        return inliers  

    def get_pointcloud_planes(self, pointcloud, visualize=False):
        planes = []

        original_pointcloud = copy.deepcopy(pointcloud)
        com_z = np.mean(original_pointcloud, axis=0)[2]
        for _ in range(5):
            inliers = self.segment_pointcloud(pointcloud)
            masked_pts = pointcloud[inliers]
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(masked_pts)
            pcd.estimate_normals()

            masked_pts_z_mean = np.mean(masked_pts, axis=0)[2]
            above_com = masked_pts_z_mean > com_z

            parallel_z = 0
            if masked_pts.shape[0] == 0:
                print('No points found in segmentation, skipping')
                continue
            for _ in range(100):
                pt_ind = np.random.randint(masked_pts.shape[0])
                pt_sampled = masked_pts[pt_ind, :]
                normal_sampled = np.asarray(pcd.normals)[pt_ind, :]

                dot_x = np.abs(np.dot(normal_sampled, [1, 0, 0]))
                dot_y = np.abs(np.dot(normal_sampled, [0, 1, 0]))
                if np.abs(dot_x) < 0.01 and np.abs(dot_y) < 0.01:
                    parallel_z += 1

            # print(parallel_z)
            if not (above_com and parallel_z > 30):
                # don't consider planes that are above the CoM
                plane_dict = {}
                plane_dict['mask'] = inliers
                plane_dict['points'] = masked_pts
                planes.append(plane_dict)

            if visualize:
                from rpo_planning.utils.visualize import PalmVis
                from rpo_planning.config.multistep_eval_cfg import get_multistep_cfg_defaults
                cfg = get_multistep_cfg_defaults()
                # prep visualization tools
                palm_mesh_file = osp.join(os.environ['CODE_BASE'],
                                            cfg.PALM_MESH_FILE)
                table_mesh_file = osp.join(os.environ['CODE_BASE'],
                                            cfg.TABLE_MESH_FILE)
                viz_palms = PalmVis(palm_mesh_file, table_mesh_file, cfg)
                viz_data = {}
                viz_data['contact_world_frame_right'] = util.pose_stamped2np(util.unit_pose())
                viz_data['contact_world_frame_left'] = util.pose_stamped2np(util.unit_pose())
                viz_data['transformation'] = util.pose_stamped2np(util.unit_pose())
                viz_data['object_pointcloud'] = masked_pts
                viz_data['start'] = masked_pts

                scene_pcd = viz_palms.vis_palms_pcd(viz_data, world=True, corr=False, full_path=True, show_mask=False, goal_number=1)
                scene_pcd.show()

            pointcloud = np.delete(pointcloud, inliers, axis=0)
        return planes



DEFAULT_SV_SERVICE = "/check_state_validity"

class StateValidity():
    def __init__(self):
        rospy.loginfo("Initializing stateValidity class")
        self.sv_srv = rospy.ServiceProxy(DEFAULT_SV_SERVICE, GetStateValidity)
        rospy.loginfo("Connecting to State Validity service")
        rospy.wait_for_service("check_state_validity")
        rospy.loginfo("Reached this point")

        if rospy.has_param('/play_motion/approach_planner/planning_groups'):
            list_planning_groups = rospy.get_param('/play_motion/approach_planner/planning_groups')
        else:
            rospy.logwarn("Param '/play_motion/approach_planner/planning_groups' not set. We can't guess controllers")
        rospy.loginfo("Ready for making Validity calls")


    def close_sv(self):
        self.sv_srv.close()


    def get_state_validity(self, robot_state, group_name='both_arms', constraints=None, print_depth=False):
        """Given a RobotState and a group name and an optional Constraints
        return the validity of the State"""
        gsvr = GetStateValidityRequest()
        gsvr.robot_state = robot_state
        gsvr.group_name = group_name
        if constraints != None:
            gsvr.constraints = constraints
        result = self.sv_srv.call(gsvr)

        if (not result.valid):
            contact_depths = []
            for i in range(len(result.contacts)):
                contact_depths.append(result.contacts[i].depth)

            max_depth = max(contact_depths)
            if max_depth < 0.0001:
                return True
            else:
                return False 
    
        return result.valid


class PalmPoseCollisionChecker(object):
    def __init__(self, gripper_name='mpalms_all_coarse.stl', table_name='table_top_collision.stl'):
        # meshes_dir = '/root/catkin_ws/src/config/descriptions/meshes'
        rospack = rospkg.RosPack()
        meshes_dir = osp.join(rospack.get_path('config'), 'descriptions/meshes')
        table_mesh = osp.join(meshes_dir, 'table', table_name)
        gripper_mesh = osp.join(meshes_dir, 'mpalm', gripper_name)

        # TODO: allow this table pose to be configurable upon construction
        self.table = CollisionBody(mesh_name=table_mesh)
        self.table.setCollisionPose(
            util.list2pose_stamped([0.11091, 0.0, 0.0, 0.0, 0.0, -0.7071045443232222, 0.7071090180427968]))

        self.gripper_left = CollisionBody(mesh_name=gripper_mesh)
        self.gripper_right = CollisionBody(mesh_name=gripper_mesh)

        self.tip2wrist_tf = [0.0, 0.071399, -0.14344421, 0.0, 0.0, 0.0, 1.0]
        self.wrist2tip_tf = [0.0, -0.071399, 0.14344421, 0.0, 0.0, 0.0, 1.0]      

    def sample_in_start_goal_collision(self, sample):
        """Function to check if a sample we have obtained during point-cloud planning
        is in collision in either the start or the goal configuration

        Args:
            sample (rpo_planning.utils.planning.pointcloud_plan.PointCloudNode): Node in 
                the search tree which is expected to have a palm pose
                and transformation matrix indicating the subgoal
        
        Returns:
            2-element tuple containing:
            - bool: Start collision status (True if in collision)
            - bool: Goal collision status (True if in collision)
        """
        palms_start = sample.palms
        transformation = sample.transformation
        transformation_pose = util.pose_from_matrix(transformation)

        palm_pose_right_start = util.list2pose_stamped(palms_start[:7])
        palm_pose_right_goal = util.transform_pose(palm_pose_right_start, transformation_pose)

        r_in_collision_start = self.check_palm_poses_collision(palm_pose_right_start)
        r_in_collision_goal = self.check_palm_poses_collision(palm_pose_right_goal)

        l_in_collision_start = False
        l_in_collision_goal = False        
        if palms_start.shape[0] > 7:
            palm_pose_left_start = util.list2pose_stamped(palms_start[7:])
            palm_pose_left_goal = util.transform_pose(palm_pose_left_start, transformation_pose)            

            l_in_collision_start = self.check_palm_poses_collision(palm_pose_left_start)
            l_in_collision_goal = self.check_palm_poses_collision(palm_pose_left_goal)
        
        start_in_collision = r_in_collision_start or l_in_collision_start
        goal_in_collision = r_in_collision_goal or l_in_collision_goal
        return start_in_collision, goal_in_collision

    def check_palm_poses_collision(self, palm_pose):
        """
        Function to check if the palm pose samples obtained are in collision
        in the start and/or goal configuration, with a given transformation
        that specifies the subgoal

        Args:
            palm_pose (PoseStamped): World frame pose of the corresponding end effector. Note, 
                this is the TIP pose (internally in this function we convert them to WRIST pose)

        Returns:
            bool: True if in collision
        """        
        wrist_pose = util.convert_reference_frame(
            pose_source=util.list2pose_stamped(self.tip2wrist_tf),
            pose_frame_target=util.unit_pose(),
            pose_frame_source=palm_pose)
        self.gripper_right.setCollisionPose(
            wrist_pose)
        in_collision = is_collision(
            self.table.collision_object, 
            self.gripper_right.collision_object)
        return in_collision


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


class PlanningFailureModeTracker(object):
    def __init__(self, skeleton):
        self.precondition_infeasibility = []
        self.start_palm_infeasibility = []
        self.start_full_infeasibility = []
        self.goal_palm_infeasibility = []
        self.goal_full_infeasibility = []

        self.path_collision_infeasibility = []
        self.path_kinematic_infeasibility = []
        self.path_full_infeasibility = []

        self.total_samples = 0
        self.skeleton = skeleton
        self.skeleton_samples = {}
        for skill in self.skeleton:
            self.skeleton_samples[skill] = 0

    def increment_total_counts(self, skill):
        self.total_samples += 1
        self.skeleton_samples[skill] += 1

    def update_infeasibility_counts(self, precondition, start_palm, goal_palm, path_full):
        self.precondition_infeasibility.append(precondition)
        self.start_palm_infeasibility.append(start_palm)
        self.goal_palm_infeasibility.append(goal_palm)
        self.path_full_infeasibility.append(path_full)

    def collect_data(self):
        data_dict = {}
        data_dict['precondition_infeasibility'] = self.precondition_infeasibility
        data_dict['start_palm_infeasibility'] = self.start_palm_infeasibility
        data_dict['goal_palm_infeasibility'] = self.goal_palm_infeasibility
        data_dict['path_full_infeasibility'] = self.path_full_infeasibility                                 
        
        data_dict['total_samples'] = self.total_samples
        data_dict['skeleton_samples'] = self.skeleton_samples

        return data_dict

    def save_data(self, fname):
        data_dict = self.collect_data()
        with open(fname, 'wb') as f:
            pickle.dump(data_dict, f)