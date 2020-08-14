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
from eval_utils.visualization_tools import correct_grasp_pos, project_point2plane
from pointcloud_planning_utils import PointCloudNode
from sampler_utils import PubSubSamplerBase
# from planning import grasp_planning_wf

# sys.path.append('/root/training/')

# import os
# import argparse
# import time
# import numpy as np
# import torch
# from torch import nn
# from torch import optim
# from torch.autograd import Variable
# from data_loader import DataLoader
# from model import VAE, GoalVAE
# from util import to_var, save_state, load_net_state, load_seed, load_opt_state

# import scipy.signal as signal

# from sklearn.mixture import GaussianMixture
# from sklearn.manifold import TSNE

# # sys.path.append('/root/training/gat/')
# # from models_vae import GoalVAE, GeomVAE, JointVAE

# sys.path.append('/root/training/gat/')
# # from models_vae import JointVAE
# from joint_model_vae import JointVAE


class PushSamplerVAEPubSub(PubSubSamplerBase):
    def __init__(self, obs_dir, pred_dir, sampler_prefix='push_vae_', pointnet=False):
        super(PushSamplerVAEPubSub, self).__init__(obs_dir, pred_dir, sampler_prefix)
        self.pointnet=pointnet

    def sample(self, state=None, state_full=None, final_trans_to_go=None):
        pointcloud_pts = state[:100]
        prediction = self.filesystem_pub_sub(state)

        # unpack from NN prediction
        ind = np.random.randint(prediction['trans_predictions'].shape[0])
        ind_contact = np.random.randint(35)
        pred_trans_pose = prediction['trans_predictions'][ind, :]
        pred_trans_pos = pred_trans_pose[:3]
        pred_trans_ori = pred_trans_pose[3:]/np.linalg.norm(pred_trans_pose[3:])
        pred_trans_pose = pred_trans_pos.tolist() + pred_trans_ori.tolist()
        pred_trans = np.eye(4)
        pred_trans[:-1, :-1] = common.quat2rot(pred_trans_ori)
        pred_trans[:-1, -1] = pred_trans_pos

        mask = prediction['mask_predictions'][ind, :]
        top_inds = np.argsort(mask)[::-1]
        pred_mask = np.zeros((mask.shape[0]), dtype=bool)
        pred_mask[top_inds[:15]] = True

        # fix palm predictions (via CoM)
        if self.pointnet:
            contact_r = prediction['palm_predictions'][ind, :7]
            contact_l = prediction['palm_predictions'][ind, 7:]
        else:
            contact_r = prediction['palm_predictions'][ind, ind_contact, :7]
            contact_l = prediction['palm_predictions'][ind, ind_contact, 7:]

        contact_r[:3] += np.mean(pointcloud_pts, axis=0)
        contact_l[:3] += np.mean(pointcloud_pts, axis=0)

        prediction = {}
        if final_trans_to_go is None:
            prediction['transformation'] = pred_trans
            prediction['transformation'][2, -1] = 0.0
        else:
            prediction['transformation'] = final_trans_to_go
        # prediction['transformation'][2, -1] = 0.0            
        prediction['palms'] = contact_r
        prediction['palms'][2] = np.clip(prediction['palms'][2], 0.05, None)
        prediction['mask'] = pred_mask
        return prediction


class PushSamplerBasic(object):
    def __init__(self, default_target):
        self.default_target = default_target
        self.sample_timeout = 5.0
        self.sample_limit = 100
        self.planes = []

    def get_palms(self, state, state_full=None):
        # check if full pointcloud available, if not use sparse pointcloud
        pcd_pts = state if state_full is None else state_full

        # compute pointcloud normals
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pcd_pts)
        pcd.estimate_normals()

        pt_samples = []
        dot_samples = []
        normal_samples = []

        # search for a point on the pointcloud with normal NOT pointing up
        sample_i = 0
        sampling_start = time.time()
        while True:
            sample_i += 1
            t = time.time() - sampling_start

            pt_ind = np.random.randint(pcd_pts.shape[0])
            pt_sampled = pcd_pts[pt_ind, :]

            normal_sampled = np.asarray(pcd.normals)[pt_ind, :]

            dot_x = np.abs(np.dot(normal_sampled, [1, 0, 0]))
            dot_y = np.abs(np.dot(normal_sampled, [0, 1, 0]))
            dot_z = np.abs(np.dot(normal_sampled, [0, 0, 1]))

            dot_samples.append(dot_z)
            pt_samples.append(pt_sampled)
            normal_samples.append(normal_sampled)

            orthogonal_z = np.abs(dot_z) < 0.01

            if orthogonal_z:
                break

            if t > self.sample_timeout or sample_i > self.sample_limit:
                dots = np.asarray(dot_samples)
                pts = np.asarray(pt_samples)
                normals = np.asarray(normal_sampled)

                # sort by dot_x
                z_sort_inds = np.argsort(dots)
                dots_z_sorted = dots[z_sort_inds]
                pts_z_sorted = pts[z_sort_inds, :]
                normal_z_sorted = normals[z_sort_inds, :]

                # pick this point
                pt_sampled = pts_z_sorted[0, :]
                normal_sampled = normal_z_sorted[0, :]
                break

            # print(dot_x, dot_y, above_com)
            time.sleep(0.01)

        # get both normal directions
        normal_vec_1 = normal_sampled
        normal_vec_2 = -normal_sampled

        # get endpoint based on normals
        endpoint_1 = pt_sampled + normal_vec_1
        endpoint_2 = pt_sampled + normal_vec_2

        # get points interpolated between endpoints and sampled point
        points_along_1_0 = np.linspace(pt_sampled, endpoint_1, 2000)
        points_along_1_1 = np.linspace(endpoint_1, pt_sampled, 2000)
        points_along_2_0 = np.linspace(pt_sampled, endpoint_2, 2000)
        points_along_2_1 = np.linspace(endpoint_2, pt_sampled, 2000)

        one_points = [
            points_along_1_0,
            points_along_1_1
        ]
        two_points = [
            points_along_2_0,
            points_along_2_1
        ]
        one_norms = [
            np.linalg.norm(points_along_1_0[0] - points_along_1_0[-1]),
            np.linalg.norm(points_along_1_1[0] - points_along_1_1[-1])
        ]
        two_norms = [
            np.linalg.norm(points_along_2_0[0] - points_along_2_0[-1]),
            np.linalg.norm(points_along_2_1[0] - points_along_2_1[-1])
        ]
        points_along_1 = one_points[np.argmax(one_norms)]
        points_along_2 = two_points[np.argmax(two_norms)]

        points = {}
        points['1'] = points_along_1
        points['2'] = points_along_2

        dists = {}
        dists['1'] = []
        dists['2'] = []

        inds = {}
        inds['1'] = []
        inds['2'] = []

        # go through all points to find the one with the smallest distance to the pointcloud
        kdtree = open3d.geometry.KDTreeFlann(pcd)
        for key in points.keys():
            for i in range(points[key].shape[0]):
                pos = points[key][i, :]
                nearest_pt_ind = kdtree.search_knn_vector_3d(pos, 1)[1][0]

                dist = np.asarray(pcd.points)[nearest_pt_ind] - pos

                inds[key].append((i, nearest_pt_ind))
                dists[key].append(dist.dot(dist))

        opposite_pt_candidates = []
        for key in points.keys():
            min_ind = np.argmin(dists[key])
            inds_sorted = np.argsort(dists[key])
            # for i, min_ind in enumerate(inds_sorted[:10]):
            min_ind = inds_sorted[-1]
            min_dist = dists[key][min_ind]
            min_point_ind = inds[key][min_ind][0]
            min_point_pcd_ind = inds[key][min_ind][1]
            nearest_pt_world = points[key][min_point_ind]
            nearest_pt_pcd_world = np.asarray(pcd.points)[min_point_pcd_ind]
            # opposite_pt_candidates.append((nearest_pt_world, min_dist))
            opposite_pt_candidates.append((nearest_pt_pcd_world, min_dist))

        # pick which one based on smaller distance
        dist_vals = [opposite_pt_candidates[0][1], opposite_pt_candidates[1][1]]
        opp_pt_sampled = opposite_pt_candidates[np.argmin(dist_vals)][0]

        # guess which one should be right and left based on y values
        both_pts_sampled = [pt_sampled, opp_pt_sampled]
        y_vals = [pt_sampled[1], opp_pt_sampled[1]]
        r_pt_ind, l_pt_ind = np.argmin(y_vals), np.argmax(y_vals)

        tip_contact_r_world = both_pts_sampled[r_pt_ind]
        tip_contact_l_world = both_pts_sampled[l_pt_ind]

        # sample a random 3D vector to get second points
        rand_dir = np.random.random(3) * 2.0 - 1.0
        second_pt_dir = rand_dir/np.linalg.norm(rand_dir)
        tip_contact_r2_world = tip_contact_r_world + second_pt_dir
        tip_contact_l2_world = tip_contact_l_world + second_pt_dir

        # get palm y vector
        nearest_pt_ind_r = kdtree.search_knn_vector_3d(tip_contact_r_world, 1)[1][0]
        nearest_pt_ind_l = kdtree.search_knn_vector_3d(tip_contact_l_world, 1)[1][0]

        normal_y_r = pcd.normals[nearest_pt_ind_r]
        normal_y_l = pcd.normals[nearest_pt_ind_l]

        # get palm z vector
        # palm_z_r = (tip_contact_r2_world - tip_contact_r_world)/np.linalg.norm((tip_contact_r2_world - tip_contact_r_world))
        # palm_z_l = (tip_contact_l2_world - tip_contact_l_world)/np.linalg.norm((tip_contact_l2_world - tip_contact_l_world))

        tip_contact_r2_world = project_point2plane(
            tip_contact_r2_world,
            normal_y_r,
            [tip_contact_r_world])[0]
        tip_contact_l2_world = project_point2plane(
            tip_contact_l2_world,
            normal_y_l,
            [tip_contact_l_world])[0]

        palm_z_r = (tip_contact_r2_world - tip_contact_r_world)/np.linalg.norm((tip_contact_r2_world - tip_contact_r_world))
        palm_z_l = (tip_contact_l2_world - tip_contact_l_world)/np.linalg.norm((tip_contact_l2_world - tip_contact_l_world))

        if np.dot(normal_y_l, palm_z_l) > 0:
            normal_y_l = -normal_y_l

        x_r, y_r, z_r = np.cross(normal_y_r, palm_z_r), normal_y_r, palm_z_r
        x_l, y_l, z_l = np.cross(normal_y_l, palm_z_l), normal_y_l, palm_z_l

        com = np.mean(pcd_pts, axis=0)
        com_r_vec = tip_contact_r_world - com
        if np.dot(com_r_vec, y_r) < 0.0:
            tmp = tip_contact_r_world
            tip_contact_r_world = tip_contact_l_world
            tip_contact_l_world = tmp

        tip_contact_r = util.pose_from_vectors(
            x_r, y_r, z_r, tip_contact_r_world
        )
        # tip_contact_l = util.pose_from_vectors(
        #     x_l, y_l, z_l, tip_contact_l_world
        # )
        tip_contact_l = util.pose_from_vectors(
            -x_r, -y_r, z_r, tip_contact_l_world
        )

        tip_contact_r_np = util.pose_stamped2np(tip_contact_r)
        tip_contact_l_np = util.pose_stamped2np(tip_contact_l)
        # embed()

        # print(pt_sampled, opp_pt_sampled)
        # from eval_utils.visualization_tools import PalmVis
        # from multistep_planning_eval_cfg import get_cfg_defaults
        # cfg = get_cfg_defaults()
        # # prep visualization tools
        # palm_mesh_file = osp.join(os.environ['CODE_BASE'],
        #                             cfg.PALM_MESH_FILE)
        # table_mesh_file = osp.join(os.environ['CODE_BASE'],
        #                             cfg.TABLE_MESH_FILE)
        # viz_palms = PalmVis(palm_mesh_file, table_mesh_file, cfg)
        # viz_data = {}
        # viz_data['contact_world_frame_right'] = tip_contact_r_np
        # viz_data['contact_world_frame_left'] = tip_contact_l_np
        # viz_data['transformation'] = util.pose_stamped2np(util.unit_pose())
        # viz_data['object_pointcloud'] = pcd_pts
        # viz_data['start'] = pcd_pts

        # scene_pcd = viz_palms.vis_palms_pcd(viz_data, world=True, corr=False, full_path=True, show_mask=False, goal_number=1)
        # scene_pcd.show()

        # embed()

        return np.concatenate([tip_contact_r_np, tip_contact_l_np])

    def get_pointcloud_planes(self, pointcloud):
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

            # from eval_utils.visualization_tools import PalmVis
            # from multistep_planning_eval_cfg import get_cfg_defaults
            # cfg = get_cfg_defaults()
            # # prep visualization tools
            # palm_mesh_file = osp.join(os.environ['CODE_BASE'],
            #                             cfg.PALM_MESH_FILE)
            # table_mesh_file = osp.join(os.environ['CODE_BASE'],
            #                             cfg.TABLE_MESH_FILE)
            # viz_palms = PalmVis(palm_mesh_file, table_mesh_file, cfg)
            # viz_data = {}
            # viz_data['contact_world_frame_right'] = util.pose_stamped2np(util.unit_pose())
            # viz_data['contact_world_frame_left'] = util.pose_stamped2np(util.unit_pose())
            # viz_data['transformation'] = util.pose_stamped2np(util.unit_pose())
            # viz_data['object_pointcloud'] = masked_pts
            # viz_data['start'] = masked_pts

            # scene_pcd = viz_palms.vis_palms_pcd(viz_data, world=True, corr=False, full_path=True, show_mask=False, goal_number=1)
            # scene_pcd.show()

            pointcloud = np.delete(pointcloud, inliers, axis=0)
        return planes

    def segment_pointcloud(self, pointcloud):
        p = pcl.PointCloud(np.asarray(pointcloud, dtype=np.float32))

        seg = p.make_segmenter_normals(ksearch=50)
        seg.set_optimize_coefficients(True)
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_normal_distance_weight(0.05)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_max_iterations(100)
        seg.set_distance_threshold(0.005)
        inliers, _ = seg.segment()

        # plane_pts = p.to_array()[inliers]
        # return plane_pts
        return inliers

    def sample(self, state=None, state_full=None, target=None, final_trans_to_go=None):
        pointcloud_pts = state if state_full is None else state_full

        # get mask, based on plane fitting and other heuristics
        planes = self.get_pointcloud_planes(pointcloud_pts)
        # pred_mask = self.segment_pointcloud(pointcloud_pts)
        # pred_mask = planes[np.random.randint(5)]['mask']
        pred_points_masked = planes[np.random.randint(len(planes))]['points']

        prediction = {}
        prediction['transformation'] = self.get_transformation(
            pointcloud_pts,
            pred_points_masked,
            target,
            final_trans_to_go)
        prediction['palms'] = self.get_palms(state, state_full)
        prediction['mask'] = pred_points_masked
        return prediction