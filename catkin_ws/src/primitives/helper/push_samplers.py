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

    def sample(self, state=None, state_full=None, final_trans_to_go=None):
        pointcloud_pts = state[:100]
        prediction = self.filesystem_pub_sub(state)

        # unpack from NN prediction
        ind = np.random.randint(prediction['trans_predictions'].shape[0])
        ind_contact = np.random.randint(5)
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

        contact_r = prediction['palm_predictions'][ind, ind_contact, :7]
        contact_l = prediction['palm_predictions'][ind, ind_contact, 7:]

        contact_r[:3] += np.mean(pointcloud_pts, axis=0)
        contact_l[:3] += np.mean(pointcloud_pts, axis=0)

        prediction = {}
        if final_trans_to_go is None:
            prediction['transformation'] = pred_trans
        else:
            prediction['transformation'] = final_trans_to_go
        prediction['palms'] = contact_r
        prediction['mask'] = pred_mask
        return prediction