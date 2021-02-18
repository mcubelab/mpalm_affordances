import os, os.path as osp
import sys
import numpy as np
import random
from collections import OrderedDict
import torch
from scipy.spatial.transform import Rotation as R

sys.path.append('..')
from skeleton_utils.skeleton_globals import SOS_token, EOS_token, PAD_token


def sample_data(data_dir):
    fnames = os.listdir(data_dir)
    fname = random.sample(fnames, 1)[0]

    data = np.load(osp.join(data_dir, fname))
    return data


def process_observations(o, o_):
    o = o[::int(o.size)]


def prepare_sequence_tokens(seq, to_idx):
    idxs = [to_idx[tok] for tok in seq]
    return torch.Tensor(idxs).long()


def to_onehot(x, n):
    if x.dim() < 2:
        x = x[:, None]
    s = x.size()
    y_onehot = torch.FloatTensor(s[0], n).to(x.device)

    y_onehot.zero_()
    y_onehot.scatter_(1, x, 1)
    return y_onehot


def to_onehot3d(x, n):
    s = x.size()
    y_onehot = torch.FloatTensor(s[0], s[1], n).to(x.device)
    y_onehot.zero_()
    y_onehot.scatter_(2, x, 1)
    return y_onehot


def process_pointcloud_sequence_batch(pcd):
    pcd = pcd[:, ::int(pcd.size(2)/100)][:, :, :100]
    o_mean, o_std = pcd.mean(2)[:, :, None, :], pcd.std(2)[:, :, None, :]
    pcd = (pcd - o_mean) / o_std
    s = pcd.size()
    pcd = torch.cat((pcd, o_mean.repeat((1, 1, s[2], 1))), dim=-1)
    return pcd


def process_pointcloud_batch(pcd):
    pcd = pcd[:, ::int(pcd.size(1)/100)][:, :100]
    o_mean, o_std = pcd.mean(1)[:, None, :], pcd.std(1)[:, None, :]
    pcd = (pcd - o_mean) / o_std
    s = pcd.size()
    pcd = torch.cat((pcd, o_mean.repeat((1, s[1], 1))), dim=-1)
    return pcd

def state_dict_to_cpu(state_dict):
    sd_cpu = OrderedDict() 
    for key, val in state_dict.items():
        sd_cpu[key] = val.to(torch.device('cpu'))
    return sd_cpu


def apply_transformation_to_pcd(point_cloud, transform_pose):
    """
    Function to apply a rigid trasnformation to a point cloud by
    getting the homogeneous transformation representation of the
    transformation and dealing with the extra rows of zeros that
    must be added to the point cloud

    Args:
        point_cloud (np.ndarray): N x 3 point cloud
        transform_pose (np.ndarray): 6D pose, expressed as [x, y, z, qx, qy, qz, qw]

    Returns:
        np.ndarray: N x 3 point cloud with transformation applied
    """
    pcd_h = np.concatenate((point_cloud, np.ones((point_cloud.shape[0], 1))), axis=1)
    transform_mat = np.eye(4)
    transform_mat[:-1, -1] = transform_pose[:3]
    transform_mat[:-1, :-1] = R.from_quat(transform_pose).as_matrix()
    transformed_pcd_h = np.matmul(transform_mat, pcd_h.T).T
    return transformed_pcd_h[:, :-1]


def apply_transformation_to_pcd_batch(pcd_batch, transform_pose_batch):
    """
    Function to apply a rigid trasnformation to a point cloud by
    getting the homogeneous transformation representation of the
    transformation and dealing with the extra rows of zeros that
    must be added to the point cloud. Applied this to a batch of point clouds
    using a batch of transform poses

    Args:
        point_cloud (np.ndarray): B x N x 3 point clouds
        transform_pose (np.ndarray): B x 7, 6D pose, expressed as [x, y, z, qx, qy, qz, qw]
    
    Returns:
        np.ndarray: B x N x 3 point clouds with transformations applied
    """
    pcd_h = np.concatenate((pcd_batch, np.tile(np.ones(pcd_batch.shape[1])[None, :, None], (pcd_batch.shape[0], 1, 1))), axis=2)
    pcd_h_t = np.transpose(pcd_h, axes=(0, 2, 1))   

    transform_mat = np.tile(np.eye(4), (pcd_batch.shape[0], 1)).reshape(pcd_batch.shape[0], 4, 4)
    transform_mat[:, :-1, :-1] = R.from_quat(transform_pose_batch[:, 3:]).as_matrix()

    transformed_pcd_h_t = np.matmul(transform_mat, pcd_h_t)
    transformed_pcd = np.transpose(transformed_pcd_h_t, axes=(0, 2, 1))[:, :, :-1]
    return transformed_pcd 

class SkillLanguage:
    def __init__(self, name):
        self.name = name
        self.skill2index = {}
        self.skill2count = {}
        self.index2skill = {PAD_token: 'PAD', SOS_token: 'SOS', EOS_token: 'EOS'}
        self.n_skills = 3 
        
    def add_skill_seq(self, seq):
        for skill in seq.split(' '):
            self.add_skill(skill)
        
    def add_skill(self, skill, keep_count=False):
        if skill not in self.skill2index:
            self.skill2index[skill] = self.n_skills
            self.skill2count[skill] = 1
            self.index2skill[self.n_skills] = skill
            self.n_skills += 1
        else:
            if keep_count:
                self.skill2count[skill] += 1
            else:
                pass

    def process_data(self, data):
        for i in range(len(data)):
            outputs = data[i][1]
            for skill in outputs:
                self.add_skill(skill)
            