import os, os.path as osp
import copy
import time
import numpy as np
import sys
import lcm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm
import torch.nn as nn

from airobot import set_log_level, log_debug, log_info, log_warn, log_critical
import rospkg
rospack = rospkg.RosPack()
sys.path.append(osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/lcm_types'))
from rpo_lcm import (
    point_t, quaternion_t, pose_stamped_t, pose_t, 
    point_cloud_t, skill_param_t, skill_param_array_t,
    dual_pose_stamped_t)
from rpo_planning.utils import common as util
from rpo_planning.utils import lcm_utils


class ModelPredictorLCM():
    def __init__(self, model, FLAGS, model_path, in_msg_names, out_msg_names):
        self.model = model
        self.FLAGS = FLAGS
        self.model_path = model_path

        self.in_msg_names = in_msg_names
        self.out_msg_names = out_msg_names

        self.lc = lcm.LCM()
        self.subs = []
        for i, name in enumerate(in_msg_names):
            sub = self.lc.subscribe(name, self.sub_handler)
            self.subs.append(sub)

    def sub_handler(self, channel, data):
        """Callback for receiving data via LCM from the
        environment observation. Stores the incoming data
        in an internal attribute variable

        Args:
            channel ([type]): [description]
            data ([type]): [description]
        """
        msg = point_cloud_t.decode(data)
        points = msg.points

        log_debug('Model predictor received message from LCM channel: %s' % channel)

        point_list = []
        num_pts = msg.num_points
        for i in range(num_pts):
            pt = [
                points[i].x,
                points[i].y,
                points[i].z
            ]
            point_list.append(pt)
        self.observation = {}
        self.observation['pointcloud_pts'] = np.asarray(point_list)
        self.observation['pub_msg_name'] = self.out_msg_names[self.in_msg_names.index(channel)]
        self.received_point_data = True

    def predict_params(self):
        model = self.model
        model_path = self.model_path
        FLAGS = self.FLAGS
        
        if FLAGS.cuda:
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")

        vae = model.eval().to(dev)

        with torch.no_grad():
            self.received_point_data = False
            while not self.received_point_data:
                self.lc.handle()

            observation = self.observation
            start = observation['pointcloud_pts'][:100]

            start_mean = np.mean(start, axis=0, keepdims=True)
            start_normalized = (start - start_mean)
            start_mean = np.tile(start_mean, (start.shape[0], 1))

            start = np.concatenate([start_normalized, start_mean], axis=1)
            kd_idx = np.arange(100, dtype=np.int64)

            start = torch.from_numpy(start)
            kd_idx = torch.from_numpy(kd_idx)

            joint_keypoint = start
            joint_keypoint = joint_keypoint[None, :, :]
            kd_idx = kd_idx[None, :]

            joint_keypoint = joint_keypoint.float().to(dev)
            kd_idx = kd_idx.long().to(dev)

            palm_predictions = []
            mask_predictions = []
            trans_predictions = []

            num_samples = 3
            for _ in range(num_samples):
                palm_repeat = []
                z = torch.randn(1, FLAGS.latent_dimension).to(dev)
                # recon_mu, ex_wt = model.decode(z, joint_keypoint, full_graph=(not FLAGS.local_graph))
                recon_mu, ex_wt = model.decode(z, joint_keypoint)
                # if len(recon_mu) == 4:
                #     output_r, output_l, pred_mask, pred_trans = recon_mu
                #     trans_predictions.append(pred_trans.detach().cpu().numpy())
                # elif len(recon_mu) == 5:

                output_r, output_l, pred_mask, pred_trans, pred_transform = recon_mu
                trans_predictions.append(pred_transform.detach().cpu().numpy())

                mask_predictions.append(pred_mask.detach().cpu().numpy())
                output_r, output_l = output_r.detach().cpu().numpy(), output_l.detach().cpu().numpy()

                if FLAGS.pointnet:
                    output_joint = np.concatenate([output_r, output_l], axis=1)
                    palm_repeat.append(output_joint)
                else:
                    output_joint = np.concatenate([output_r, output_l], axis=2)
                    ex_wt = ex_wt.detach().cpu().numpy().squeeze()
                    sort_idx = np.argsort(ex_wt)[None, ::-1]
                    # sort_idx = np.argsort(ex_wt)[None, :]
                    for i in range(output_joint.shape[0]):
                        for j in range(output_joint.shape[1]):
                            j = sort_idx[i, j]
                            pred_info = output_joint[i, j]
                            palm_repeat.append(pred_info.tolist())
                palm_predictions.append(palm_repeat)
            palm_predictions = np.asarray(palm_predictions).squeeze()
            mask_predictions = np.asarray(mask_predictions).squeeze()
            trans_predictions = np.asarray(trans_predictions).squeeze()

            param_msg = skill_param_array_t()
            param_msg.num_entries = num_samples 
            for i in range(palm_predictions.shape[0]):
                skp = skill_param_t()
                skp.num_points = palm_predictions.shape[1] 
                for j in range(skp.num_points):
                    contact_r = lcm_utils.list2pose_stamped_lcm(palm_predictions[i, j, :7])
                    if palm_predictions[i, j].shape[0] == 7:
                        contact_l = copy.deepcopy(contact_r)
                    else:
                        contact_l = lcm_utils.list2pose_stamped_lcm(palm_predictions[i, j, 7:])
                    dual_contact_pose = dual_pose_stamped_t()
                    dual_contact_pose.right_pose = contact_r
                    dual_contact_pose.left_pose = contact_l
                    skp.contact_pose.append(dual_contact_pose)
                skp.subgoal_pose = lcm_utils.list2pose_stamped_lcm(trans_predictions[i])
                skp.mask_probs = mask_predictions[i].tolist()
                param_msg.skill_parameter_array.append(skp)
            
            log_debug('Model predictor sending message to LCM channel: %s' % observation['pub_msg_name'])
            self.lc.publish(observation['pub_msg_name'], param_msg.encode())
                