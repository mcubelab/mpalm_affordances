#!/usr/bin/env python
# coding: utf-8

import os
import os.path as osp
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import copy

from mpl_toolkits.mplot3d import axes3d
import plotly.graph_objects as go
from plotly.subplots import make_subplots


root_data_dir = 'test'
ss_data_0 = {}

data_dirs = os.listdir(root_data_dir)

for fname in data_dirs:
    if 'metadata' in fname:
        continue
    obj_name = str(fname).split('_eval_data.pkl')[0]
    with open(osp.join(root_data_dir, fname), 'rb') as f:
        ss_data_0[obj_name] = pickle.load(f, encoding='latin1')                     
            
            
overall_gs = []
overall_mp = []
overall_fs = []
overall_attempts = []
overall_pos_err = []
overall_ori_err = []
overall_pos_std = []
overall_ori_std = []

full_pos_errs = []
full_ori_errs = []

full_success = []

attempts = 0

total_trials = 0
grasp_trials = 0
print(ss_data_0.keys())
for key in ss_data_0.keys():
    if 'metadata' in key:
        continue
    trials = ss_data_0[key]['trials']
    if trials == 0:
        continue
    gs = ss_data_0[key]['grasp_success']
    try:
        model = ss_data_0[key]['predictions'][0]['model_path']
    except:
        model=None
        pass

    mps = ss_data_0[key]['mp_success']
    fs = ss_data_0[key]['face_success']
    
    pos_errs = []
    ori_errs = []

    
#         for i, err in enumerate(ss_data_0[key]['final_pos_error_filtered']):
#             if np.abs(err) < 0.8:
#                 pos_errs.append(err)
#                 ori_errs.append(ss_data_0[key]['final_ori_error_filtered'][i])
#                 full_pos_errs.append(err)
#                 full_ori_errs.append(ss_data_0[key]['final_ori_error_filtered'][i])
    
    for i, err in enumerate(ss_data_0[key]['final_pos_error']):
        if np.abs(err) < 0.8:
            pos_errs.append(err)
            ori_errs.append(ss_data_0[key]['final_ori_error'][i])
            full_pos_errs.append(err)
            full_ori_errs.append(ss_data_0[key]['final_ori_error'][i])
            
        if np.abs(err) < 0.03 and np.abs(ss_data_0[key]['final_ori_error'][i]) < np.deg2rad(20):
            full_success.append(True)
        else:
            full_success.append(False)

    pos_err_mean, pos_err_std = np.mean(pos_errs), np.std(pos_errs)
    ori_err_mean, ori_err_std = np.mean(ori_errs), np.std(ori_errs)    
    

    overall_gs.append(gs)
    overall_mp.append(mps)
    overall_fs.append(fs)
    overall_pos_err.append(pos_err_mean)
    overall_pos_std.append(pos_err_std)
    overall_ori_err.append(ori_err_mean)
    overall_ori_std.append(ori_err_std)
    total_trials += trials
    overall_attempts = []
    for i, att in enumerate(ss_data_0[key]['mp_attempts']):
        overall_attempts.append(att)
        if att < 15:
            grasp_trials += 1

kvs = {}
kvs['global_success'] = '%f' % (np.sum(full_success) * 100.0 / total_trials)
#     kvs['global_success'] = '%f +/- %f/%f' % (np.sum(full_success) * 100.0 / total_trials, get_ci(full_success)[0], get_ci(full_success)[1])
kvs['global_success_2'] = '%f' % (np.sum(full_success) * 100.0 / grasp_trials)
#     kvs['global_success'] = '%f' % (np.sum(full_success) * 100.0 / len(full_success))
for k, v in kvs.items():
    if isinstance(v, str):
        string += '%s: %s, \n' % (k, v)
    else:
        string += '%s: %f , \n' % (k, v)
print(string)
print('\n\n\n')
data_dict = {}
data_dict['global_success_rate'] = [
    np.sum(full_success) * 100.0 / total_trials
]

if grasping:
    np.savez(
        'grasping_ss_data.npz',
        gat_joint_mask=data_dict['global_success_rate'],
        gat_indep_mask=data_dict['global_success_rate'],
        pointnet_joint_mask=data_dict['global_success_rate'],
        pointnet_indep_mask=data_dict['global_success_rate'],
        gat_joint_trans=data_dict['global_success_rate'],
        gat_indep_trans=data_dict['global_success_rate']
    )
elif pulling:
    np.savez(
        'pulling_ss_data.npz',
        pointnet_joint_trans=data_dict['global_success_rate'],
        pointnet_indep_trans=data_dict['global_success_rate'],
        gat_joint_trans=data_dict['global_success_rate'],
        gat_indep_trans=data_dict['global_success_rate']
    )
elif pushing:
    np.savez(
        'pushing_ss_data.npz',
        pointnet_joint_trans=data_dict['global_success_rate'],
        pointnet_indep_trans=data_dict['global_success_rate'],
        gat_joint_trans=data_dict['global_success_rate'],
        gat_indep_trans=data_dict['global_success_rate']
    )
else:
    pass
