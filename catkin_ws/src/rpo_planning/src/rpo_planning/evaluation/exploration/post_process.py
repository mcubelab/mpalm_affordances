import os
import os.path as osp
import pickle
import sys
import numpy as np


def get_ci(data, N=100000):
    mean_estimates = []
    data = np.asarray(data)
    for _ in range(N):
        re_sample_idx = np.random.randint(0, len(data), data.shape)
        mean_estimates.append(np.mean(data[re_sample_idx]))

    sorted_estimates = np.sort(np.array(mean_estimates))
    conf_interval = [sorted_estimates[int(0.025 * N)], sorted_estimates[int(0.975 * N)]]
    return conf_interval


"""
Final re-run data below
"""

# root_data_dir = '/root/catkin_ws/src/primitives/data/pull/gat_joint_rerun_noise_all3_200_pg/'
# root_data_dir = '/root/catkin_ws/src/primitives/data/pull/gat_joint_rerun_noise_single_200_gp/'
root_data_dir = '/root/catkin_ws/src/primitives/data/pull/gat_joint_rerun_noise_single_200_pgp/'

# root_data_dir = '/root/catkin_ws/src/primitives/data/pull/gat_joint_rerun_noise_double_single_200_pg/'
# root_data_dir = '/root/catkin_ws/src/primitives/data/pull/gat_joint_rerun_noise_double_single_200_gp/'
# root_data_dir = '/root/catkin_ws/src/primitives/data/pull/gat_joint_rerun_noise_double_single_200_pgp/'

# root_data_dir = '/root/catkin_ws/src/primitives/data/pull/gat_joint_rerun_single_200_pg/'
# root_data_dir = '/root/catkin_ws/src/primitives/data/pull/gat_joint_rerun_single_200_gp/'
# root_data_dir = '/root/catkin_ws/src/primitives/data/pull/gat_joint_rerun_single_200_pgp/'


# root_data_dir = '/root/catkin_ws/src/primitives/data/pull/baseline_rerun_single_200_pg/'
# root_data_dir = '/root/catkin_ws/src/primitives/data/pull/baseline_rerun_single_200_gp/'
# root_data_dir = '/root/catkin_ws/src/primitives/data/pull/baseline_rerun_single_200_pgp/'

# root_data_dir = '/root/catkin_ws/src/primitives/data/pull/gat_joint_test_pulling/'

data_dirs = os.listdir(root_data_dir)


ss_data_0 = {}
for fname in data_dirs:
    if 'metadata' in fname:
        continue
    obj_name = str(fname).split('_ms_eval_data.pkl')[0]
    with open(osp.join(root_data_dir, fname), 'rb') as f:
        ss_data_0[obj_name] = pickle.load(f, encoding='latin1')        


overall_planning_time = []
overall_gs = []
overall_mp = []
overall_fs = []
overall_pos_err = []
overall_ori_err = []
overall_pos_std = []
overall_ori_std = []

full_pos_errs = []
full_ori_errs = []

attempts = 0

total_trials = 0
total_mp_trials = 0
for key in ss_data_0.keys():
    if 'metadata' in key:
        continue
    gs = sum(ss_data_0[key]['grasp_success'])
    mps = ss_data_0[key]['mp_success']
    exs = ss_data_0[key]['execute_success']
    planning_time = ss_data_0[key]['planning_time']
    
    pos_errs = []
    ori_errs = []
    for i, err in enumerate(ss_data_0[key]['final_pos_error_filtered']):
        if np.abs(err) < 0.8 and np.abs(ss_data_0[key]['final_ori_error_filtered'][i]) < 1.572 and mps:
            print('filtering based on orientation outlieres!')
            pos_errs.append(err)
            ori_errs.append(ss_data_0[key]['final_ori_error_filtered'][i])
            full_pos_errs.append(err)
            full_ori_errs.append(ss_data_0[key]['final_ori_error_filtered'][i])

    pos_err_mean, pos_err_std = np.mean(pos_errs), np.std(pos_errs)
    ori_err_mean, ori_err_std = np.mean(ori_errs), np.std(ori_errs)    

    overall_mp.append(mps)
    total_trials += 1

    if mps:
        overall_gs.append(gs)
        overall_pos_err.append(pos_err_mean)
        overall_pos_std.append(pos_err_std)
        overall_ori_err.append(ori_err_mean)
        overall_ori_std.append(ori_err_std)
        overall_planning_time.append(planning_time)
        total_mp_trials += 1
print('overall\n\n\n')
string = ''
kvs = {}
kvs['trials'] = total_trials
kvs['gs'] = '%f +/- %f' % (np.sum(overall_gs) * 100.0 / total_mp_trials, np.std(overall_gs) * 100.0 / total_mp_trials)
kvs['mps'] = '%f +/- %f' % (np.sum(overall_mp) * 100.0 / total_trials, np.std(overall_mp) * 100.0 / total_trials)
kvs['pos_err'] = '%f +/- %f' % (np.mean(full_pos_errs), np.std(full_pos_errs))
kvs['ori_err'] = '%f +/- %f' % (np.mean(full_ori_errs), np.std(full_ori_errs))
kvs['time'] = '%f +/- %f' % (np.mean(overall_planning_time), np.std(overall_planning_time))

kvs['gs2'] = '%f +/- %f/%f' % (np.sum(overall_gs) * 100.0 / total_mp_trials, get_ci(overall_gs)[0], get_ci(overall_gs)[1])
kvs['mps2'] = '%f +/- %f/%f' % (np.sum(overall_mp) * 100.0 / total_trials, get_ci(overall_mp)[0], get_ci(overall_mp)[1])
kvs['pos_err2'] = '%f +/- %f/%f' % (np.mean(full_pos_errs), get_ci(full_pos_errs)[0], get_ci(full_pos_errs)[1])
kvs['ori_err2'] = '%f +/- %f/%f' % (np.mean(full_ori_errs), get_ci(full_ori_errs)[0], get_ci(full_ori_errs)[1])
kvs['time2'] = '%f +/- %f/%f' % (np.mean(overall_planning_time), get_ci(overall_planning_time)[0], get_ci(overall_planning_time)[1])


# # Breakdown of failure results


# root_data_dir = '/root/catkin_ws/src/primitives/data/pull/test_failure_modes_pgp_2/'
root_data_dir = '/root/catkin_ws/src/primitives/data/pull/gat_joint_test_failure_modes_pgp_0_test/'
# root_data_dir = '/root/catkin_ws/src/primitives/data/pull/baseline_better_heuristics_pgp_goal_feas_track_samples_0/'

"""
_2 data below was almost good, but got it messed up with some CPU issues running at same time, 
and turned off the start/goal collision checker for the baseline
"""

# root_data_dir = '/root/catkin_ws/src/primitives/data/pull/gat_joint_pgp_track_samples_2/'
# root_data_dir = '/root/catkin_ws/src/primitives/data/pull/baseline_better_heuristics_pgp_goal_feas_track_samples_2/'

"""
_3 data trying with only running one at a time, so that computation between the two can't
accidently mess each other up, and change things half way through if one of them goes down. 
Also trying to ensure that the baseline always has the new start/goal heuristics on, to help
boost performance
"""

# root_data_dir = '/root/catkin_ws/src/primitives/data/pull/gat_joint_pgp_track_samples_3/'
# root_data_dir = '/root/catkin_ws/src/primitives/data/pull/baseline_better_heuristics_pgp_goal_feas_track_samples_3/'
# root_data_dir = '/root/catkin_ws/src/primitives/data/pull/baseline_discrete_angles_centoid_pgp_0'

data_dirs = os.listdir(root_data_dir)

ss_data_0 = {}
for fname in data_dirs:
    if 'metadata' in fname:
        continue
    obj_name = str(fname).split('_ms_eval_data.pkl')[0]
    with open(osp.join(root_data_dir, fname), 'rb') as f:
        ss_data_0[obj_name] = pickle.load(f, encoding='latin1')  


# print(ss_data_0.keys())
print(len(ss_data_0.keys()))


total_samples_list = []
success_samples_list = []
failure_samples_list = []

total_start_infeas = []
total_goal_infeas = []
total_path_infeas = []
total_precon_infeas = []

for key in ss_data_0.keys():
    data = ss_data_0[key]
    failure_data = data['planning_failure']
    if len(failure_data) < 1:
        continue

    total_failures = len(failure_data[0]['path_kinematic_infeasibility'])

    if total_failures > 0:
        failure_modes = failure_data[0].keys()
        mode_data_dict = {}
        for i, mode in enumerate(failure_modes):
            if mode in ['total_samples', 'skeleton_samples']:
                continue
            num_failures = 0
            for j in range(total_failures):
                num_failures += failure_data[0][mode][j][0]

            mode_data_dict[mode] = num_failures * 100.0 / total_failures
        total_start_infeas.append(mode_data_dict['start_palm_infeasibility'])
        total_goal_infeas.append(mode_data_dict['goal_palm_infeasibility'])
        total_path_infeas.append(mode_data_dict['path_full_infeasibility'])
        total_precon_infeas.append(mode_data_dict['precondition_infeasibility'])
          
    total_samples_list.append(failure_data[0]['total_samples'])
    if data['mp_success']:
        success_samples_list.append(failure_data[0]['total_samples'])
    else:
        failure_samples_list.append(failure_data[0]['total_samples'])
    
#     print('total samples: %d, mp_success: %s, samples per skill: ' % (failure_data[0]['total_samples'], data['mp_success']))    
#     for k, v in failure_data[0]['skeleton_samples'].items():
#         print(k, v)    
    
#     per_sum = 0
#     for k, v in mode_data_dict.items():
#         print(k, v)
# #         if k in ['start_palm_infeasibility', 'goal_palm_infeasibility', 'path_full_infeasibility']:
#         per_sum += v
#     print('total ' + str(per_sum))
#     print('\n\n\n')
    
print('average total samples: ' + str(np.mean(total_samples_list)))
print('average success samples: ' + str(np.mean(success_samples_list)))
print('average failure samples: ' + str(np.mean(failure_samples_list)))

print('total failed attempts: ' + str(total_failures))
print('average start palm infeasibility: ' + str(np.mean(total_start_infeas)))
print('average goal palm infeasibility: ' + str(np.mean(total_goal_infeas)))
print('average path infeasibility: ' + str(np.mean(total_path_infeas)))
print('average precon infeasibility: ' + str(np.mean(total_precon_infeas)))
