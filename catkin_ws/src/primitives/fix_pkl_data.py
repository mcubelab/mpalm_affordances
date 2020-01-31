import pickle
import copy
import os
import time

# repo_dir = '/home/anthony/repos/research/mpalm_affordances/catkin_ws/src/primitives/'
repo_dir = '/root/catkin_ws/src/primitives/'
data = []

i = 0

grasp_data_dir = os.path.join(repo_dir, 'data/grasp/face_ind_test_0')
new_grasp_data_dir = os.path.join(repo_dir, 'data/grasp/face_ind_test_0_fixed')

for filename in os.listdir(grasp_data_dir):
    if filename.endswith('.pkl') and filename != 'metadata.pkl' and not os.path.exists(os.path.join(new_grasp_data_dir, filename)):
        with open(os.path.join(grasp_data_dir, filename), 'rb') as f:
            data.append(pickle.load(f))

        print("i: " + str(i))

        new_data = {}
        for key in data[i].keys():
            if key != 'planner_args' and key != 'obs':
                new_data[key] = copy.deepcopy(data[i][key])

        new_data['obs'] = {}

        for key in data[i]['obs'].keys():
            if key != 'pcd_full':
                new_data['obs'][key] = copy.deepcopy(data[i]['obs'][key])

        with open(os.path.join(new_grasp_data_dir, filename), 'wb') as new_f:
            pickle.dump(new_data, new_f)

        i += 1

# pull_data_dir = os.path.join(repo_dir, 'data/pull/face_ind_large_0/')
# new_pull_data_dir = os.path.join(repo_dir, 'data/pull/face_ind_large_0_fixed/')

# for filename in os.listdir(pull_data_dir):
#     if filename.endswith('.pkl') and filename != 'metadata.pkl' and not os.path.exists(os.path.join(new_pull_data_dir, filename)):
#         with open(os.path.join(pull_data_dir, filename), 'rb') as f:
#             # data = pickle.load(f)
#             data.append(pickle.load(f))

#         new_data = {}
#         for key in data[i].keys():
#             if key != 'planner_args' and key != 'obs':
#                 new_data[key] = copy.deepcopy(data[i][key])

#         new_data['obs'] = {}

#         for key in data[i]['obs'].keys():
#             if key != 'pcd_full':
#                 new_data['obs'][key] = copy.deepcopy(data[i]['obs'][key])

#         with open(os.path.join(new_pull_data_dir, filename), 'wb') as new_f:
#             pickle.dump(new_data, new_f)
        
#         print("i: " + str(i))
#         i += 1           
