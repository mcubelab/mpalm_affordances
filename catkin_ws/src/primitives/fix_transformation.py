import os
import shutil
import random
import copy
import argparse
import numpy as np
from helper import util
import pickle


def main(args):
    #repo_dir = '/home/anthony/repos/research/mpalm_affordances/catkin_ws/src/primitives/'
    repo_dir = '/root/catkin_ws/src/primitives/'

    i = 0

    # grasp_data_dir = os.path.join(repo_dir, 'data/grasp/face_ind_test_0_fixed')
    full_data_dir = os.path.join(repo_dir, args.grasp_dir)
    #full_data_dir = os.path.join(repo_dir, args.pull_dir)

    for filename in os.listdir(full_data_dir):
        if filename.endswith('.pkl') and filename != 'metadata.pkl':
            with open(os.path.join(full_data_dir, filename), 'rb') as f:
                try:
			data = pickle.load(f)
		except EOFError:
			continue
            print("i: " + str(i))

            start_mat = util.matrix_from_pose(
                util.list2pose_stamped(data['start']))
            goal_mat = util.matrix_from_pose(
                util.list2pose_stamped(data['goal']))

            T_mat = np.matmul(goal_mat, np.linalg.inv(start_mat))

            keypoints_start_homog = np.hstack(
                (data['keypoints_start'], np.ones(
                    (data['keypoints_start'].shape[0], 1)))
            )
            data['keypoints_goal_corrected'] = np.matmul(
                T_mat, keypoints_start_homog.T).T[:, :3]
            data['transformation_corrected'] = util.pose_stamped2list(util.pose_from_matrix(T_mat))

            with open(os.path.join(full_data_dir, filename), 'wb') as new_f:
                pickle.dump(data, new_f)

            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--data_path', type=str)
    parser.add_argument('--grasp_dir', type=str)
    #parser.add_argument('--pull_dir', type=str)

    args = parser.parse_args()
    main(args)
