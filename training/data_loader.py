import torch
import os
import random
import copy
import numpy as np
import pickle


class DataLoader(object):
    def __init__(self, data_dir):
        # path to data?
        self.data_dir = data_dir
        self.start_reps = ['pose', 'keypoints', 'pcd']

    def create_random_ordering(self, size=None):
        self.filenames = []

        for filename in os.listdir(self.data_dir):
            if filename.endswith('.pkl') and filename != 'metadata.pkl':
                self.filenames.append(os.path.join(self.data_dir, filename))

        random.shuffle(self.filenames)
        if size is not None:
            self.filenames = self.filenames[:size]

    def load_batch(self, start_ind, batch_size, start_rep='pose'):
        if start_ind + batch_size < len(self.filenames):
            batch_filenames = self.filenames[start_ind:start_ind+batch_size]
        else:
            batch_filenames = self.filenames[start_ind:]

        batch_inputs = []
        batch_targets = []
        for i, fname in enumerate(batch_filenames):
            with open(fname, 'rb') as f:
                try:
                    data = pickle.load(f)
                except EOFError as e:
                    print(e)
                    continue
            # input_sample, target_sample = \
            #     self.load_sample(data, start_rep=start_rep)
            sample = self.load_sample(data, start_rep=start_rep)
            input_sample, target_sample = sample[0], sample[1:]
            batch_inputs.append(input_sample)
            batch_targets.append(target_sample)

        return np.asarray(batch_inputs, dtype=np.float32), np.asarray(batch_targets, dtype=np.float32)

    def load_sample(self, data, start_rep):
        if start_rep not in self.start_reps:
            raise ValueError('Start state representation not recognized')

        if start_rep == 'pose':
            start_sample = data['start']
            goal_sample = data['goal']
            input_sample = start_sample + goal_sample

        elif start_rep == 'keypoints':
            # have to take care of what order?
            start_sample = data['keypoints_start']
            # goal_sample = data['keypoints_goal']
            # input_sample = np.hstack(
            #     (start_sample.flatten(), goal_sample.flatten())).tolist()
            goal_sample = data['goal']
            input_sample = start_sample.flatten().tolist() + goal_sample
        elif start_rep == 'pcd':
            start_sample = np.concatenate(data['obs']['pcd_pts']).tolist()
            goal_sample = data['goal']
            # goal_sample = data['keypoints_goal']

            input_sample = start_sample + goal_sample

        if isinstance(data['contact_obj_frame'], dict):
            full_input_sample = input_sample + data['contact_obj_frame']['right'] + data['contact_obj_frame']['left']
            target_sample_right = data['contact_obj_frame']['right']
            target_sample_left = data['contact_obj_frame']['left']
            return full_input_sample, target_sample_right, target_sample_left
        else:
            full_input_sample = input_sample + data['contact_obj_frame']
            target_sample = data['contact_obj_frame']
            return full_input_sample, target_sample


def main():
    # data_dir = '/home/anthony/repos/research/mpalm_affordances/catkin_ws/src/primitives/data/pull/face_ind_large_0_fixed'
    data_dir = '/root/catkin_ws/src/primitives/data/grasp/face_ind_test_0_fixed'
    # data_dir = '/root/catkin_ws/src/primitives/data/pull/face_ind_large_0_fixed'
    dataloader = DataLoader(data_dir=data_dir)
    dataloader.create_random_ordering()

    batch = dataloader.load_batch(0, 100, start_rep='keypoints')
    from IPython import embed
    embed()

if __name__ == "__main__":
    main()
