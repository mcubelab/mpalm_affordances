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
        assert(os.path.exists(self.data_dir))
        self.start_reps = ['pose', 'keypoints', 'pcd']

    def create_random_ordering(self, size=None):
        self.filenames = []

        for filename in os.listdir(self.data_dir):
            # if filename.endswith('.pkl') and filename != 'metadata.pkl':
            if len(filename.split('.')) == 1:
                self.filenames.append(os.path.join(self.data_dir, filename, 'pkl', filename+'.pkl'))

        random.shuffle(self.filenames)
        if size is not None:
            self.filenames = self.filenames[:size]
        # print('samples: ')
        # print(self.filenames)
        # go = raw_input('press enter to start...')

    def load_dataset(self, start_rep='pose', goal_rep='pose', task='contact'):
        inputs = []
        decoder_inputs = []
        targets = []
        for i, fname in enumerate(self.filenames):
            with open(fname, 'rb') as f:
                try:
                    data = pickle.load(f)
                except EOFError as e:
                    print(e)
                    continue
            # input_sample, target_sample = \
            #     self.load_sample(data, start_rep=start_rep)
            if task == 'contact':
                sample = self.load_sample(data, start_rep=start_rep, goal_rep=goal_rep)
            elif task == 'goal':
                sample = self.load_goal_sample(data, start_rep=start_rep, goal_rep=goal_rep)
            elif task == 'transformation':
                sample = self.load_transformation_sample(data)
            input_sample, decoder_input_sample, target_sample = sample[0], sample[1], sample[2:]
            inputs.append(input_sample)
            decoder_inputs.append(decoder_input_sample)
            targets.append(target_sample)

        assert(len(inputs) > 0 and len(decoder_inputs) > 0 and len(targets) > 0)
        return (np.asarray(inputs, dtype=np.float32),
                np.asarray(decoder_inputs, dtype=np.float32),
                np.asarray(targets, dtype=np.float32))

    def sample_batch(self, dataset, start_ind, batch_size):
        inputs, decoder_inputs, targets = dataset

        if start_ind + batch_size < inputs.shape[0]:
            batch_inputs = inputs[start_ind:start_ind+batch_size, :]
            batch_decoder_inputs = decoder_inputs[start_ind:start_ind+batch_size, :]
            batch_targets = targets[start_ind:start_ind+batch_size, :]
        else:
            batch_inputs = inputs[start_ind:, :]
            batch_decoder_inputs = decoder_inputs[start_ind:, :]
            batch_targets = targets[start_ind:, :]
        return batch_inputs, batch_decoder_inputs, batch_targets

    def load_batch(self, start_ind, batch_size, start_rep='pose', goal_rep='pose', task='contact'):
        if start_ind + batch_size < len(self.filenames):
            batch_filenames = self.filenames[start_ind:start_ind+batch_size]
        else:
            batch_filenames = self.filenames[start_ind:]

        batch_inputs = []
        batch_decoder_inputs = []
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
            if task == 'contact':
                sample = self.load_sample(data, start_rep=start_rep, goal_rep=goal_rep)
            elif task == 'goal':
                sample = self.load_goal_sample(data, start_rep=start_rep, goal_rep=goal_rep)
            elif task == 'transformation':
                sample = self.load_transformation_sample(data)
            input_sample, decoder_input_sample, target_sample = sample[0], sample[1], sample[2:]
            batch_inputs.append(input_sample)
            batch_decoder_inputs.append(decoder_input_sample)
            batch_targets.append(target_sample)

        assert(len(batch_inputs) > 0 and len(batch_decoder_inputs) > 0 and len(batch_targets) > 0)
        return (np.asarray(batch_inputs, dtype=np.float32),
                np.asarray(batch_decoder_inputs, dtype=np.float32),
                np.asarray(batch_targets, dtype=np.float32))

    def load_sample(self, data, start_rep, goal_rep):
        if start_rep not in self.start_reps:
            raise ValueError('Start state representation not recognized')

        if start_rep == 'pose':
            start_sample = data['start']
        elif start_rep == 'keypoints':
            # have to take care of what order?
            start_sample = data['keypoints_start'].flatten().tolist()
        elif start_rep == 'pcd':
            start_sample = np.concatenate(data['obs']['pcd_pts']).tolist()

        if goal_rep == 'pose':
            goal_sample = data['goal']
            input_sample = start_sample + goal_sample
        elif goal_rep == 'keypoints':
            goal_sample = data['keypoints_goal_corrected']
            input_sample = start_sample + goal_sample.flatten().tolist()

        if isinstance(data['contact_obj_frame'], dict):
            if data['contact_obj_frame']['left'] is None:
                full_input_sample = input_sample + data['contact_obj_frame']['right']
                target_sample_right = data['contact_obj_frame']['right']
                return full_input_sample, input_sample, target_sample_right
            elif data['contact_obj_frame']['right'] is None:
                full_input_sample = input_sample + data['contact_obj_frame']['left']
                target_sample_right = data['contact_obj_frame']['left']
                return full_input_sample, input_sample, target_sample_left
            else:
                full_input_sample = input_sample + data['contact_obj_frame']['right'] + data['contact_obj_frame']['left']
                target_sample_right = data['contact_obj_frame']['right']
                target_sample_left = data['contact_obj_frame']['left']
                return full_input_sample, input_sample, target_sample_right, target_sample_left
        else:
            full_input_sample = input_sample + data['contact_obj_frame']
            target_sample = data['contact_obj_frame']
            return full_input_sample, input_sample, target_sample

    def load_goal_sample(self, data, start_rep, goal_rep):
        if start_rep not in self.start_reps:
            raise ValueError('Start state representation not recognized')

        if start_rep == 'pose':
            start_sample = data['start']
        elif start_rep == 'keypoints':
            # have to take care of what order?
            start_sample = data['keypoints_start'].flatten().tolist()
        elif start_rep == 'pcd':
            start_sample = np.concatenate(data['obs']['pcd_pts']).tolist()

        if goal_rep == 'pose':
            goal_sample = data['goal']
        elif goal_rep == 'keypoints':
            goal_sample = data['keypoints_goal_corrected'][:, :3].flatten().tolist()

        input_sample = start_sample
        target_sample = goal_sample
        return input_sample+target_sample, input_sample, target_sample

    def load_transformation_sample(self, data):
        input_sample = data['transformation_corrected']
        target_sample = data['transformation_corrected']
        keypoints = data['keypoints_start'].flatten().tolist()
        #return input_sample, data['transformation_corrected'], target_sample
        return keypoints+input_sample, keypoints, target_sample

def main():
    # data_dir = '/home/anthony/repos/research/mpalm_affordances/catkin_ws/src/primitives/data/pull/face_ind_large_0_fixed'
    data_dir = '/root/catkin_ws/src/primitives/data/pull/face_ind_large_0_fixed/train/'
    # data_dir = '/root/catkin_ws/src/primitives/data/pull/face_ind_large_0_fixed'
    dataloader = DataLoader(data_dir=data_dir)
    dataloader.create_random_ordering()

    import time
    start = time.time()
    # batch = dataloader.load_batch(0, 100, start_rep='keypoints')
    dataset = dataloader.load_dataset(start_rep='keypoints')
    batch = dataloader.sample_batch(dataset, 0, 100)
    print('Time: ' + str(time.time() - start))
    from IPython import embed
    embed()

if __name__ == "__main__":
    main()
