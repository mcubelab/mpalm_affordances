import os
import os.path as osp

import torchvision.transforms.functional as TF
import random

import torch.utils.data as data
# from scipy.misc import imread
# from scipy.misc import imsave
import torch
import os
import os.path as osp

# from scipy.misc import imresize
import numpy as np

# from transformations import quaternion_from_euler


class RobotDataset(data.Dataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, train=False):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.base_path = "/data/vision/billf/scratch/yilundu/dataset/numpy_robot_grasp"
        np_files = os.listdir(self.base_path)
        np_files = sorted(np_files)
        self.data = [osp.join(self.base_path, np_file) for np_file in np_files]
        idx = int(len(self.data) * 0.9)

        if train:
            self.data = self.data[:idx]
        else:
            self.data = self.data[idx:]


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        path = self.data[index]
        data = np.load(path, allow_pickle=True)
        start = data['start']
        goal = data['goal']
        contact_obj_frame = data['contact_obj_frame']

        return start, goal, contact_obj_frame

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.data)

class RobotDatasetGrasp(data.Dataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, train=False):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.base_path = "/data/vision/billf/scratch/yilundu/dataset/numpy_robot_grasp"
        np_files = os.listdir(self.base_path)
        np_files = sorted(np_files)
        self.data = [osp.join(self.base_path, np_file) for np_file in np_files]
        idx = int(len(self.data) * 0.9)

        if train:
            self.data = self.data[:idx]
        else:
            self.data = self.data[idx:]


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        path = self.data[index]
        data = np.load(path, allow_pickle=True)
        start = data['start']
        # print(start.shape)
        goal = data['goal']
        # print(goal.shape)
        obj_frame = data['contact_obj_frame'].item()
        contact_obj_frame = np.concatenate([obj_frame['right'], obj_frame['left']])
        # print(len(obj_frame['right']))
        # print(len(obj_frame['left']))
        left_min = np.array([-.02, -0.08, -0.025])
        left_max = np.array([0.05, 0.08, 0.05])

        left_quaternion = quaternion_from_euler(*np.random.uniform(-np.pi, np.pi, (3,)))
        right_quaternion = quaternion_from_euler(*np.random.uniform(-np.pi, np.pi, (3,)))

        right_min = np.array([-0.05, -0.08, -0.025])
        right_max = np.array([0.022, 0.08, 0.05])

        left_sample = np.random.uniform(0, 1, (3,)) * (left_max - left_min) + left_min
        right_sample = np.random.uniform(0, 1, (3,)) * (right_max - right_min) + right_min

        left_frame = np.concatenate([left_sample, left_quaternion])
        right_frame = np.concatenate([right_sample, right_quaternion])
        contact_obj_frame_neg = np.concatenate([right_frame, left_frame])

        return start, goal, contact_obj_frame, contact_obj_frame_neg

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.data)


class RobotKeypointsDatasetGrasp(data.Dataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, split='train'):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # self.base_path = "/data/vision/billf/scratch/yilundu/dataset/numpy_robot_keypoint"
        # self.base_path = "/data/scratch/asimeonov/repos/research/mpalm_affordances/catkin_ws/src/primitives/data/grasp/numpy_robot_pcd"
        self.base_path = "/root/catkin_ws/src/primitives/data/grasp/numpy_grasp_diverse_0"        
        np_files = os.listdir(self.base_path)
        np_files = sorted(np_files)
        self.data = [osp.join(self.base_path, np_file) for np_file in np_files]
        idx = int(len(self.data) * 0.9)
        idx_of = int(len(self.data) * 0.05)
        self.split = split

        if split == 'train':
            self.data = self.data[:idx]
        elif split == 'overfit':
            self.data = self.data[:idx_of]
        else:
            self.data = self.data[idx:]


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        path = self.data[index]
        data = np.load(path, allow_pickle=True)
        start = data['start'][:100]
        start_mean = np.mean(start, axis=0, keepdims=True)
        start_normalized = (start - start_mean)
        start_mean = np.tile(start_mean, (start.shape[0], 1))

        start = np.concatenate([start_normalized, start_mean], axis=1)
        transformation = data['transformation']

        centroid = np.mean(data['object_pointcloud'], axis=0)

        # obj_frame_right = data['contact_obj_frame_right']
        # obj_frame_left = data['contact_obj_frame_left']

        # obj_frame_right = data['contact_world_frame_right']
        # obj_frame_left = data['contact_world_frame_left']

        obj_frame_right = np.concatenate(
            [data['contact_world_frame_right'][:3] - centroid, data['contact_world_frame_right'][3:]])
        obj_frame_left = np.concatenate(
            [data['contact_world_frame_left'][:3] - centroid, data['contact_world_frame_left'][3:]])

        # obj_frame_right = np.concatenate(
        #     [data['contact_world_frame_right'][:3] - centroid, 
        #      data['contact_world_frame_2_right'][:3] - centroid], axis=0)
        # obj_frame_left = np.concatenate(
        #     [data['contact_world_frame_left'][:3] - centroid, 
        #      data['contact_world_frame_2_left'][:3] - centroid], axis=0)

        # r_normal = (data['contact_world_frame_2_right'][:3] - data['contact_world_frame_right'][:3])
        # l_normal = (data['contact_world_frame_2_left'][:3] - data['contact_world_frame_left'][:3])

        # r_normal = r_normal/np.linalg.norm(r_normal)
        # l_normal = l_normal/np.linalg.norm(l_normal)

        # obj_frame_right = np.concatenate(
        #     [data['contact_world_frame_right'][:3] - centroid, r_normal], axis=0)
        # obj_frame_left = np.concatenate(
        #     [data['contact_world_frame_left'][:3] - centroid, l_normal], axis=0)

        keypoint_dist_left = data['keypoint_dist_left'][:100]
        keypoint_dist_right = data['keypoint_dist_right'][:100]

        keypoint_dist = np.stack([keypoint_dist_left, keypoint_dist_right], axis=1).min(axis=1)
        select_idx = np.argsort(keypoint_dist)
        obj_frame = np.concatenate([obj_frame_left, obj_frame_right])
        decoder_x = np.concatenate([transformation])

        object_mask_down = data['object_mask_down'][:100]
        translation = np.array([data['transformation'][0], data['transformation'][1]])        

        if self.split == "train" or self.split == 'overfit':
            # return start, transformation, obj_frame, select_idx, decoder_x
            return start, transformation, obj_frame, select_idx, decoder_x, object_mask_down, translation
        else:
            try:
                start_vis = data['start_vis']
                goal_vis = data['goal_vis']
                mesh_file = data['mesh_file']
                vis_misc = np.concatenate([start_vis, goal_vis])

                return start, transformation, obj_frame, select_idx, vis_misc, decoder_x, str(mesh_file)
            except:
                return self.__getitem__((index+1) % len(self.data))

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.data)


class RobotKeypointsDatasetGraspMask(data.Dataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, split='train'):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # self.base_path = "/data/vision/billf/scratch/yilundu/dataset/numpy_robot_keypoint"
        # self.base_path = "/data/scratch/asimeonov/repos/research/mpalm_affordances/catkin_ws/src/primitives/data/grasp/numpy_grasp_table_pcd"
        self.base_path = "/root/catkin_ws/src/primitives/data/grasp/numpy_grasp_table_pcd"        
        np_files = os.listdir(self.base_path)
        np_files = sorted(np_files)
        self.data = [osp.join(self.base_path, np_file) for np_file in np_files]
        idx = int(len(self.data) * 0.9)
        self.split = split

        if split == 'train':
            self.data = self.data[:idx]
        else:
            self.data = self.data[idx:]


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        path = self.data[index]
        data = np.load(path, allow_pickle=True)
        start = data['start'][:100]

        start_mean = np.mean(start, axis=0, keepdims=True)
        start_normalized = (start - start_mean)
        start_mean = np.tile(start_mean, (start.shape[0], 1))

        start = np.concatenate([start_normalized, start_mean], axis=1)
        object_mask_down = data['object_mask_down'][:100]
        translation = np.array([data['transformation'][0], data['transformation'][1]])

        return start, object_mask_down, translation

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.data)

if __name__ == "__main__":
    data = RobotDatasetGrasp()
    dat = data[0]
    print("here")
