import pickle
import sys
import os
import os.path as osp
import numpy as np
from IPython import embed
from tqdm import tqdm

sys.path.append(os.environ['CODE_BASE'] + 'catkin_ws/src/primitives')
from helper import util2 as util


def apply_transformation_np(source, transformation):
    """Apply registration result to pointclouds represented as numpy arrays

    Args:
        source (np.ndarray): Source pointcloud to be transformed
        transformation (np.ndarray): Homogeneous 4x4 transformation matrix

    Result:
        np.ndarray: Transformed source point cloud
    """
    source_homog = np.ones((source.shape[0], 4))
    source_homog[:, :-1] = source

    source_transformed = np.matmul(transformation, source_homog.T).T[:, :-1]
    return source_transformed


if __name__ == "__main__":
    np_dir_root = '/root/catkin_ws/src/primitives/data/pull'
    np_dir = 'numpy_pull_cuboid_yaw'
    numpy_files = osp.join(np_dir_root, np_dir)
    numpy_files_new = osp.join(np_dir_root, np_dir + '_start_goal_pcd')
    
    if not osp.exists(numpy_files_new):
        os.makedirs(numpy_files_new)

    # for i, pickle_files in enumerate(all_pickle_files):
    for nf in tqdm(os.listdir(numpy_files)):
        np_data = np.load(osp.join(numpy_files, nf), allow_pickle=True)
        
        # load start data
        start_pointcloud = np_data['object_pointcloud']
        start_pointcloud_100 = np_data['start']
        transformation_pose_np = np_data['transformation']

        # create homogenenous matrix and pointclouds
        transformation_H = util.matrix_from_pose(util.list2pose_stamped(transformation_pose_np))
        goal_pointcloud = apply_transformation_np(start_pointcloud, transformation_H)
        goal_pointcloud_100 = apply_transformation_np(start_pointcloud_100, transformation_H)

        # save all the new data
        save_path = osp.join(numpy_files_new, str(nf).split('.npz')[0] + '_start_goal_pcd.npz')
#        print(save_path)
        np.savez(save_path,
                 start=np_data['start'],
                 start_translated=np_data['start_translated'],
                 goal=np_data['goal'],
                 transformation=np_data['transformation'],
                 object_mask_down=np_data['object_mask_down'],
                 object_mask=np_data['object_mask'],
                 object_pointcloud=np_data['object_pointcloud'],
                 keypoint_dist_left=np_data['keypoint_dist_left'],
                 keypoint_dist_right=np_data['keypoint_dist_right'],
                 contact_obj_frame_left=np_data['contact_obj_frame_left'],
                 contact_obj_frame_right=np_data['contact_obj_frame_right'],
                 contact_obj_frame_2_left=np_data['contact_obj_frame_2_left'],
                 contact_obj_frame_2_right=np_data['contact_obj_frame_2_right'],
                 contact_world_frame_left=np_data['contact_world_frame_left'],
                 contact_world_frame_right=np_data['contact_world_frame_right'],
                 contact_world_frame_2_left=np_data['contact_world_frame_2_left'],
                 contact_world_frame_2_right=np_data['contact_world_frame_2_right'],
                 start_vis=np_data['start_vis'],
                 goal_vis=np_data['goal_vis'],
                 mesh_file=np_data['mesh_file'],
                 goal_pointcloud=goal_pointcloud,
                 goal_pointcloud_100=goal_pointcloud_100)
            # except:
            #     print("skipped")
            #     continue
