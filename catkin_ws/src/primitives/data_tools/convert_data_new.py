import pickle
import os
import os.path as osp
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    #pickle_base = "/data/vision/billf/scratch/yilundu/dataset/face_ind_test_0_fixed/train/"

    pickle_base = "/data/scratch/asimeonov/repos/research/mpalm_affordances/catkin_ws/src/primitives/data/grasp/grasping_multi_diverse"
    pickle_paths = os.listdir(pickle_base)
    all_pickle_files = []
    for path in pickle_paths:
        all_pickle_files.append(os.listdir(osp.join(pickle_base, path)))
    #numpy_files = "/data/vision/billf/scratch/yilundu/dataset/numpy_robot_keypoint/"
    numpy_files = "/data/scratch/asimeonov/repos/research/mpalm_affordances/catkin_ws/src/primitives/data/grasp/numpy_grasp_diverse_0/"

    if not osp.exists(numpy_files):
        os.makedirs(numpy_files)

    for i, pickle_files in enumerate(all_pickle_files):
        for pf in tqdm(pickle_files):
            pkl_base_path = osp.join(pickle_base, pickle_paths[i], pf, "pkl", "{}.pkl".format(pf))
            pcd_base_path = osp.join(pickle_base, pickle_paths[i], pf, "pcd", "{}_pcd.pkl".format(pf))            
            try:
                with open(pkl_base_path, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                with open(pcd_base_path, 'rb') as f:
                    pcd_data = pickle.load(f, encoding='latin1')                    
                prefix = '%d_' % i
                pf = prefix + pf
                save_path = osp.join(numpy_files, pf.split(".")[0] + ".npz")

                start_vis = data['start']
                goal_vis = data['goal']
                start = data['down_pcd_pts']
                goal = data['keypoints_goal']

                contact_obj_frame_right = data['contact_obj_frame']['right']
                contact_obj_frame_left = data['contact_obj_frame']['left']
                contact_obj_frame_2_right = data['contact_obj_frame_2']['right']
                contact_obj_frame_2_left = data['contact_obj_frame_2']['left']
                contact_world_frame_right = data['contact_world_frame']['right']
                contact_world_frame_left = data['contact_world_frame']['left']
                contact_world_frame_2_right = data['contact_world_frame_2']['right']
                contact_world_frame_2_left = data['contact_world_frame_2']['left']
                #contact_obj_frame_right = data['contact_obj_frame']['right'][:3]
                #contact_obj_frame_left = data['contact_obj_frame']['left'][:3]
                #contact_obj_frame_2_right = data['contact_obj_frame_2']['right'][:3]
                #contact_obj_frame_2_left = data['contact_obj_frame_2']['left'][:3]

                transformation = data['transformation']
                keypoint_dist_left = data['down_pcd_dists']['left']
                keypoint_dist_right = data['down_pcd_dists']['right']

                mesh_file = data['mesh_file'].split('objects/cuboids/')[1]

                # obj_translation = data['start'][:3]
                obj_translation = data['center_of_mass']
                start_translated = data['down_pcd_pts'] - obj_translation
                object_mask_down = data['down_pcd_mask']
                object_mask = data['pcd_mask']
                table_mask = data['table_contact_mask']
                object_pointcloud = np.concatenate(pcd_data['pts'], axis=0)
    #            table_pointcloud = np.concatenate(pcd_data['table_pcd_pts'], axis=0)
                #print('Save path: ' + save_path)
                np.savez(save_path,
                         start=start,
                         start_translated=start_translated,
                         goal=goal,
                         transformation=transformation,
                         object_mask_down=object_mask_down,
                         object_mask=object_mask,
                         object_pointcloud=object_pointcloud,
                         keypoint_dist_left=keypoint_dist_left,
                         keypoint_dist_right=keypoint_dist_right,
                         contact_obj_frame_left=contact_obj_frame_left,
                         contact_obj_frame_right=contact_obj_frame_right,
                         contact_obj_frame_2_left=contact_obj_frame_2_left,
                         contact_obj_frame_2_right=contact_obj_frame_2_right,
                         contact_world_frame_left=contact_world_frame_left,
                         contact_world_frame_right=contact_world_frame_right,
                         contact_world_frame_2_left=contact_world_frame_2_left,
                         contact_world_frame_2_right=contact_world_frame_2_right,
                         start_vis=start_vis,
                         goal_vis=goal_vis,
                         mesh_file=mesh_file)
            except:
                print("skipped")
                continue
