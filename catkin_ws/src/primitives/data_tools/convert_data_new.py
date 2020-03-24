import pickle
import os
import os.path as osp
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    #pickle_base = "/data/vision/billf/scratch/yilundu/dataset/face_ind_test_0_fixed/train/"
    
    pickle_base = "/data/scratch/asimeonov/repos/research/mpalm_affordances/catkin_ws/src/primitives/data/grasp/"
    pickle_paths = os.listdir(pickle_base)
    all_pickle_files = []
    for path in pickle_paths:
        all_pickle_files.append(os.listdir(osp.join(pickle_base, path)))
    #numpy_files = "/data/vision/billf/scratch/yilundu/dataset/numpy_robot_keypoint/"
    numpy_files = "/data/scratch/asimeonov/repos/research/mpalm_affordances/catkin_ws/src/primitives/data/grasp/numpy_robot_pcd/"

    if not osp.exists(numpy_files):
        os.makedirs(numpy_files)

    for i, pickle_files in enumerate(all_pickle_files):
        for pf in tqdm(pickle_files):
            base_path = osp.join(pickle_base, pickle_paths[i], pf, "pkl", "{}.pkl".format(pf))
            try:
                with open(base_path, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                prefix = '%d_' % i
                pf = prefix + pf
                save_path = osp.join(numpy_files, pf.split(".")[0] + ".npz")

                start_vis = data['start']
                goal_vis = data['goal']
                start = data['down_pcd_pts']
                goal = data['keypoints_goal']
                contact_obj_frame_right = data['contact_obj_frame']['right']
                contact_obj_frame_left = data['contact_obj_frame']['left']
                transformation = data['transformation']
                keypoint_dist_left = data['down_pcd_dists']['left']
                keypoint_dist_right = data['down_pcd_dists']['right']
                
                mesh_file = data['mesh_file'].split('objects/cuboids/')[1]
                #print(mesh_file)
                np.savez(save_path, start=start, goal=goal, transformation=transformation, keypoint_dist_left=keypoint_dist_left, keypoint_dist_right=keypoint_dist_right, contact_obj_frame_left=contact_obj_frame_left, contact_obj_frame_right=contact_obj_frame_right, start_vis=start_vis, goal_vis=goal_vis, mesh_file=mesh_file)
            except:
                print("skipped")
                continue
