import pickle
import numpy as np
import argparse
import os


def label_dist_order(data):
    corners = data['keypoints_start']
    l_order = []    
    r_order = []
    r_norms = None
    l_norms = None
    if data['contact_world_frame']['right'] is not None:
        r_pos = data['contact_world_frame']['right'][:3]
        dists = (np.asarray(corners) - r_pos)
        
        r_norms = np.linalg.norm(dists, axis=1)
        r_order = np.argsort(r_norms).tolist()

    if data['contact_world_frame']['left'] is not None:
        l_pos = data['contact_world_frame']['left'][:3] 
        dists = (np.asarray(corners) - l_pos)
        
        l_norms = np.linalg.norm(dists, axis=1)
        l_order = np.argsort(l_norms).tolist()   
    orders = {}
    orders['right'] = r_order
    orders['left'] = l_order

    dists = {}
    dists['right'] = r_norms
    dists['left'] = l_norms
    return orders, dists


def main(args):
    repo_dir = args.repo_dir

    i = 0

    full_data_dir = os.path.join(repo_dir, args.data_dir)
    print("Loading from : " + full_data_dir)
    raw_input('Press enter to continue\n')

    for filename in os.listdir(full_data_dir):
        if filename == 'metadata.pkl':
            continue
        if len(os.listdir(os.path.join(full_data_dir, filename))) > 0:
            pkl_dir = os.path.join(full_data_dir, filename, 'pkl')
            if len(os.listdir(pkl_dir)) > 0:
                fname = os.path.join(pkl_dir, filename + '.pkl')
                with open(fname, 'rb') as f:
                    data = pickle.load(f)
                
                orders, dists = label_dist_order(data)
                data['keypoint_order'] = orders
		data['keypoint_dists'] = dists

                with open(fname, 'wb') as f:
                    pickle.dump(data, f)
                
                i += 1
                print(i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--repo_dir', type=str, default='/root/catkin_ws/src/primitives/')

    args = parser.parse_args()
    main(args)
