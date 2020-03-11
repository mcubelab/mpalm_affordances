import pickle
import numpy as np
import os
import shutil
import random
import copy
import argparse
import cv2
from IPython import embed
import open3d


def main(args):
    #repo_dir = '/home/anthony/repos/research/mpalm_affordances/catkin_ws/src/primitives/'
    repo_dir = '/root/catkin_ws/src/primitives/'

    i = 0

    full_data_dir = os.path.join(repo_dir, args.data_path)

    pickle_dir = 'pkl'
    img_dir = 'img'
    pcd_dir = 'pcd'

    for filename in os.listdir(full_data_dir):
        if filename.endswith('.pkl') and filename != 'metadata.pkl':
            with open(os.path.join(full_data_dir, filename), 'rb') as f:
                try:
                    data = pickle.load(f)
                except EOFError:
                    continue
            # print("i: " + str(i))

            # embed()

            if not isinstance(data, dict):
                print("DATA CORRUPT!")
                continue
            new_data = {}
            for key in data.keys():
                if not key == 'obs':
                    new_data[key] = copy.deepcopy(data[key])

            raw_fname = filename.split('.pkl')[0]
            
            pcd_fname = os.path.join(full_data_dir, raw_fname, pcd_dir, raw_fname + '_pcd.pkl')
            
            depth_fnames = [raw_fname + '_depth_%d.png' % j for j in range(3)]
            rgb_fnames = [raw_fname + '_rgb_%d.png' % j for j in range(3)]
            seg_fnames = [raw_fname + '_seg_%d.pkl' % j for j in range(3)]

            if not os.path.exists(os.path.join(full_data_dir, raw_fname, pickle_dir)):
                os.makedirs(os.path.join(full_data_dir, raw_fname, pickle_dir))
            if not os.path.exists(os.path.join(full_data_dir, raw_fname, img_dir)):
                os.makedirs(os.path.join(full_data_dir, raw_fname, img_dir))
            if not os.path.exists(os.path.join(full_data_dir, raw_fname, pcd_dir)):
                os.makedirs(os.path.join(full_data_dir, raw_fname, pcd_dir))

            with open(os.path.join(full_data_dir, raw_fname, pickle_dir, filename), 'wb') as new_f:
                pickle.dump(new_data, new_f)

            if 'obs' in data.keys():
                # save depth
                for k, fname in enumerate(rgb_fnames):
                    rgb_fname = os.path.join(full_data_dir, raw_fname, img_dir, rgb_fnames[k])
                    depth_fname = os.path.join(full_data_dir, raw_fname, img_dir, depth_fnames[k])
                    seg_fname = os.path.join(full_data_dir, raw_fname, img_dir, seg_fnames[k])

                    # save depth
                    scale = 1000.0
                    sdepth = data['obs']['depth'][k] * scale
                    cv2.imwrite(depth_fname, sdepth.astype(np.uint16))

                    # save rgb
                    cv2.imwrite(rgb_fname, data['obs']['rgb'][k])

                    # save seg
                    # cv2.imwrite(seg_fname, data['obs']['seg'][i])
                    with open(seg_fname, 'wb') as f:
                        pickle.dump(data['obs']['seg'][k], f, protocol=pickle.HIGHEST_PROTOCOL)

                # save pointcloud as pcd
                # pcd = open3d.geometry.PointCloud()
                # pcd.points = open3d.utility.Vector3dVector(np.concatenate(data['obs']['pcd_pts'], axis=0))
                # pcd.colors = open3d.utility.Vector3dVector(np.concatenate(data['obs']['pcd_colors'], axis=0) / 255.0)
                # open3d.io.write_point_cloud(pcd_fname, pcd)

                # save pointcloud arrays as .pkl with high protocol
                pcd_pts = data['obs']['pcd_pts']
                pcd_colors = data['obs']['pcd_colors']

                pcd = {}
                pcd['pts'] = pcd_pts
                pcd['colors'] = pcd_colors
                with open(pcd_fname, 'wb') as f:
                    pickle.dump(pcd, f, protocol=pickle.HIGHEST_PROTOCOL)

                # embed()

            i += 1
            print("i: " + str(i), " filename: " + str(filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)

    args = parser.parse_args()
    main(args)