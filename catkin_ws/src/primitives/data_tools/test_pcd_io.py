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
import time
import h5py
import matplotlib.pyplot as plt

def print_results(write_time, read_time, size):
    print("Write time: " + str(write_time))
    print("Read time: " + str(read_time))
    print("Size (KB): " + str(size/1000.0))


data_dir = '/root/catkin_ws/src/primitives/data/pull/pull_face_all_0/test/'
test_file = os.path.join(data_dir, '2333.pkl')

with open(test_file, 'rb') as f:
    data = pickle.load(f)

# pcd_pts = data['obs']['pcd_pts']
# pcd_colors = data['obs']['pcd_colors']

# pcd = {}
# pcd['pts'] = pcd_pts
# pcd['colors'] = pcd_colors

# test_fname = os.path.join(data_dir, '2333_pcd_test')

# # test pickle
# start_write = time.time()
# with open(test_fname + '.pkl', 'wb') as f:
#     pickle.dump(pcd, f, protocol=pickle.HIGHEST_PROTOCOL)
#     # pickle.dump(pcd_pts, f)
# end_write = time.time()

# start_read = time.time()
# with open(test_fname + '.pkl', 'rb') as f:
#     data = pickle.load(f)
# pcd_pts_out = np.concatenate(data['pts'], axis=0)
# pcd_colors_out = np.concatenate(data['colors'], axis=0) / 255.0
# end_read = time.time()

# pcd_out = open3d.geometry.PointCloud()
# pcd_out.points = open3d.utility.Vector3dVector(pcd_pts_out)
# pcd_out.colors = open3d.utility.Vector3dVector(pcd_colors_out)
# open3d.visualization.draw_geometries([pcd_out])

# size = os.path.getsize(test_fname+'.pkl')

# print(".pkl results: ")
# print_results(end_write-start_write, end_read-start_read, size)
# print("--------------------\n\n")

# # test hdf5
# start_write = time.time()
# with h5py.File(test_fname + '.hdf5', 'w') as f:
#     pts_0 = f.create_dataset('pts_0', data=pcd_pts[0])
#     pts_1 = f.create_dataset('pts_1', data=pcd_pts[1])
#     pts_2 = f.create_dataset('pts_2', data=pcd_pts[2])
#     colors_0 = f.create_dataset('colors_0', data=pcd_colors[0])
#     colors_1 = f.create_dataset('colors_1', data=pcd_colors[1])
#     colors_2 = f.create_dataset('colors_2', data=pcd_colors[2])        
# end_write = time.time()

# start_read = time.time()
# pts = []
# colors = []
# with h5py.File(test_fname + '.hdf5', 'r') as f:
#     for i in range(3):
#         pts_str = 'pts_%d' % i
#         pts.append(np.array(f[pts_str]))
#         colors_str = 'colors_%d' % i
#         colors.append(np.array(f[colors_str]))
# end_read = time.time()

# pcd_out = open3d.geometry.PointCloud()
# pcd_out.points = open3d.utility.Vector3dVector(np.concatenate(pts, axis=0))
# pcd_out.colors = open3d.utility.Vector3dVector(np.concatenate(colors, axis=0)/255.0)
# open3d.visualization.draw_geometries([pcd_out])

# size = os.path.getsize(test_fname + '.hdf5')

# print(".hdf5 results: ")
# print_results(end_write-start_write, end_read-start_read, size)
# print("---------------------\n\n")

# # test ply
# start_write = time.time()
# pcd = open3d.geometry.PointCloud()
# pcd.points = open3d.utility.Vector3dVector(np.concatenate(pcd_pts, axis=0))
# pcd.colors = open3d.utility.Vector3dVector(np.concatenate(pcd_colors, axis=0) / 255.0)
# open3d.io.write_point_cloud(test_fname + '.ply', pcd)    
# end_write = time.time()

# start_read = time.time()
# pcd = open3d.io.read_point_cloud(test_fname + '.ply')
# pcd_pts_out = np.asarray(pcd.points)
# pcd_colors_out = np.asarray(pcd.colors)
# end_read = time.time()

# pcd_out = open3d.geometry.PointCloud()
# pcd_out.points = open3d.utility.Vector3dVector(pcd_pts_out)
# pcd_out.colors = open3d.utility.Vector3dVector(pcd_colors_out)
# open3d.visualization.draw_geometries([pcd_out])

# size = os.path.getsize(test_fname+'.ply')

# print(".ply results: ")
# print_results(end_write-start_write, end_read-start_read, size)
# print("--------------------\n\n")


# # test pcd
# start_write = time.time()
# pcd = open3d.geometry.PointCloud()
# pcd.points = open3d.utility.Vector3dVector(np.concatenate(pcd_pts, axis=0))
# pcd.colors = open3d.utility.Vector3dVector(np.concatenate(pcd_colors, axis=0) / 255.0)
# open3d.io.write_point_cloud(test_fname + '.pcd', pcd)    
# end_write = time.time()

# start_read = time.time()
# # with open(test_fname + '.ply', 'rb') as f:
# #     pickle.load(f)
# pcd = open3d.io.read_point_cloud(test_fname + '.pcd')
# pcd_pts_out = np.asarray(pcd.points)
# pcd_colors_out = np.asarray(pcd.colors)
# end_read = time.time()

# pcd_out = open3d.geometry.PointCloud()
# pcd_out.points = open3d.utility.Vector3dVector(pcd_pts_out)
# pcd_out.colors = open3d.utility.Vector3dVector(pcd_colors_out)
# open3d.visualization.draw_geometries([pcd_out])

# size = os.path.getsize(test_fname+'.pcd')

# print(".pcd results: ")
# print_results(end_write-start_write, end_read-start_read, size)
# print("--------------------\n\n")

# embed()

seg = data['obs']['seg'][0]
test_fname = os.path.join(data_dir, '2333_seg_test')

# test pickle
start_write = time.time()
with open(test_fname + '.pkl', 'wb') as f:
    pickle.dump(seg, f, protocol=pickle.HIGHEST_PROTOCOL)
    # pickle.dump(pcd_pts, f)
end_write = time.time()

start_read = time.time()
with open(test_fname + '.pkl', 'rb') as f:
    data = pickle.load(f)
end_read = time.time()

plt.imshow(data)
plt.show()

size = os.path.getsize(test_fname+'.pkl')

print(".pkl results: ")
print_results(end_write-start_write, end_read-start_read, size)
print("--------------------\n\n")

# test hdf5
start_write = time.time()
with h5py.File(test_fname + '.hdf5', 'w') as f:
    pts_0 = f.create_dataset('seg_0', data=seg)   
end_write = time.time()

start_read = time.time()
with h5py.File(test_fname + '.hdf5', 'r') as f:
    seg_out = np.array(f['seg_0'])
end_read = time.time()

plt.imshow(data)
plt.show()

size = os.path.getsize(test_fname + '.hdf5')

print(".hdf5 results: ")
print_results(end_write-start_write, end_read-start_read, size)
print("---------------------\n\n")

# test cv2 with png
start_write = time.time()
cv2.imwrite(test_fname + '.png', seg)   
end_write = time.time()

start_read = time.time()
data = cv2.imread(test_fname + '.png')
end_read = time.time()

plt.imshow(data)
plt.show()

size = os.path.getsize(test_fname + '.png')

print(".cv2 results: ")
print_results(end_write-start_write, end_read-start_read, size)
print("---------------------\n\n")

embed()