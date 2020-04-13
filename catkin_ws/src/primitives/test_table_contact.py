import open3d
pcd_1 = open3d.geometry.PointCloud()
# pcd_1.points = open3d.utility.Vector3dVector(obs['down_pcd_pts'])
# pcd_1.colors = open3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0]]).repeat(obs['down_pcd_pts'].shape[0], axis=0))
pcd_1.points = open3d.utility.Vector3dVector(pcd_pts)
pcd_1.colors = open3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0]]).repeat(pcd_pts.shape[0], axis=0))

#open3d.visualization.draw_geometries([pcd_1])

pcd_2 = open3d.geometry.PointCloud()
pcd_2.points = open3d.utility.Vector3dVector(pcd_goal)
pcd_2.colors = open3d.utility.Vector3dVector(np.array([[0.0, 1.0, 0.0]]).repeat(pcd_goal.shape[0], axis=0))
# open3d.visualization.draw_geometries([pcd_2])

#open3d.visualization.draw_geometries([pcd_1, pcd_2])

import copy
pcd_3 = copy.deepcopy(pcd_2)
#colors1 = np.array([[1.0, 0.0, 0.0]]).repeat(obs['down_pcd_pts'].shape[0], axis=0)
#colors3 = np.array([[0.0, 1.0, 0.0]]).repeat(obs['down_pcd_pts'].shape[0], axis=0)
colors1 = np.array([[1.0, 0.0, 0.0]]).repeat(pcd_goal.shape[0], axis=0)
colors3 = np.array([[0.0, 1.0, 0.0]]).repeat(pcd_goal.shape[0], axis=0)
colors1[pcd_table_inds, :] = [0.0, 0.0, 1.0]
colors3[pcd_table_inds, :] = [0.0, 0.0, 1.0]
pcd_3.colors = open3d.utility.Vector3dVector(colors3)
pcd_1.colors = open3d.utility.Vector3dVector(colors1)
#open3d.visualization.draw_geometries([pcd_3])
open3d.visualization.draw_geometries([pcd_1, pcd_3])

# pcd_table = open3d.geometry.PointCloud()
# pcd_table.points = open3d.utility.Vector3dVector(np.concatenate(obs['table_pcd_pts'], axis=0))
pcd_table.colors = open3d.utility.Vector3dVector(np.concatenate(obs['table_pcd_colors'], axis=0) / 255.0)
# #open3d.visualization.draw_geometries([pcd_table])

# #pcd_table = pcd_table.uniform_down_sample(int(np.asarray(pcd_table.points).shape[0]/1000.0))

# # open3d.visualization.draw_geometries([pcd_table, pcd_1, pcd_3])

# table_kdtree = open3d.geometry.KDTreeFlann(pcd_table)

# table_contact_inds = []
# table_contact_pts = []
# table_contact_mask = np.zeros(np.asarray(pcd_table.points).shape[0], dtype=np.bool)

# for i, table_contact_pos in enumerate(pcd_goal[pcd_table_inds]):
#     #nearest_pt_ind = table_kdtree.search_knn_vector_3d(table_contact_pos, 1)[1][0]
#     nearest_pt_ind = table_kdtree.search_knn_vector_3d(table_contact_pos, 20)[1]    
#     nearest_pt = np.asarray(pcd_table.points)[nearest_pt_ind]
#     table_contact_pts.append(nearest_pt)
#     table_contact_inds.append(nearest_pt_ind)
#     table_contact_mask[nearest_pt_ind] = True

table_colors = np.asarray(pcd_table.colors)
table_colors[table_contact_inds, :] = [1.0, 0.0, 1.0]
pcd_table.colors = open3d.utility.Vector3dVector(table_colors)
open3d.visualization.draw_geometries([pcd_table])

# open3d.visualization.draw_geometries([pcd_1, pcd_3, pcd_table])
open3d.visualization.draw_geometries([pcd_1, pcd_table])