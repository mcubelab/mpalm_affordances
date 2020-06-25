import open3d
import numpy as np
import copy
import time
import plotly.graph_objects as go
from airobot.utils import common


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
    # source_homog = np.hstack(
    #     (source, np.ones(source.shape[0], 1))
    # )

    source_transformed = np.matmul(transformation, source_homog.T).T[:, :-1]
    return source_transformed


def plot_registration_result(source, source_transformed, target):
    red_marker = {
        'size': 3.0,
        'color': 'red',                # set color to an array/list of desired values
        'colorscale': 'Viridis',   # choose a colorscale
        'opacity': 1.0
    }
    blue_marker = {
        'size': 3.0,
        'color': 'blue',                # set color to an array/list of desired values
        'colorscale': 'Viridis',   # choose a colorscale
        'opacity': 1.0
    }
    gray_marker = {
        'size': 0.8,
        'color': 'gray',                # set color to an array/list of desired values
        'colorscale': 'Viridis',   # choose a colorscale
        'opacity': 0.5
    }

    source_original_data = {
        'type': 'scatter3d',
        'x': source[:, 0],
        'y': source[:, 1],
        'z': source[:, 2],
        'mode': 'markers',
        'marker': blue_marker
    }

    source_transformed_data = {
        'type': 'scatter3d',
        'x': source_transformed[:, 0],
        'y': source_transformed[:, 1],
        'z': source_transformed[:, 2],
        'mode': 'markers',
        'marker': red_marker
    }

    target_data = {
        'type': 'scatter3d',
        'x': target[:, 0],
        'y': target[:, 1],
        'z': target[:, 2],
        'mode': 'markers',
        'marker': gray_marker
    }

    plane_data = {
        'type': 'mesh3d',
        'x': [-1, 1, 1, -1],
        'y': [-1, -1, 1, 1],
        'z': [0, 0, 0, 0],
        'color': 'gray',
        'opacity': 0.5,
        'delaunayaxis': 'z'
    }
    fig_data = []
    fig_data.append(source_original_data)
    fig_data.append(source_transformed_data)
    fig_data.append(target_data)

    fig = go.Figure(data=fig_data)
    camera = {
        'up': {'x': 0, 'y': 0,'z': 1},
        'center': {'x': 0.45, 'y': 0, 'z': 0.0},
        'eye': {'x': -1.0, 'y': 0.0, 'z': 0.01}
    }
    scene = {
        'xaxis': {'nticks': 10, 'range': [-0.1, 0.9]},
        'yaxis': {'nticks': 16, 'range': [-0.5, 0.5]},
        'zaxis': {'nticks': 8, 'range': [-0.01, 0.99]}
    }
    width = 700
    margin = {'r': 20, 'l': 10, 'b': 10, 't': 10}
    fig.update_layout(
        scene=scene,
        scene_camera=camera,
        width=width,
        margin=margin
    )
    return fig


def draw_registration_result(source, target, transformation):
    """
    Display point cloud geometries in a window after finding an
    alignment transformation from source to target

    Args:
        source (open3d.geometry.PointCloud): Source pointcloud, to be
            transformed
        target (open3d.geometry.PointCloud): Target pointcloud, that
            source should be aligned to
        transformation (np.ndarray): Homogeneous transformation matrix,
            shape: 4 X 4
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    open3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size,
                           radius_normal=None,
                           radius_feature=None):
    """Preprocess point cloud by applying a voxel-based downsampling,
    estimating the point cloud normals, and computing the FPFH features
    that are used by the RANSAC registration algorithm

    Args:
        pcd (open3d.geometry.PointCloud): The pointcloud to be preprocessed
        voxel_size (float): Voxel size to downsample into

    Returns:
        open3d.geometry.PointCloud: The processed pointcloud (voxelized and with
            normals estimated)
        open3d.registration.Feature: The FPFH feature for the pointcloud
    """
    # # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    if radius_normal is None:
        radius_normal = voxel_size * 2.0
    if radius_feature is None:
        radius_feature = voxel_size * 5.0

    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        open3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = open3d.registration.compute_fpfh_feature(
        pcd_down,
        open3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    """Global registration of the source pointcloud to the target point cloud,
    using the RANSAC registration algorithm based on feature matching between
    the target and source

    Args:
        source_down (open3d.geometry.PointCloud): Downsampled source pointcloud
        target_down (open3d.geometry.PointCloud): Downsampled target pointcloud
        source_fpfh (open3d.registration.Feature): FPFH feature descriptor of source
        target_fpfh (open3d.registration.Feature): FPFH feature descriptor of target
        voxel_size (float): Downsampling voxel size

    Returns:
        open3d.registration.RegistrationResult: Result of the RANSAC registration
    """
    distance_threshold = voxel_size * 1.5
    # distance_threshold = 0.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)

    estimation_method = open3d.registration.TransformationEstimationPointToPoint(False)
    ransac_n = 4
    correspondence_checkers = [
        open3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        open3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
    ]
    convergence_criteria = open3d.registration.RANSACConvergenceCriteria(4000000, 500)
    result = open3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        estimation_method, ransac_n, correspondence_checkers, convergence_criteria)
    return result


def refine_registration(source, target, init_trans, voxel_size):
    """
    Function to refine pointcloud registration result with a local ICP-based
    registration. Takes as input the result of a global registration attempt
    for initialization.

    Args:
        source (open3d.geometry.PointCloud): source pointcloud to be aligned
        target (open3d.geometry.PointCloud): target pointcloud to align to
        init_trans (np.ndarray): Result from global registration method,
            transformation used as intialization to ICP
        voxel_size (float): Voxel size that point cloud is downsampled to

    Result:
        open3d.registration.RegistrationResult: Result of ICP registration
    """
    distance_threshold = 0.05
    convergence_criteria = open3d.registration.ICPConvergenceCriteria(
        max_iteration=100,
        relative_fitness=0.0,
        relative_rmse=0.0
    )
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    result = open3d.registration.registration_icp(
        source, target, distance_threshold, init_trans,
        open3d.registration.TransformationEstimationPointToPoint(),
        convergence_criteria)
    return result


def init_grasp_trans(source_raw, fwd=True, inplace=True, target=None, pp=False):
    """Set up in initial guess for the homogeneous transformation
    that corresponds to the grasp primitive, translating everything
    to the origin and performing a pure forward reorientation

    Args:
        source_raw (np.ndarray): Source pointcloud
        fwd (bool, optional): If True, guess that we want to
            flip forward, else backward. Defaults to True.
    """
    trans_to_origin = np.mean(source_raw, axis=0)
    # trans_to_origin = np.mean(obj_raw, axis=0)

    if not pp:
        if fwd:
            init_rot = common.euler2rot([0, np.pi/2, 0], 'xyz')
        else:
            init_rot = common.euler2rot([0, -np.pi/2, 0], 'xyz')

        # translate the source to the origin
        T_0 = np.eye(4)
        T_0[:-1, -1] = -trans_to_origin

        # apply pure rotation in the world frame, based on prior knowledge that
        # grasping tends to pitch forward/backward
        T_1 = np.eye(4)
        T_1[:-1, :-1] = init_rot

        # translate in [x, y] back away from origin
        T_2 = np.eye(4)
        T_2[0, -1] = trans_to_origin[0]
        T_2[1, -1] = trans_to_origin[1]

        if target is not None and not inplace:
            # translate to target
            T_target = np.eye(4)
            trans_to_target = np.mean(target, axis=0)
            T_target[0, -1] = trans_to_target[0]
            T_target[1, -1] = trans_to_target[1]
            T_target[2, -1] = trans_to_target[2]
            init_trans = np.matmul(T_target, np.matmul(T_1, T_0))
        else:
            # compose transformations in correct order
            init_trans = np.matmul(T_2, np.matmul(T_1, T_0))
    else:
        if target is None:
            raise ValueError('Target must not be None if using Pick and Place!')

        T_0 = np.eye(4)
        T_0[0, -1] = -trans_to_origin[0]
        T_0[1, -1] = -trans_to_origin[1]

        # T_1 = np.eye(4)
        # T_1[:-1, :-1] = init_rot

        T_target = np.eye(4)
        trans_to_target = np.mean(target, axis=0)
        T_target[0, -1] = trans_to_target[0]
        T_target[1, -1] = trans_to_target[1]
        T_target[2, -1] = trans_to_target[2]

        init_trans = np.matmul(T_target, T_0)
    return init_trans


def full_registration_np(source_np, target_np, init_trans=None):
    """Run full global + local registration routine, using numpy array
    pointclouds as input

    Args:
        source_np (np.ndarray): Source pointcloud, numpy array
        target_np (np.ndarray): Target pointcloud, numpy array

    Returns:
        np.ndarray: Homogeneous transformation matrix result of registration
    """
    start_time = time.time()
    source_pcd = open3d.geometry.PointCloud()
    target_pcd = open3d.geometry.PointCloud()
    source_pcd.points = open3d.utility.Vector3dVector(source_np)
    target_pcd.points = open3d.utility.Vector3dVector(target_np)

    voxel_size = 0.01

    source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size,
                                                      radius_normal=1.0,
                                                      radius_feature=1.0)
    target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)

    if init_trans is None:
        result_ransac = execute_global_registration(
            source_down, target_down,
            source_fpfh, target_fpfh,
            voxel_size)
        init_trans = result_ransac.transformation

    result_icp = refine_registration(
        source_down, target_down,
        init_trans, voxel_size)
    # result_icp = refine_registration(
    #     source_pcd, target_down,
    #     init_trans, voxel_size)

    # print('Time taken for registration: ' + str(time.time() - start_time))
    return result_icp.transformation


if __name__ == "__main__":
    source_pcd = open3d.geometry.PointCloud()
    source_pcd.points = open3d.utility.Vector3dVector(np.asarray(pcd_1.points)[pcd_table_inds, :])
    source_down, source_fpfh = preprocess_point_cloud(source_pcd, 0.01)
    target_down, target_fpfh = preprocess_point_cloud(pcd_table, 0.01)

    result_ransac = execute_global_registration(
        source_down, target_down,
        source_fpfh, target_fpfh,
        0.025)

    draw_registration_result(source_down, target_down, result_ransac.transformation)

    source_pcd.estimate_normals()
    pcd_table.estimate_normals()
    result_icp = refine_registration(source_pcd, pcd_table, result_ransac, 0.01)

    init_trans = np.array([[ 2.22044605e-16, -0.00000000e+00, -1.00000000e+00,
            0.00000000e+00],
        [ 0.00000000e+00,  1.00000000e+00, -0.00000000e+00,
            0.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00,  2.22044605e-16,
            0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            1.00000000e+00]])
    voxel_size = 0.01
    distance_threshold = voxel_size * 0.4
    result_icp_2 = open3d.registration.registration_icp(
        source_pcd, pcd_table, distance_threshold, init_trans,
        open3d.registration.TransformationEstimationPointToPlane())

    # draw_registration_result(source_pcd, pcd_table, result_icp.transformation)
    draw_registration_result(source_pcd, pcd_table, result_icp_2.transformation)