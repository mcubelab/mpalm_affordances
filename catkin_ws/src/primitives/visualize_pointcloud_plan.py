import os
import os.path as osp
import sys
import time
import argparse
import numpy as np
import rospy
import signal
import threading
import pickle
import open3d
import trimesh
import copy
from IPython import embed

from airobot import Robot
from airobot.utils import common
import pybullet as p

from helper import util
from macro_actions import ClosedLoopMacroActions
from closed_loop_eval import SingleArmPrimitives, DualArmPrimitives

# from closed_loop_experiments_cfg import get_cfg_defaults
from multistep_planning_eval_cfg import get_cfg_defaults
from data_tools.proc_gen_cuboids import CuboidSampler
from data_gen_utils import YumiCamsGS, DataManager, MultiBlockManager, GoalVisual
import simulation
from helper import registration as reg
from helper.pointcloud_planning import PointCloudTree
from helper.pointcloud_planning_utils import PointCloudNode
from eval_utils.visualization_tools import PCDVis, PalmVis

from io import BytesIO
from PIL import Image


def downsample_scene_pcds(viz_data, scene, final=True):
    new_data = viz_data
    # voxel=0.0075
    voxel = 0.0075 if final else 0.015
    pcd_full = open3d.geometry.PointCloud()
    pcd_full.points = open3d.utility.Vector3dVector(new_data['object_pointcloud'])
    pcd_down = pcd_full.voxel_down_sample(voxel)
            
    obj_pcd_1 = trimesh.PointCloud(np.asarray(pcd_down.points))
    num_pts = obj_pcd_1.vertices.shape[0]
    colors = np.zeros((num_pts, 4), dtype=np.int64)
    colors[:, 0] = 255 #128
    colors[:, -1] = 255 if final else 5
    obj_pcd_1.colors = colors
    colors1 = colors

    obj_pcd_2 = trimesh.PointCloud(np.asarray(pcd_down.points))
    obj_pcd_2.apply_transform(util.matrix_from_pose(util.list2pose_stamped(new_data['transformation'])))
    num_pts = obj_pcd_2.vertices.shape[0]
    colors = np.zeros((num_pts, 4), dtype=np.int64)
    # colors[:, 0] = 255 #128
    # colors[:, 2] = 255 #128
    colors[:, 1] = 255
    colors[:, -1] = 255 if final else 5
    obj_pcd_2.colors = colors
    colors2 = colors

    keys = dict(scene.geometry).keys()
    new_scene = trimesh.Scene()
    for key in keys:
        if 'geometry' in key:
            continue
        new_scene.add_geometry(scene.geometry[key])

    new_scene.add_geometry(obj_pcd_1)
    new_scene.add_geometry(obj_pcd_2)
    return new_scene

def main():

    cfg = get_cfg_defaults()
    cfg.freeze()

    with open('data/planning/test_buffer_1.pkl', 'rb') as f:
        buffers = pickle.load(f)

    # preprocess to make looping easier
    buffers['final'] = [buffers['final']]
    buffer_keys = buffers.keys()
    buffers[buffer_keys[-2]] = buffers['final']
    buffers.pop('final')
    
    
    palm_mesh_file = osp.join(os.environ['CODE_BASE'], cfg.PALM_MESH_FILE)
    table_mesh_file = osp.join(os.environ['CODE_BASE'], cfg.TABLE_MESH_FILE)
    viz_palms = PalmVis(palm_mesh_file, table_mesh_file, cfg)
    viz_pcd = PCDVis()

    skeleton = [None, 'pull', 'grasp', 'pull', 'pull']
    buffer_scenes = {}

    cam_transform = np.array(
        [[-0.15366573, -0.4393257 ,  0.88508744,  1.36359307],
        [ 0.98636458, -0.01478718,  0.16390928,  0.26688281],
        [-0.05892161,  0.89820614,  0.43560759,  0.63174457],
        [ 0.        ,  0.        ,  0.        ,  1.        ]]
    )

    cam_transform_2 = np.array(
        [[-1.74800393e-02,  6.41618266e-01, -7.66824914e-01, -6.18572483e-01],
        [-9.99627565e-01, -2.72897256e-02, -4.70215869e-05,  6.45023975e-03],
        [-2.09566114e-02,  7.66538499e-01,  6.41856331e-01,  7.48762274e-01],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]        
    )

    # good_camera_euler = [1.0513555,  -0.02236318, -1.62958927]
    good_camera_euler = [1.0513555,  -0.02236318, np.deg2rad(40)]


    inc = 0
    real_plan = []
    parent = buffers[buffers.keys()[-1]][0].parent
    while parent is not None:
        real_plan.append(parent)
        node = buffers[buffers.keys()[parent[0]]][parent[1]]
        parent = node.parent
    real_plan.reverse()
    real_plan.append(real_plan[-1])
    print(real_plan)
    # for i, buffer_slot in enumerate(buffers.keys()):
    #     if i == 0:
    #         continue
    #     primitive_name = skeleton[i]
    #     buffer_scenes[buffer_slot] = []

    #     # prev_state = buffers[buffers.keys()[0]][0]
    #     for j in range(len(buffers[buffer_slot])):
    #         new_state = buffers[buffer_slot][j]
    #         parent = new_state.parent
    #         prev_state = buffers[buffers.keys()[parent[0]]][parent[1]]

    #         viz_data = {}
    #         viz_data['contact_world_frame_right'] = new_state.palms_raw[:7]
    #         if primitive_name == 'grasp':
    #             viz_data['contact_world_frame_left'] = new_state.palms_raw[7:]
    #         else:
    #             viz_data['contact_world_frame_left'] = new_state.palms_raw[:7]
    #         viz_data['transformation'] = util.pose_stamped2np(util.pose_from_matrix(new_state.transformation))
    #         viz_data['object_pointcloud'] = prev_state.pointcloud_full
    #         viz_data['start'] = prev_state.pointcloud_full

    #         scene = viz_palms.vis_palms_pcd(viz_data, world=True, corr=True, full_path=True, show_mask=False, goal_number=1)
    #         keys = dict(scene.geometry).keys()
    #         for key in keys:
    #             if isinstance(scene.geometry[key], trimesh.base.Trimesh):
    #                 colors = scene.geometry[key].visual.face_colors
    #                 colors[:, -1] = 255
    #                 scene.geometry[key].visual.face_colors = colors            
    #             elif isinstance(scene.geometry[key], trimesh.points.PointCloud):
    #                 colors = scene.geometry[key].colors
    #                 colors[:, -1] = 255
    #                 scene.geometry[key].colors= colors            

    #         scene = downsample_scene_pcds(viz_data, scene)
    #         scene.set_camera(angles=good_camera_euler, center=[0.4, 0.0, 0.2], distance=2.0)
    #         # img = scene.save_image(resolution=(1920,1080))
    #         # rendered = Image.open(BytesIO(img)).convert("RGB")
    #         # fname = 'data/planning/pcd_plan_rendered_%d.jpg' % inc
    #         # inc += 1
    #         # rendered.save(fname)

    #         # scene_pcd.show()
    #         buffer_scenes[buffer_slot].append(scene)

    real_short_plans = []
    for i in range(len(buffers.keys())):
        for j in range(len(buffers[buffers.keys()[-i-1]])):
            print(-i-1, j)
            node = buffers[buffers.keys()[-i-1]][j]
            final_node = node
            parent = node.parent
            plan = []
            while parent is not None:
                plan.append(parent)
                node = buffers[buffers.keys()[parent[0]]][parent[1]]
                parent = node.parent
            plan.reverse()
            real_short_plans.append((plan, final_node))

    short_plan_scenes = []
    for i in range(len(real_short_plans)):
        short_plan = real_short_plans[i][0]
        final_node = real_short_plans[i][1]
        plan_scenes = []
        final = False
        for j, parent in enumerate(short_plan):
            primitive_name = skeleton[j+1]
            
            prev_state = buffers[buffers.keys()[parent[0]]][parent[1]]
            if j < len(short_plan) - 1:
                print('skipping ahead')
                new_parent = short_plan[j+1]
                new_state = buffers[buffers.keys()[new_parent[0]]][new_parent[1]]
            else:
                final = True
                print('final node')
                new_state = final_node

            viz_data = {}
            viz_data['contact_world_frame_right'] = new_state.palms_raw[:7]
            if primitive_name == 'grasp':
                viz_data['contact_world_frame_left'] = new_state.palms_raw[7:]
            else:
                viz_data['contact_world_frame_left'] = new_state.palms_raw[:7]
            viz_data['transformation'] = util.pose_stamped2np(util.pose_from_matrix(new_state.transformation))
            viz_data['object_pointcloud'] = prev_state.pointcloud_full
            viz_data['start'] = prev_state.pointcloud_full

            scene = viz_palms.vis_palms_pcd(viz_data, world=True, corr=True, full_path=True, show_mask=False, goal_number=1)
            keys = dict(scene.geometry).keys()
            for key in keys:
                if isinstance(scene.geometry[key], trimesh.base.Trimesh):
                    colors = scene.geometry[key].visual.face_colors
                    colors[:, -1] = 200 if final else 5
                    scene.geometry[key].visual.face_colors = colors            
                elif isinstance(scene.geometry[key], trimesh.points.PointCloud):
                    colors = scene.geometry[key].colors
                    colors[:, -1] = 200 if final else 5
                    scene.geometry[key].colors= colors            

            scene = downsample_scene_pcds(viz_data, scene, final)
            scene.set_camera(angles=good_camera_euler, center=[0.4, 0.0, 0.2], distance=2.0)

            # scene.show()
            plan_scenes.append(scene)

            scene_full = trimesh.scene.scene.append_scenes(plan_scenes)
            # scene_full.set_camera(angles=good_camera_euler, center=[0.4, 0.0, 0.2], distance=2.0)
            scene_full.set_camera(angles=good_camera_euler, center=[0.4, 0.0, 0.2], distance=1.0)

            # if (i == len(buffer_scenes.values()) - 1):
            # if ((j == real_plan[i+1][1]) and (i >= len(buffer_scenes.values()) - 2)) or (i == len(buffer_scenes.values()) - 1):             
            img = scene_full.save_image(resolution=(1920,1080))
            rendered = Image.open(BytesIO(img)).convert("RGB")
            fname = 'data/planning/pcd_plan_rendered_short_plan_CAM1_%d_%d.jpg' % (i, j)
            rendered.save(fname)            
            # scene_full.show()            



    # flat_scenes = []
    # inc2 = 0
    # for i, l in enumerate(buffer_scenes.values()):
    #     for j, s in enumerate(l):
    #         if (j == real_plan[i+1][1]) or (i == len(buffer_scenes.values()) - 1):
    #         # if ((j == real_plan[i+1][1]) and (i >= len(buffer_scenes.values()) - 2)) or (i == len(buffer_scenes.values()) - 1):             
    #         # if (i == len(buffer_scenes.values()) - 1):            
    #             print('real plan')
    #             scene = s
    #             keys = dict(scene.geometry).keys()
    #             for key in keys:
    #                 if isinstance(scene.geometry[key], trimesh.base.Trimesh):
    #                     colors = scene.geometry[key].visual.face_colors
    #                     colors[:, -1] = 255
    #                     scene.geometry[key].visual.face_colors = colors            
    #                 elif isinstance(scene.geometry[key], trimesh.points.PointCloud):
    #                     colors = scene.geometry[key].colors
    #                     colors[:, -1] = 255
    #                     scene.geometry[key].colors= colors                   
    #         flat_scenes.append(s)
    #         scene_full = trimesh.scene.scene.append_scenes(flat_scenes)
    #         scene_full.set_camera(angles=good_camera_euler, center=[0.4, 0.0, 0.2], distance=2.0)

    #         # if (i == len(buffer_scenes.values()) - 1):
    #         # if ((j == real_plan[i+1][1]) and (i >= len(buffer_scenes.values()) - 2)) or (i == len(buffer_scenes.values()) - 1):             
    #         img = scene_full.save_image(resolution=(1920,1080))
    #         rendered = Image.open(BytesIO(img)).convert("RGB")
    #         fname = 'data/planning/pcd_plan_rendered_full_alpha_%d.jpg' % inc2
    #         inc2 += 1
    #         # rendered.save(fname)

    keys = dict(scene.geometry).keys()
    new_scene = trimesh.Scene()
    for key in keys:
        if 'geometry_3' in key or 'table' in key:
            new_scene.add_geometry(scene.geometry[key])
        pass
    new_scene.set_camera(angles=good_camera_euler, center=[0.4, 0.0, 0.2], distance=1.0)
    img = new_scene.save_image(resolution=(1920,1080))
    rendered = Image.open(BytesIO(img)).convert("RGB")
    fname = 'data/planning/pcd_plan_rendered_CAM1_init_pcd.jpg'
    rendered.save(fname)   


    viz_data['start'] = buffers[0][0].pointcloud_full
    viz_data['object_pointcloud'] = buffers[0][0].pointcloud_full    
    viz_data['transformation'] = util.pose_stamped2np(util.pose_from_matrix(buffers[3][0].transformation_so_far))
    final_scene = downsample_scene_pcds(viz_data, new_scene)
    final_scene.set_camera(angles=good_camera_euler, center=[0.4, 0.0, 0.2], distance=1.0)
    # final_scene.show()
    img = final_scene.save_image(resolution=(1920,1080))
    rendered = Image.open(BytesIO(img)).convert("RGB")
    fname = 'data/planning/pcd_plan_rendered_CAM1_final2_pcd.jpg'
    rendered.save(fname)       


    # good_camera_euler = [1.0513555,  -0.02236318, np.deg2rad(40)]
    # scene_full.set_camera(angles=good_camera_euler, center=[0.4, 0.0, 0.2], distance=1.0)
    scene_full.show()

    embed()    

if __name__ == "__main__":
    main()