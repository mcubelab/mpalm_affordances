import os, os.path as osp
import numpy as np
import trimesh
import open3d
import cv2
from PIL import Image
from io import BytesIO
import plotly.graph_objects as go
import threading

from airobot import set_log_level, log_debug, log_info, log_warn, log_critical

from rpo_planning.utils import common as util 
from rpo_planning.utils.visualize import PalmVis
from rpo_planning.utils.planning.pointcloud_plan import PointCloudNode


class FloatingPalmPlanVisualizer:
    """
    Class for helping visualize plans found by the RPO planner, without
    the whole robot. Meant to show the full interpolated motion of the
    plans with in the context of the scene point cloud that was observed.

    Attributes:
        palm_mesh_file (str): Path to .stl file for palm mesh
        table_mesh_file (str): Path to .stl file for table mesh
        cfg (YACS CfgNode): Config variables
        skill_dict (dict): Keys are the names of the skills,
            values are the rpo_planning.utils.planning.skill objects,
            which contain the get_nominal_plan method.
        palm_visualizer (rpo_planning.utils.visualize.PalmVis): Interface
            to set of utils for obtaining trimesh scene visualizations of
            individual snapshots in time with the object point cloud and 
            palms meshes
    """
    def __init__(self, palm_mesh_file, table_mesh_file, cfg, 
                 skill_dict, environment_mesh_file=None):
        self.palm_mesh_file = palm_mesh_file
        self.table_mesh_file = table_mesh_file
        self.cfg = cfg
        self.skill_dict = skill_dict
        self.environment_mesh_file = environment_mesh_file
        self.palm_visualizer = PalmVis(
            palm_mesh_file=self.palm_mesh_file,
            table_mesh_file=self.table_mesh_file,
            cfg=self.cfg
        )
        self._setup_visual()
        self.background_ready = True
    
    def _setup_visual(self):
        self.good_camera_euler = [1.0513555,  -0.02236318, np.deg2rad(40)]

        # markers for point clouds
        self.black_marker = {
            'size': 1.5,
            'color': 'black',
            'colorscale': 'Viridis',
            'opacity': 0.3
        }

        self.red_marker = {
            'size': 1.5,
            'color': 'red',
            'colorscale': 'Viridis',
            'opacity': 0.8
        }

        # ground (table) plane
        self.plane_data = {
            'type': 'mesh3d',
            'x': [-1, 1, 1, -1],
            'y': [-1, -1, 1, 1],
            'z': [0, 0, 0, 0],
            'color': 'gray',
            'opacity': 0.5,
            'delaunayaxis': 'z'}
        

    def _get_dense_plan(self, skeleton, raw_plan):
        """
        Takes raw plan obtained by RPO Planner, consisting of PointCloudNodes
        and uses the low-level primitive skill planners to get a dense sequence of
        configurations of the point cloud object

        Args:
            skeleton (list): Each element is a string representing what sequence
                of skills was used.
            raw_plan (list): Plan found by RPO planner, containing PointCloudNode
                objects representing the states that are reached, but which the robot
                must reach in sequence
        """
        dense_plan = []
        current_dense_node = raw_plan[0]
        for i, skill_name in enumerate(skeleton):
            # obtain the relevant getter function for obtaining the primitive plan
            get_fn = self.skill_dict[skill_name].get_nominal_plan
            state = raw_plan[i+1]

            # create argument for low-level primitive motion planner
            plan_args = {}
            if 'grasp' in skill_name:
                plan_args['palm_pose_l_world'] = util.list2pose_stamped(state.palms[7:].tolist())
            else:
                # just copying the right to the left, cause it's not being used anyways
                plan_args['palm_pose_l_world'] = util.list2pose_stamped(state.palms[:7].tolist())
            plan_args['palm_pose_r_world'] = util.list2pose_stamped(state.palms[:7].tolist())
            plan_args['transformation'] = util.pose_from_matrix(state.transformation)
            if 'grasp' in skill_name:
                plan_args['N'] = 60
            else:
                plan_args['N'] = self.skill_dict[skill_name].calc_n(state.transformation[0, -1],
                                            state.transformation[1, -1])
            
            # obtain the dense plan and then loop through it
            _step_dense_plan = get_fn(plan_args)
            for step_dense_plan in _step_dense_plan:
                for j in range(len(step_dense_plan['palm_poses_world']) - 2):
                    # get transformation between dense steps
                    step_transformation = util.get_transform(
                        step_dense_plan['palm_poses_world'][j+1][1],
                        step_dense_plan['palm_poses_world'][j][1])
                    step_transformation = util.matrix_from_pose(step_transformation)
                    
                    # convert the palm poses to numpy arrays for initializing the PointCloudNode
                    palms_np = np.concatenate((
                        util.pose_stamped2np(step_dense_plan['palm_poses_world'][j][1]),
                        util.pose_stamped2np(step_dense_plan['palm_poses_world'][j][0])), axis=0)
                    
                    # Initilize a point cloud node for each dense step, using the transformation and palms from the dense primitive plan
                    new_dense_node = PointCloudNode()
                    new_dense_node.init_state(current_dense_node, step_transformation, skill=skill_name)
                    dual = False if 'grasp' not in skill_name else True
                    new_dense_node.init_palms(palms_np,
                                            correction=True,
                                            prev_pointcloud=current_dense_node.pointcloud_full,
                                            dual=dual)

                    # append to global PointCloudNode plan
                    dense_plan.append(new_dense_node)
                    current_dense_node = new_dense_node
        return dense_plan
        
    def plotly_get_scene_data(self, object_pcd, scene_pcd, r_palm_mesh, l_palm_mesh):
        # object
        object_pcd_data = {
            'type': 'scatter3d',
            'x': object_pcd[:, 0],
            'y': object_pcd[:, 1],
            'z': object_pcd[:, 2],
            'mode': 'markers',
            'marker': self.black_marker
        }        

        # full scene
        scene_pcd_data = {
            'type': 'scatter3d',
            'x': scene_pcd[:, 0],
            'y': scene_pcd[:, 1],
            'z': scene_pcd[:, 2],
            'mode': 'markers',
            'marker': self.black_marker
        }        

        # palms
        both_palm_mesh_data = []
        for mesh in [r_palm_mesh, l_palm_mesh]:
            # tigo, tjgo, tkgo = np.asarray(mesh.triangles).T
            # xgo, ygo, zgo = np.asarray(mesh.vertices).T
            # o3d_mesh = mesh.as_open3d
            o3d_mesh = open3d.geometry.TriangleMesh()
            o3d_mesh.vertices=open3d.utility.Vector3dVector(mesh.vertices)
            o3d_mesh.triangles=open3d.utility.Vector3iVector(mesh.faces)

            tigo, tjgo, tkgo = np.asarray(o3d_mesh.triangles).T
            xgo, ygo, zgo = np.asarray(o3d_mesh.vertices).T
            palm_mesh_data = {
                'type': 'mesh3d',
                'x': xgo,
                'y': ygo,
                'z': zgo,
                'i': tigo,
                'j': tjgo,
                'k': tkgo
            }
            both_palm_mesh_data.append(palm_mesh_data)


        return object_pcd_data, scene_pcd_data, both_palm_mesh_data

    def render_plan(self, skeleton, plan, scene_pcd, video_fname, fps=10, return_frames=False):
        """
        Takes a plan obtained by the RPO planner and renders how the
        palms and the object point cloud are expected to move, in
        the context of the full scene point cloud

        Args:
            skeleton (list): Each element is a string representing what sequence
                of skills was used.
            plan (list): Plan found by RPO planner, containing PointCloudNode
                objects representing the states reached.
            scene_pcd (np.ndarray): Full scene point cloud
            video_fname (str): Name of the video file that should be written
            fps (int): Frames per second to use for this video
            return_frames (bool): If True, return the sequence of frames
        """
        # create dense plan that interpolates between the nodes in the raw plan
        dense_plan = self._get_dense_plan(skeleton, plan)

        rgb_frames = []

        # for each step in the dense plan
        iteration = 0
        for i, config in enumerate(dense_plan):
        # for i in range(1, len(dense_plan) - 1):
            # config = dense_plan[i-1]
            iteration += 1
            print('it: %d' % iteration)
            if i == len(dense_plan) - 1:
                continue
            pcd_data = {}
            # pcd_data['start'] = config.pointcloud_full
            # pcd_data['object_pointcloud'] = config.pointcloud_full
            pcd_data['start'] = config.pointcloud
            pcd_data['object_pointcloud'] = config.pointcloud
            pcd_data['transformation'] = np.asarray(util.pose_stamped2list(util.pose_from_matrix(dense_plan[i+1].transformation)))
            pcd_data['contact_world_frame_right'] = np.asarray(dense_plan[i+1].palms[:7])
            if 'pull' in config.skill:
                pcd_data['contact_world_frame_left'] = np.asarray(dense_plan[i+1].palms[:7])
            else:
                pcd_data['contact_world_frame_left'] = np.asarray(dense_plan[i+1].palms[7:])

            # get trimesh scene
            scene, scene_list = self.palm_visualizer.vis_palms_pcd(
                pcd_data, world=True, centered=False, corr=False, return_scene_list=True)
            obj_pcd, table_mesh, r_palm_meshes, l_palm_meshes, goal_obj_pcds = scene_list

            object_pcd_data, scene_pcd_data, both_palm_mesh_data = self.plotly_get_scene_data(
                np.asarray(obj_pcd.vertices), scene_pcd, r_palm_meshes, l_palm_meshes
            )

            # create a 3D plotly scene
            fig_data = [object_pcd_data, scene_pcd_data] + both_palm_mesh_data
            fig = go.Figure(fig_data)
            camera = {
                'up': {'x': 0, 'y': 0,'z': 1},
                'center': {'x': 0.45, 'y': 0, 'z': 0.0},
                'eye': {'x': 1.5, 'y': 0.0, 'z': 0.5}
            }
            scene = {
                'xaxis': {'nticks': 10, 'range': [-0.1, 0.9]},
                'yaxis': {'nticks': 16, 'range': [-0.6, 0.6]},
                'zaxis': {'nticks': 8, 'range': [-0.01, 1.0]}
            }
            width = 700
            margin = {'r': 20, 'l': 10, 'b': 10, 't': 10}
            fig.update_layout(
                scene=scene,
                scene_camera=camera,
                width=width,
                margin=margin
            )
            
            # save the img
            # fig.write_html('test_html.html')
            # from IPython import embed
            # embed()
            img = fig.to_image()            

            rendered = Image.open(BytesIO(img)).convert("RGB")
            np_img = np.array(rendered)
            rgb_frames.append(np_img)

            # plt.imshow(np_img)
            # plt.show()
            # fname = 'data/planning/pcd_plan_rendered_short_plan_CAM1_%d_%d.jpg' % (i, j)
            # rendered.save(fname)     

            # write the sequence of imgs to video/gif
        print("got images, ready for video")
        size = rgb_frames[0].shape[:2]
        rgb_video = cv2.VideoWriter(
            'test_video.avi', cv2.VideoWriter_fourcc(*'mp4v'), 10, (size[1], size[0]))
        for i, frame in enumerate(rgb_frames):
            rgb_video.write(frame[:, :, ::-1])  # convert from rgb to bgr
        rgb_video.release()

        # return the video
        if return_frames:
            return rgb_frames
        else:
            return None

    def render_plan_background(self, skeleton, plan, scene_pcd, video_fname):
        """
        Takes a plan obtained by the RPO planner and renders how the
        palms and the object point cloud are expected to move, in
        the context of the full scene point cloud

        Args:
            skeleton (list): Each element is a string representing what sequence
                of skills was used.
            plan (list): Plan found by RPO planner, containing PointCloudNode
                objects representing the states reached.
            scene_pcd (np.ndarray): Full scene point cloud
            video_fname (str): Name of the video file that should be written
            fps (int): Frames per second to use for this video
        """
        # create dense plan that interpolates between the nodes in the raw plan
        if not self.background_ready:
            log_warn('Point cloud plan render: Currently rendering another video, please wait until completed')
        else:
            dense_plan = self._get_dense_plan(skeleton, plan)

            p = threading.Thread(
                target=self._background_render,
                args=(dense_plan, scene_pcd, video_fname)
            )
            p.daemon = True
            p.start()

    def _background_render(self, dense_plan, scene_pcd, video_fname, fps=10):
        """
        Takes a plan obtained by the RPO planner and renders how the
        palms and the object point cloud are expected to move, in
        the context of the full scene point cloud. We render by creating a sequence
        of static 3D plots using plotly that contain the point cloud and mesh objects
        , convert these plots to images, and then stitch the image sequence.
        To be used as target function for background video processing

        Args:
            dense_plan (list): Each element contains PointCloudNode, this is the
                densely interpolated version of the raw plan found by the planner
            scene_pcd (np.ndarray): Full scene point cloud
            video_fname (str): Name of the video file that should be written
            fps (int): Frames per second to use for this video
        """
        self.background_ready = False
        rgb_frames = []
        iteration = 0
        for i, config in enumerate(dense_plan):
            iteration += 1
            if i == len(dense_plan) - 1:
                continue
            pcd_data = {}
            pcd_data['start'] = config.pointcloud
            pcd_data['object_pointcloud'] = config.pointcloud
            pcd_data['transformation'] = np.asarray(util.pose_stamped2list(util.pose_from_matrix(dense_plan[i+1].transformation)))
            pcd_data['contact_world_frame_right'] = np.asarray(dense_plan[i+1].palms[:7])
            if 'pull' in config.skill:
                pcd_data['contact_world_frame_left'] = np.asarray(dense_plan[i+1].palms[:7])
            else:
                pcd_data['contact_world_frame_left'] = np.asarray(dense_plan[i+1].palms[7:])

            # get trimesh scene
            scene, scene_list = self.palm_visualizer.vis_palms_pcd(
                pcd_data, world=True, centered=False, corr=False, return_scene_list=True)
            obj_pcd, table_mesh, r_palm_meshes, l_palm_meshes, goal_obj_pcds = scene_list

            object_pcd_data, scene_pcd_data, both_palm_mesh_data = self.plotly_get_scene_data(
                np.asarray(obj_pcd.vertices), scene_pcd, r_palm_meshes, l_palm_meshes
            )

            # create a 3D plotly scene
            fig_data = [object_pcd_data, scene_pcd_data] + both_palm_mesh_data
            fig = go.Figure(fig_data)
            camera = {
                'up': {'x': 0, 'y': 0,'z': 1},
                'center': {'x': 0.45, 'y': 0, 'z': 0.0},
                'eye': {'x': 1.5, 'y': 0.0, 'z': 0.5}
            }
            scene = {
                'xaxis': {'nticks': 10, 'range': [-0.1, 0.9]},
                'yaxis': {'nticks': 16, 'range': [-0.6, 0.6]},
                'zaxis': {'nticks': 8, 'range': [-0.01, 1.0]}
            }
            width = 700
            margin = {'r': 20, 'l': 10, 'b': 10, 't': 10}
            fig.update_layout(
                scene=scene,
                scene_camera=camera,
                width=width,
                margin=margin
            )
            
            # save the img
            # fig.write_html('test_html.html')
            img = fig.to_image()            

            rendered = Image.open(BytesIO(img)).convert("RGB")
            np_img = np.array(rendered)
            rgb_frames.append(np_img)

        # write the sequence of imgs to video/gif
        size = rgb_frames[0].shape[:2]
        rgb_video = cv2.VideoWriter(
            video_fname, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]))
        for i, frame in enumerate(rgb_frames):
            rgb_video.write(frame[:, :, ::-1])  # convert from rgb to bgr
        rgb_video.release()
        self.background_ready = True