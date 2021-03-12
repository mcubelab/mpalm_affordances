"""

    ***
    This is adapted from old code from the previous "Tactile Dexterity" project,
    we use the provided functionality for data generation in simulation
    ***

"""
import numpy as np
import sys, os
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from shapely.geometry import Polygon
from tf.transformations import quaternion_from_euler
from stl import mesh
import tf
import trimesh
from copy import deepcopy
import fcl
import rospy

from rpo_planning.utils import common as util
from rpo_planning.utils import helper
from rpo_planning.utils.planning.collision import initialize_collision_object, is_collision


class Body(object):
    def __init__(self, mesh_name):
        self.mesh = mesh.Mesh.from_file(mesh_name)
        self.trimesh = trimesh.load(mesh_name)

        self.faces = list(self.mesh.vectors)
        self.normals = list(self.mesh.normals)

    def transform_object(self, pose):
        T = util.matrix_from_pose(pose)
        euler = tf.transformations.euler_from_matrix(T.transpose(), 'rxyz')
        #mesh library performs rotatin with DCM rather than R matrix
        trans = tf.transformations.translation_from_matrix(T)
        mesh_new = deepcopy(self.mesh)
        mesh_new.rotate([.5, 0.0,0.0], euler[0])
        mesh_new.rotate([0.0, .5 ,0.0], euler[1])
        mesh_new.rotate([0.0, 0.0, .5], euler[2])
        mesh_new.x += trans[0]
        mesh_new.y += trans[1]
        mesh_new.z += trans[2]
        vertices = list(np.array(deepcopy(self.trimesh.vertices)))
        vertices_transformed = []
        for vertex in vertices:
            point_transformed = helper.vector_transform(T, vertex)
            vertices_transformed.append(point_transformed)

        return mesh_new, vertices_transformed

    def plot_meshes(self, mesh_list):
        # Optionally render the rotated cube faces
        from matplotlib import pyplot
        from mpl_toolkits import mplot3d
        # Create a new plot
        figure = pyplot.figure()
        axes = mplot3d.Axes3D(figure)
        # Render the cube
        for mesh in mesh_list:
            axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors))
            axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors))
        # Auto scale to the mesh size
        scale = mesh.points.flatten(-1)
        axes.auto_scale_xyz(scale, scale, scale)
        pyplot.xlabel('x')
        pyplot.ylabel('y')
        # Show the plot to the screen
        pyplot.show()

    def plot_proposals(self, mesh, proposals):
        # Optionally render the rotated cube faces
        from matplotlib import pyplot
        from mpl_toolkits import mplot3d
        # Create a new plot
        figure = pyplot.figure()
        axes = mplot3d.Axes3D(figure)
        #plot proposals
        for proposal in proposals:
            points_elon = helper.elongate_vector(proposal, 0.03)
            axes.plot([points_elon[0][0], points_elon[1][0]],
                      [points_elon[0][1], points_elon[1][1]],
                      [points_elon[0][2], points_elon[1][2]])
        # Render the cube
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors))
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors))
        # Auto scale to the mesh size
        scale = mesh.points.flatten(-1)
        axes.auto_scale_xyz(scale, scale, scale)
        pyplot.xlabel('x')
        pyplot.ylabel('y')
        # Show the plot to the screen
        pyplot.show()

    def setCollisionPose(self, collision_object, pose_world):
        T_gripper_pose_world = util.matrix_from_pose(pose_world)
        R = T_gripper_pose_world[0:3, 0:3]
        t = T_gripper_pose_world[0:3, 3]
        collision_object.setRotation(R)
        collision_object.setTranslation(t)
        self.collision_object = collision_object

class Object(Body):
    def __init__(self, mass = .1, mesh_name = "config/descriptions/meshes/objects/realsense_box_experiments.stl"):
        super(Object, self).__init__(mesh_name)
        self.mass = mass
        self.object_name = mesh_name
        self.compute_centroid()
        self._initialize_properties()

    def compute_centroid(self):
        x_total = 0
        y_total = 0
        z_total = 0

        for face in self.mesh.vectors:
            for counter in range(3):
                x_total +=face[counter][0]
                y_total +=face[counter][1]
                z_total +=face[counter][2]

        num_vertices = len(self.mesh.vectors) * len(self.mesh.vectors[0])
        self.centroid = np.zeros(3)
        self.centroid[0] = x_total / num_vertices
        self.centroid[1] = y_total / num_vertices
        self.centroid[2] = z_total / num_vertices

        self.t_mesh = trimesh.load_mesh(self.object_name)
        self.stable_poses_trimesh = self.t_mesh.compute_stable_poses()[0]
        self.centroid = self.t_mesh.center_mass

        self.stable_placement_list = None
        self.project_points = None
        self.finish_stable_placements_grasp()

    def _initialize_properties(self):
        self.volume = self.mesh.get_mass_properties()[0]

    def finish_stable_placements_grasp(self):
        self.stable_placement_dict = {}
        self.stable_placement_dict['face'] = []
        self.stable_placement_dict['id'] = []
        self.stable_placement_dict['T'] = []
        self.stable_placement_dict['T_rot'] = []

        place_id = 0
        vector_base = np.array([0,0,-1])

        for counter, pose in enumerate(self.stable_poses_trimesh):
            face = counter
            id = counter
            T = pose
            T_rot = T[:-1, :-1]

            self.stable_placement_dict['face'].append(face)
            self.stable_placement_dict['id'].append(id)
            self.stable_placement_dict['T_rot'].append(T_rot)
            self.stable_placement_dict['T'].append(T)

