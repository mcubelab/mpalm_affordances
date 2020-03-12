import numpy as np
import sys, os
sys.path.append(os.environ['CODE_BASE'] + '/catkin_ws/src/config/src')
from helper import helper, roshelper
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from shapely.geometry import Polygon
from tf.transformations import quaternion_from_euler
import util
from stl import mesh
import os
import tf
import trimesh
from copy import deepcopy
import fcl
import rospy

request = fcl.CollisionRequest()
result = fcl.CollisionResult()

def initialize_collision_object(tris, verts):
    m = fcl.BVHModel()
    m.beginModel(len(verts), len(tris))
    m.addSubModel(verts, tris)
    m.endModel()
    t = fcl.Transform()
    return fcl.CollisionObject(m, t)

def is_collision(body1, body2):
    return fcl.collide(body1, body2, request, result)

class Body(object):
    def __init__(self, mesh_name):
        self.mesh = mesh.Mesh.from_file(os.environ["CODE_BASE"] + "/catkin_ws/src/" + mesh_name)
        self.trimesh = trimesh.load(os.environ["CODE_BASE"] + "/catkin_ws/src/" + mesh_name)
        
        # self.mesh = mesh.Mesh.from_file(
        #     mesh_name)
        # self.trimesh = trimesh.load(
        #     mesh_name)
        
        self.faces = list(self.mesh.vectors)
        self.normals = list(self.mesh.normals)

    def transform_object(self, pose):
        T = roshelper.matrix_from_pose(pose)
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
            points_elon = util.elongate_vector(proposal, 0.03)
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
        T_gripper_pose_world = roshelper.matrix_from_pose(pose_world)
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
        print("computing centroid")
        self.compute_centroid()
        print("initializing properties")
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

        stable_placement_list = []
        projected_point_list = []
        for face, normal in zip(self.mesh.vectors, self.mesh.normals):
            projected_point, dist = util.project_point2plane(self.centroid, normal, face)
            projected_point_list.append(projected_point)
            in_triangle = util.point_in_triangle(projected_point, face)
            if in_triangle:
                stable_placement_list.append(True)
            else:
                stable_placement_list.append(False)
        self.stable_placement_list = stable_placement_list
        self.projected_points = np.array(projected_point_list)
        self.eliminate_dual_placements()

    def define_polygon(self, vertices):
        polygon_shapely = Polygon(vertices)
        obj_centroid = np.array([polygon_shapely.centroid.x, polygon_shapely.centroid.y])
        polygon = Polygon(vertices - obj_centroid)
        polygon.vertices = vertices - obj_centroid
        polygon.obj_centroid = obj_centroid
        delaunay = Delaunay(polygon.vertices)
        polygon.triangles = polygon.vertices[delaunay.simplices]
        return polygon

    def compute_convex_hull(self, center_point, vector):
        point_list = []
        #1. find all faces that have same coordinates
        counter = 0
        face_list = []
        for face, normal in zip(self.faces, self.normals):
            point_projected_to_face, dist = util.project_point2plane(center_point, normal, face)
            if abs(dist) < 0.001:
                for i in range(3):
                    T = util.rotation_matrix(vector, np.array([0,0,-1]))
                    rotated_face = np.matmul(T[0:3, 0:3], face[i, :])
                    point_list.append(rotated_face[0:2])
                face_list.append(counter)
            counter += 1
        q = ConvexHull(point_list)
        q_points = []
        for index in q.vertices:
            q_points.append(q._points[index])
        q_points.sort(key=helper.angle_2d)
        return q_points

    def find_common_faces(self, center_point):
        #1. find all faces that have same coordinates
        counter = 0
        face_list = []
        for face, normal in zip(self.faces, self.normals):
            point_projected_to_face, dist = util.project_point2plane(center_point, normal, face)
            if abs(dist) < 0.001:
                face_list.append(counter)
            counter += 1
        return face_list

    def merge_faces(self, common_face_list):
        #1. find all faces that have same coordinates
        point_list = []
        for face in common_face_list:
            for point in self.faces[face]:
                point_list.append(point)
        return point_list

    def rotate_face(self, face, source_vec, target_vec=np.array([0,0,-1])):
        point_list = []
        for i in range(len(face)):
            T = helper.rotation_matrix(source_vec, target_vec)
            rotated_face = np.matmul(T[0:3, 0:3], face[i])
            point_list.append(rotated_face[0:3])
        return point_list

    def compute_planar_convex_hull(self, point_list):
        q = ConvexHull(point_list)
        q_points = []
        for index in q.vertices:
            q_points.append(q._points[index])
        q_points.sort(key=helper.angle_2d)
        return q_points

    def compute_convex_hulls(self, center_point, vector):
        common_faces = self.find_common_faces(center_point)
        merged_face = self.merge_faces(common_faces)
        rotated_face = self.rotate_face(merged_face, vector, target_vec=np.array([0,0,-1]))
        rotated_face_2d = helper.collapse_list_2d(rotated_face)
        convex_hull_planar = self.compute_planar_convex_hull(rotated_face_2d)
        convex_hull_planar_3d = helper.extend_list_3d(convex_hull_planar, element=rotated_face[0][2])
        convex_hull_3d = self.rotate_face(convex_hull_planar_3d, np.array([0,0,-1]), target_vec=vector)
        normal_convex_hull_3d = util.normal_from_points(convex_hull_3d, vector)
        return convex_hull_planar, convex_hull_3d, normal_convex_hull_3d

    def find_neighboor_faces(self, convex_face_id, convex_face_list):
        convex_face = convex_face_list[convex_face_id]
        neighboor_list = []
        common_points_list = []
        for counter, face in enumerate(convex_face_list):
            is_neighboor, common_points = self.is_common_point(convex_face, face)
            if is_neighboor and convex_face_id is not counter:
                neighboor_list.append(counter)
                common_points_list.append(common_points)
        return neighboor_list, common_points_list

    def is_common_point(self, point_list1, point_list2):
        common_points_list = []
        for i in point_list1:
            for j in point_list2:
                if np.allclose(i,j,atol=1e-2):
                    common_points_list.append(i)
        if len(common_points_list)==0:
            return False, []
        else:
            return True, common_points_list

    def _initialize_properties(self):
        self.volume = self.mesh.get_mass_properties()[0]

    def eliminate_dual_placements(self):
        for place_id, placement in enumerate(self.stable_placement_list):
            for point_id, point in enumerate(self.projected_points):
                if np.linalg.norm(point - self.projected_points[place_id]) < 0.001 and self.stable_placement_list[place_id] and self.stable_placement_list[point_id] and point_id != place_id:
                    self.stable_placement_list[place_id] = False

        self.stable_placement_dict = {}
        self.stable_placement_dict['face'] = []
        self.stable_placement_dict['id'] = []
        self.stable_placement_dict['vector'] = []
        self.stable_placement_dict['T'] = []
        self.stable_placement_dict['T_rot'] = []
        self.stable_placement_dict['mesh'] = []
        self.stable_placement_dict['convex_face_planar'] = []
        self.stable_placement_dict['convex_face_3d'] = []
        self.stable_placement_dict['normal_face_3d'] = []
        self.stable_placement_dict['convex_face_stable_config'] = []
        self.stable_placement_dict['normal_stable_config'] = []
        self.stable_placement_dict['neighboors'] = []
        self.stable_placement_dict['common_points'] = []
        self.stable_placement_dict['polygon'] = []
        self.stable_placement_dict['sides'] = []
        place_id = 0
        vector_base = np.array([0,0,-1])

        for counter, placement in enumerate(self.stable_placement_list):
            if placement:
                face = counter
                id = place_id
                vector = self.projected_points[counter] - self.centroid
                T_rot = helper.rotation_matrix(vector_base, vector).transpose()
                trans = tf.transformations.translation_from_matrix(T_rot)
                quat  = tf.transformations.quaternion_from_matrix(T_rot)
                pose = roshelper.list2pose_stamped(np.concatenate((trans, quat)))
                mesh_rotated, trimesh_rotated = self.transform_object(pose)
                trans[2] -= mesh_rotated.min_[2]
                T = deepcopy(T_rot)
                T[2,3] = trans[2]
                pose_object = roshelper.list2pose_stamped(np.concatenate((trans, quat)))
                mesh, trimesh_transformed = self.transform_object(pose_object)
                convex_face_planar, convex_face_3d, normal_hull_3d = \
                    self.compute_convex_hulls(self.projected_points[counter],
                                              self.projected_points[counter] - self.centroid)
                polygon = self.define_polygon(convex_face_planar)

                side_list = []
                for counter, vertice in enumerate(polygon.vertices):
                    if counter<len(polygon.vertices)-1:
                        side_list.append([vertice, polygon.vertices[counter+1]])
                    else:
                        side_list.append([vertice, polygon.vertices[0]])

                perimeter_list = []
                for side in side_list:
                    side_dict = {}
                    side_dict['center_b'] = (side[1] + side[0]) / 2
                    side_dict['center_3d_b'] = np.array([side_dict['center_b'][0], side_dict['center_b'][1], 0])
                    side_dict['length'] = np.linalg.norm(side[1] - side[0])
                    axis = (side[1] - side[0]) / side_dict['length']
                    side_dict['contact_x_b'] = np.array([axis[0], axis[1], 0])
                    side_dict['contact_z_b'] = np.array([0,0,-1])
                    side_dict['contact_y_b'] = np.cross(side_dict['contact_z_b'], side_dict['contact_x_b'])
                    if np.dot(side_dict['center_3d_b'], side_dict['contact_y_b']) < 0:
                        side_dict['contact_x_b'] = -np.array([axis[0], axis[1], 0])
                        side_dict['contact_y_b'] = np.cross(side_dict['contact_z_b'], side_dict['contact_x_b'])
                    side_dict['edge_frame_b'] = [side_dict['contact_x_b'],
                                                 side_dict['contact_y_b'],
                                                 side_dict['contact_z_b']]
                    x_vec = (side_dict['center_b'] - polygon.obj_centroid) / (np.linalg.norm(side_dict['center_b'] - polygon.obj_centroid))
                    z_vec = np.array([0,0,1])
                    y_vec = np.cross(z_vec, x_vec)
                    side_dict['contact_frame_b'] = [x_vec, y_vec, z_vec]
                    side_dict['contact_angle'] = float(helper.unwrap([helper.angle_2d(x_vec)- np.pi], 0, 2*np.pi))
                    side_dict['Ccb'] = helper.C3_2d(side_dict['contact_angle'])
                    vertices_rotated_list = []
                    for vertex in list(polygon.vertices):
                        vertices_rotated_list.append(np.matmul(side_dict['Ccb'].transpose(), vertex))
                    polygon_rotated = self.define_polygon(np.array(vertices_rotated_list))
                    side_dict['polygon_rotated'] = polygon_rotated
                    perimeter_list.append(side_dict)
                perimeter_list.sort(key=util.sort_angle)
                self.stable_placement_dict['face'].append(face)
                self.stable_placement_dict['id'].append(id)
                self.stable_placement_dict['vector'].append(vector)
                self.stable_placement_dict['T_rot'].append(T_rot)
                self.stable_placement_dict['T'].append(T)
                self.stable_placement_dict['mesh'].append(mesh)
                self.stable_placement_dict['convex_face_planar'].append(convex_face_planar)
                self.stable_placement_dict['convex_face_3d'].append(convex_face_3d)
                self.stable_placement_dict['normal_face_3d'].append(normal_hull_3d)
                self.stable_placement_dict['polygon'].append(polygon)
                self.stable_placement_dict['sides'].append(perimeter_list)
            place_id += 1
        #~ find neighboors of each stable placements (levering primitive)
        for convex_face_id, convex_face in enumerate(self.stable_placement_dict['convex_face_3d']):
            neighboor_list, common_points_list = self.find_neighboor_faces(convex_face_id, self.stable_placement_dict['convex_face_3d'])
            self.stable_placement_dict['neighboors'].append(neighboor_list)
            self.stable_placement_dict['common_points'].append(common_points_list)
            T = self.stable_placement_dict['T'][convex_face_id]
            T_rot = self.stable_placement_dict['T_rot'][convex_face_id]
            face_rotated_list = []
            normal_rotated_list = []
            for convex_face_id_2, convex_face_2 in enumerate(self.stable_placement_dict['convex_face_3d']):
                face_stable_config = util.transform_point_list(convex_face_2, T)
                normal_stable_config = self.stable_placement_dict['vector'][convex_face_id_2]
                normal_new = helper.vector_transform(T_rot, normal_stable_config / np.linalg.norm(normal_stable_config))
                face_rotated_list.append(face_stable_config)
                normal_rotated_list.append(normal_new)
            self.stable_placement_dict['convex_face_stable_config'].append(face_rotated_list)
            self.stable_placement_dict['normal_stable_config'].append(normal_rotated_list)

        # frank hack: reorder placements such that [0] element is identity orientation
        for counter, T in enumerate(self.stable_placement_dict['T']):
            # frank hack: make sure that nominal placement has identity orientation
            if T[0, 0] == 1 and T[1, 1] == 1 and T[2, 2] == 1:
                nominal_placement = counter
                break
            else:
                nominal_placement = 0

        for key in self.stable_placement_dict.keys():
            first_placement_tmp = self.stable_placement_dict[key][0]
            self.stable_placement_dict[key][0] = self.stable_placement_dict[key][nominal_placement]
            self.stable_placement_dict[key][nominal_placement] = first_placement_tmp

class CollisionBody(Body):
    def __init__(self, mesh_name = "config/descriptions/meshes/yumi/table_top_collision.stl"):
        super(CollisionBody, self).__init__(mesh_name)
        self.collision_object = initialize_collision_object(self.trimesh.faces, self.trimesh.vertices)

