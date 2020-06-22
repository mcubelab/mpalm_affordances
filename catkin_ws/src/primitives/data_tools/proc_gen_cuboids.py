import trimesh
import numpy as np
import copy
from IPython import embed
import argparse
import pybullet as p
import os


class CuboidSampler(object):
    def __init__(self, stl_path, pb_client=None):
        self.nominal_cuboid = trimesh.load_mesh(stl_path)
        # self.nominal_cuboid.apply_scale(0.5)    
        self.max_extent = [200.0, 200.0, 200.0]
        self.min_extent = [50.0, 50.0, 50.0]

        self.length_vertex_pairs = [(0, 4), (1, 5), (3, 7), (2, 6)]
        self.width_vertex_pairs = [(1, 3), (0, 2), (5, 7), (4, 6)]
        self.height_vertex_pairs = [(1, 0), (3, 2), (5, 4), (7, 6)]

        self.length_ind = 0
        self.width_ind = 1
        self.height_ind = 2

        self.pb_client = pb_client

    def scale_axis(self, scale, vertices, axis='l'):
        assert(axis in ['l', 'w', 'h'])

        scaled_vertices = copy.deepcopy(vertices)
        if axis == 'l':
            vertex_pairs = self.length_vertex_pairs
            xyz_ind = self.length_ind
        elif axis == 'w':
            vertex_pairs = self.width_vertex_pairs
            xyz_ind = self.width_ind
        elif axis == 'h':
            vertex_pairs = self.height_vertex_pairs
            xyz_ind = self.height_ind

        current_dist = np.abs(
            vertices[vertex_pairs[0][0]][xyz_ind] - vertices[vertex_pairs[0][1]][xyz_ind])
        dist_to_add = current_dist*(scale - 1.0)/2.0

        for ind_pair in vertex_pairs:
            vert_0, vert_1 = vertices[ind_pair[0]], vertices[ind_pair[1]]

            min_vert = ind_pair[np.argmin([vert_0[xyz_ind], vert_1[xyz_ind]])]
            max_vert = ind_pair[np.argmax([vert_0[xyz_ind], vert_1[xyz_ind]])]

            scaled_vertices[min_vert][xyz_ind] -= dist_to_add
            scaled_vertices[max_vert][xyz_ind] += dist_to_add
        
        return scaled_vertices

    def clamp_vertices(self, vertices):
        for i in range(vertices.shape[0]):
            for j in range(len(self.max_extent)):
                max_extent = self.max_extent[j]/2.0
                min_extent = self.min_extent[j]/2.0

                if np.abs(vertices[i][j]) > max_extent:
                    vertices[i][j] = max_extent if vertices[i][j] > 0 else -max_extent
                if np.abs(vertices[i][j]) < min_extent:
                    vertices[i][j] = min_extent if vertices[i][j] > 0 else -min_extent
        
        return vertices

    def sample_cuboid(self, scale_list):
        """scale each axis of the cuboid by some amount
        
        Args:
            scale_list (list): [scale_l, scale_w, scale_h] values
        """
        new_cuboid = copy.deepcopy(self.nominal_cuboid)
        new_vertices = new_cuboid.vertices
        new_vertices = self.scale_axis(scale_list[0], new_vertices, 'l')
        new_vertices = self.scale_axis(scale_list[1], new_vertices, 'w')
        new_vertices = self.scale_axis(scale_list[2], new_vertices, 'h')

        # new_cuboid.vertices = self.clamp_vertices(new_vertices)
        new_cuboid.vertices = new_vertices

        return new_cuboid

    def sample_random_cuboid(self):
        scale = np.random.rand(3) * (1.5 - 0.3) + 0.3
        cuboid = self.sample_cuboid(scale.tolist())
        return cuboid

    def sample_random_cuboid_stl(self, fname):
        cuboid = trimesh.load_mesh(fname)
        return cuboid

    def sample_cuboid_pybullet(self, stl_file, keypoints=False, goal=False, 
                               yumi_robot_id=0, yumi_tip_id=12, table_id=27):
        mesh = self.sample_random_cuboid_stl(stl_file)
        
        sphere_ids = []
        # obj_scale = np.ones(3)*1.025
        # obj_scale = obj_scale.tolist()
        obj_scale = [1.025, 1.75, 1.025]
        
        if keypoints:
            obj_id = self.pb_client.load_geom(
                shape_type='mesh', 
                visualfile=stl_file, 
                collifile=stl_file, 
                mesh_scale=obj_scale,
                base_pos=[0.45, 0, np.abs(mesh.vertices[0][-1]*1.) + 0.01], 
                rgba=[0.7, 0.2, 0.2, 0.3],
                mass=0.035)            
            
            pts = []
            for i in range(8):
                pts.append(mesh.vertices[i]*1.) 
            
            for i in range(len(pts)):
                sphere = self.pb_client.load_geom(
                    shape_type='sphere', 
                    base_pos=[0.3, 0.0, 0.3], 
                    size=0.005, 
                    rgba=[0.9, 0.0, 0.0, 0.8],
                    mass=0.00001)
                sphere_ids.append(sphere)
                
                p.setCollisionFilterPair(yumi_robot_id, sphere, table_id, -1, enableCollision=False)
                p.setCollisionFilterPair(yumi_robot_id, sphere, yumi_tip_id, -1, enableCollision=False)
                for jnt_id in range(table_id):
                    p.setCollisionFilterPair(yumi_robot_id, sphere, jnt_id, -1, enableCollision=False)          
                p.setCollisionFilterPair(sphere, obj_id, -1, -1, enableCollision=False)
                
                p.createConstraint(
                    obj_id, -1, 
                    sphere, -1, 
                    p.JOINT_POINT2POINT, [0, 0, 0], 
                    parentFramePosition=pts[i], childFramePosition=[0, 0, 0])
        else:
            obj_id = self.pb_client.load_geom(
                shape_type='mesh', 
                visualfile=stl_file, 
                collifile=stl_file, 
                mesh_scale=obj_scale,
                base_pos=[0.45, 0, np.abs(mesh.vertices[0][-1]*1.) + 0.01], 
                rgba=[0.7, 0.2, 0.2, 1.0],
                mass=0.03)

        goal_obj_id = None            
        if goal:
            goal_obj_id = self.pb_client.load_geom(
                shape_type='mesh', 
                visualfile=stl_file, 
                collifile=stl_file, 
                mesh_scale=obj_scale,
                base_pos=[0.45, 0, np.abs(mesh.vertices[0][-1]*1.) + 0.01], 
                rgba=[0.1, 1.0, 0.1, 0.25],
                mass=0.03)
            p.setCollisionFilterPair(goal_obj_id, obj_id, -1, -1, enableCollision=False)
            if len(sphere_ids) > 0:
                for sphere_id in sphere_ids:
                    p.setCollisionFilterPair(sphere_id, goal_obj_id, -1, -1, enableCollision=False)

        return obj_id, sphere_ids, mesh, goal_obj_id

    def delete_cuboid(self, obj_id, goal_obj_id=None, keypoint_ids=None):
        if keypoint_ids is not None:
            if len(keypoint_ids) > 0:
                for kp_id in keypoint_ids:
                    self.pb_client.remove_body(kp_id)
        self.pb_client.remove_body(obj_id)
        if goal_obj_id is not None:
            self.pb_client.remove_body(goal_obj_id)


def main(args):
    # sampler = CuboidSampler('/home/anthony/Downloads/nominal_cuboid_half.stl')
    sampler = CuboidSampler('/root/catkin_ws/src/primitives/objects/cuboids/nominal_cuboid.stl')    
    # scene = trimesh.Scene()

    # cuboid = sampler.sample_cuboid([args.x, args.y, args.z])

    # cuboid_list = []
    # for i in range(10):
    #     scale = np.random.rand(3) * (5.0 - 1.0) + 1.0
    #     cuboid = sampler.sample_cuboid(scale.tolist())
    #     cuboid.visual.face_colors = [200, 200, 200, i/10.0*(200.0-10.0) + 10.0]
    #     cuboid_list.append(cuboid)
    
    # # scene.add_geometry([sampler.nominal_cuboid, new_cuboid])
    # scene.add_geometry(cuboid_list)    
    # scene.show()
    # embed()

    meshes_dir = '/root/catkin_ws/src/config/descriptions/meshes/'
    for i in range(5000):
        cuboid = sampler.sample_random_cuboid()
        cuboid.apply_scale(0.001)
        fname = os.path.join(meshes_dir, 'objects/cuboids/test_cuboid_smaller_%d.stl' % i)
        cuboid.export(fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', type=float, default=1.0)
    parser.add_argument('-y', type=float, default=1.0)
    parser.add_argument('-z', type=float, default=1.0)
    args = parser.parse_args()
    main(args)
