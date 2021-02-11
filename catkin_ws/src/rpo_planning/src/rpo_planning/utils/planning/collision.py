import fcl
import os
import trimesh
import copy
import numpy as np

from rpo_planning.utils import common as util

request = fcl.CollisionRequest()
result = fcl.CollisionResult()


def initialize_collision_object(tris, verts):
    """
    Function to create an fcl collision object from the triangle mesh
    by specifying the vertices and triangular faces of the mesh

    Args:
        tris (list): List of triangle faces
        verts (list): List of 3D vertices

    Returns:
        fcl.CollisionObject: Object that can be used for collision
            queries with fcl library
    """
    m = fcl.BVHModel()
    m.beginModel(len(verts), len(tris))
    m.addSubModel(verts, tris)
    m.endModel()
    t = fcl.Transform()
    return fcl.CollisionObject(m, t)


def is_collision(body1, body2):
    """
    Function to check if two bodies are in collision, at a particular
    pose

    Args:
        body1 (CollisionBody): body 1
        body2 (CollisionBody): body 2

    Returns:
        bool: True if in collision, otherwise False
    """
    return fcl.collide(body1, body2, request, result)


class CollisionBody():
    def __init__(self, mesh_name):
        self.trimesh = trimesh.load(mesh_name)
        self.collision_object = initialize_collision_object(
            self.trimesh.faces,
            self.trimesh.vertices)

    def setCollisionPose(self, pose_world):
        """
        Function to set the world frame pose of the colllision object
        before querying whether or not this collides with another body

        Args:
            pose_world (rpo_planning.utils.common.PoseStamped): 6D pose of
                the object
        """
        T_gripper_pose_world = util.matrix_from_pose(pose_world)
        R = T_gripper_pose_world[:-1, :-1]
        t = T_gripper_pose_world[:-1, -1]
        self.collision_object.setRotation(R)
        self.collision_object.setTranslation(t)


class CheckCollisions():
    def __init__(self, gripper_name, table_name):
        self.gripper_left = CollisionBody(mesh_name=gripper_name)
        self.gripper_right = CollisionBody(mesh_name=gripper_name)
        self.table = CollisionBody(mesh_name=table_name)

    def check_collisions_with_table(self, gripper_pose, arm='l'):
        if arm == 'l':
            self.gripper_left.setCollisionPose(
                gripper_pose)
            _is_collision = is_collision(
                self.gripper_left.collision_object,
                self.table.collision_object)
        else:
            self.gripper_right.setCollisionPose(
                gripper_pose)
            _is_collision = is_collision(
                self.gripper_right.collision_object,
                self.table.collision_object)

        return _is_collision

    def avoid_collision(self, pose_gripper, arm="l", tol=0.006,
                        height_above_table=0.012, axis=[-1, 0, 0]):
        pose_gripper_safe = copy.deepcopy(pose_gripper)
        pose_gripper_safe.pose.position.z -= height_above_table
        while self.check_collisions_with_table(pose_gripper_safe, arm=arm):
            pose_gripper_safe = util.offset_local_pose(
                pose_gripper_safe,
                np.array(axis) * tol)
        pose_gripper = copy.deepcopy(pose_gripper_safe)
        pose_gripper.pose.position.z += height_above_table
        return pose_gripper
