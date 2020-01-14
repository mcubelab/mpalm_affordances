import sys, os
sys.path.append(os.environ['CODE_BASE'] + '/catkin_ws/src/config/src')
import itertools
import numpy as np
import rospy
from helper import helper
from helper import roshelper
from helper import visualize_helper
import util
from copy import deepcopy
import tf.transformations as tfm
from geometry_msgs.msg import PoseStamped
import objects

class LeverSampling(object):
    def __init__(self, sampler):
        self.sampler = sampler
        self.samples_dict = {}
        self.samples_dict['placement_id'] = []
        self.samples_dict['placement_end_id'] = []
        self.samples_dict['sample_ids'] = []
        self.samples_dict['rotation_points'] = []
        self.samples_dict['face_ids'] = []
        self.samples_dict['gripper_poses'] = []
        self.samples_dict['face_normals'] = []
        self.generate_lever_samples()

        self.samples_dict['object_pose'] = []
        self.compute_object_poses()

    def compute_object_poses(self):
        for placement_id in range(len(self.samples_dict['placement_id'])):
            T = self.sampler.object.stable_placement_dict['T'][placement_id]
            object_pose = roshelper.pose_from_matrix(T, frame_id='yumi_body')
            object_poses_list = []
            for i in range(len(self.samples_dict['sample_ids'][placement_id])):
                object_poses_list.append(object_pose)
            self.samples_dict['object_pose'].append(object_poses_list)

    def find_opposite_face(self, face_anchor_id, neighboors):
        #frank hack: this is only valid for boxes
        all_face_ids = range(len(self.sampler.object.stable_placement_dict['convex_face_3d']))
        helper.intersection(neighboors, all_face_ids)
        different_elements = np.setdiff1d(all_face_ids, neighboors + [face_anchor_id])
        assert (len(list(different_elements))==1, "more than one opposite face")
        return int(different_elements[0])

    def get_points_face(self, face_anchor_id, placement_id):
        neighboors_anchor_face = self.sampler.object.stable_placement_dict['neighboors'][face_anchor_id]
        face_rotate_id = self.find_opposite_face(face_anchor_id, neighboors_anchor_face)
        face_list_placement = self.sampler.object.stable_placement_dict['convex_face_stable_config'][
            placement_id]
        face_rotate_sampling_base = face_list_placement[face_rotate_id]
        face_anchor_sampling_base = face_list_placement[face_anchor_id]
        return face_rotate_sampling_base, face_anchor_sampling_base, face_rotate_id

    def get_face_normals(self, face_anchor_id, placement_id):
        neighboors_anchor_face = self.sampler.object.stable_placement_dict['neighboors'][face_anchor_id]
        face_rotate_id = self.find_opposite_face(face_anchor_id, neighboors_anchor_face)
        normal_list_placement = self.sampler.object.stable_placement_dict['normal_stable_config'][
            placement_id]
        normal_rotate_sampling_base = normal_list_placement[face_rotate_id]
        normal_anchor_sampling_base = normal_list_placement[face_anchor_id]
        return normal_rotate_sampling_base, normal_anchor_sampling_base

    def get_highest_points(self, face_rotate_sampling_base, face_anchor_sampling_base):
        points_rotate_sampling_base = util.find_highest_points(face_rotate_sampling_base)
        points_anchor_sampling_base = util.find_highest_points(face_anchor_sampling_base)
        return points_rotate_sampling_base, points_anchor_sampling_base

    def compute_lever_z_axes(self, face_anchor_id, placement_id):
        face_rotate_sampling_base, face_anchor_sampling_base, face_rotate_id = self.get_points_face(face_anchor_id, placement_id)
        points_rotate_sampling_base, points_anchor_sampling_base = self.get_highest_points(face_rotate_sampling_base, face_anchor_sampling_base)
        z_axis_rotate_sampling_base = helper.axis_from_points(points_rotate_sampling_base)
        z_axis_anchor_sampling_base = helper.axis_from_points(points_anchor_sampling_base, vec_guide=z_axis_rotate_sampling_base)
        return z_axis_rotate_sampling_base, z_axis_anchor_sampling_base

    def compute_lever_y_axes(self, face_anchor_id, placement_id):
        face_rotate_sampling_base, face_anchor_sampling_base, face_rotate_id = self.get_points_face(face_anchor_id, placement_id)
        y_axis_rotate_sampling_base, y_axis_anchor_sampling_base = self.get_face_normals(face_anchor_id, placement_id)
        return y_axis_rotate_sampling_base, y_axis_anchor_sampling_base

    def compute_lever_x_axes(self, y_axis_rotate_sampling_base, y_axis_anchor_sampling_base, z_axis_rotate_sampling_base, z_axis_anchor_sampling_base):
        x_axis_rotate_sampling_base = np.cross(y_axis_rotate_sampling_base, z_axis_rotate_sampling_base)
        x_axis_anchor_sampling_base = np.cross(y_axis_anchor_sampling_base, z_axis_anchor_sampling_base)
        return x_axis_rotate_sampling_base, x_axis_anchor_sampling_base

    def find_lever_points(self, face_anchor_id, placement_id):
        face_rotate_sampling_base, face_anchor_sampling_base, face_rotate_id = self.get_points_face(face_anchor_id, placement_id)
        points_rotate_sampling_base, points_anchor_sampling_base = self.get_highest_points(face_rotate_sampling_base, face_anchor_sampling_base)
        lever_point_rotate_sampling_base = util.find_mid_point(points_rotate_sampling_base)
        lever_point_anchor_sampling_base = util.find_mid_point(points_anchor_sampling_base)
        return lever_point_rotate_sampling_base, lever_point_anchor_sampling_base, face_rotate_id

    def compute_nominal_gripper_poses(self,face_anchor_id, placement_id, trans_anchor, trans_rotate):
        z_axis_rotate_sampling_base, z_axis_anchor_sampling_base = self.compute_lever_z_axes(face_anchor_id,
                                                                                             placement_id)
        y_axis_rotate_sampling_base, y_axis_anchor_sampling_base = self.compute_lever_y_axes(face_anchor_id,
                                                                                             placement_id)
        x_axis_rotate_sampling_base, x_axis_anchor_sampling_base = self.compute_lever_x_axes(y_axis_rotate_sampling_base,
                                                                                             y_axis_anchor_sampling_base,
                                                                                             z_axis_rotate_sampling_base,
                                                                                             z_axis_anchor_sampling_base)
        gripper_pose_rotate = roshelper.pose_from_vectors(x_axis_rotate_sampling_base,
                                                          y_axis_rotate_sampling_base,
                                                          z_axis_rotate_sampling_base,
                                                          trans_rotate,
                                                          frame_id="proposals_base")
        gripper_pose_anchor = roshelper.pose_from_vectors(x_axis_anchor_sampling_base,
                                                          y_axis_anchor_sampling_base,
                                                          z_axis_anchor_sampling_base,
                                                          trans_anchor,
                                                          frame_id="proposals_base")
        return gripper_pose_rotate, gripper_pose_anchor

    def find_lever_angle_sign(self, gripper_pose_proposals_base):
        #1. extract vectors
        T = roshelper.matrix_from_pose(gripper_pose_proposals_base)
        x_vec, y_vec, z_vec = helper.matrix2vec(T)
        #2. return sign based on condition
        if x_vec[2]>0:
            return -1
        else:
            return 1

    def tilt_gripper_poses(self, gripper_rotate, gripper_anchor, rotate_angle=0*np.pi/180, anchor_angle=0*np.pi/180):
        #1. convert to gripper frame
        gripper_rotate_gripper_frame = roshelper.convert_reference_frame(gripper_rotate,
                                                                         gripper_rotate,
                                                                         roshelper.unit_pose(),
                                                                         frame_id = "gripper_rotate")
        gripper_anchor_gripper_frame = roshelper.convert_reference_frame(gripper_anchor,
                                                                         gripper_anchor,
                                                                         roshelper.unit_pose(),
                                                                         frame_id = "gripper_anchor")
        #2. rotate
        rotate_angle = self.find_lever_angle_sign(gripper_rotate) * rotate_angle
        anchor_angle = self.find_lever_angle_sign(gripper_anchor) * anchor_angle
        pose_transform_rotate = roshelper.pose_from_matrix(tfm.euler_matrix(0, 0, rotate_angle,'sxyz'),
                                                           frame_id="proposals_base")
        pose_transform_rotation_anchor = roshelper.pose_from_matrix(tfm.euler_matrix(0, 0, anchor_angle,'sxyz'),
                                                            frame_id="proposals_base")
        gripper_rotate_tilded_gripper_frame = roshelper.transform_pose(gripper_rotate_gripper_frame, pose_transform_rotate)
        gripper_anchor_tilded_gripper_frame = roshelper.transform_pose(gripper_anchor_gripper_frame, pose_transform_rotation_anchor)
        #3. convert back to proposals base frame
        gripper_rotate_tilded_proposals_base = roshelper.convert_reference_frame(gripper_rotate_tilded_gripper_frame,
                                                                         roshelper.unit_pose(),
                                                                         gripper_rotate,
                                                                         frame_id = "proposals_base")
        gripper_anchor_tilded_proposals_base = roshelper.convert_reference_frame(gripper_anchor_tilded_gripper_frame,
                                                                         roshelper.unit_pose(),
                                                                         gripper_anchor,
                                                                         frame_id = "proposals_base")
        return gripper_rotate_tilded_proposals_base, gripper_anchor_tilded_proposals_base

    def generate_pose_samples(self, gripper_pose_rotate, gripper_pose_anchor):
        gripper_poses_list = []
        gripper_nominal_list = [gripper_pose_anchor, gripper_pose_rotate]
        # gripper_index_list = [[0,1],[0,1]]
        # flip_index_list = [[None, None],["x","y"]]
        # for i in range(len(flip_index_list)):
        #     gripper_index = gripper_index_list[i]
        #     flip_index = flip_index_list[i]
        #     gripper_left = roshelper.flip_orientation(gripper_nominal_list[gripper_index[0]], flip_axis=flip_index[0], constant_axis=flip_index[1])
        #     gripper_right = roshelper.flip_orientation(gripper_nominal_list[gripper_index[1]], flip_axis=flip_index[0], constant_axis=flip_index[1])
        #     gripper_poses = [gripper_left, gripper_right]
        #     gripper_poses_list.append(gripper_poses)

        return gripper_nominal_list

    def generate_lever_samples(self):
        #1. loop through stable placements
        lever_id = 0
        for placement_id, face in enumerate(self.sampler.object.stable_placement_dict['convex_face_3d']):
            #2. rotate all grasp points, normals, grasp_poses
            lever_id_list_new = []
            lever_rotation_points_points_list_new = []
            lever_face_list_new = []
            lever_placement_start_list_new = []
            lever_placement_end_list_new = []
            lever_gripper_pose_list_new = []
            lever_face_normals_list_new = []
            #2. determine neighboor faces (all neighboors have associated potential lever action)
            neighboors = self.sampler.object.stable_placement_dict['neighboors'][placement_id]
            for face_anchor_id in neighboors:
                index = self.sampler.object.stable_placement_dict['neighboors'][placement_id].index(face_anchor_id)
                rotation_points = self.sampler.object.stable_placement_dict['common_points'][placement_id][index]
                #3. identify lever points
                lever_point_rotate_sampling_base, lever_point_anchor_sampling_base, face_rotate_id = \
                    self.find_lever_points(face_anchor_id, placement_id)
                #4. determine gripper pose from lever points
                gripper_pose_rotate, gripper_pose_anchor = self.compute_nominal_gripper_poses(face_anchor_id,
                                                                                                placement_id,
                                                                                                lever_point_anchor_sampling_base,
                                                                                                lever_point_rotate_sampling_base)
                gripper_pose_rotate_tilde, gripper_pose_anchor_tilde = self.tilt_gripper_poses(gripper_pose_rotate, gripper_pose_anchor)
                gripper_poses_list = self.generate_pose_samples(gripper_pose_rotate_tilde,
                                                                gripper_pose_anchor_tilde)
                face_normals = [self.sampler.object.stable_placement_dict['normal_stable_config'][placement_id][face_anchor_id],
                                self.sampler.object.stable_placement_dict['normal_stable_config'][placement_id][face_rotate_id]]
                # 3. check collisions between grippers and table
                lever_id_list_new.append(lever_id)
                lever_face_list_new.append(lever_id)
                lever_rotation_points_points_list_new.append(rotation_points)
                lever_placement_start_list_new.append(placement_id)
                lever_placement_end_list_new.append(face_anchor_id)
                lever_gripper_pose_list_new.append(gripper_poses_list)
                lever_face_normals_list_new.append(face_normals)
                lever_id += 1
            self.samples_dict['sample_ids'].append(lever_id_list_new)
            self.samples_dict['rotation_points'].append(lever_rotation_points_points_list_new)
            self.samples_dict['face_ids'].append([face_anchor_id, face_rotate_id])
            self.samples_dict['placement_id'].append(lever_placement_start_list_new)
            self.samples_dict['placement_end_id'].append(lever_placement_end_list_new)
            self.samples_dict['gripper_poses'].append(lever_gripper_pose_list_new)
            self.samples_dict['face_normals'].append(lever_face_normals_list_new)

def identify_placement_ids(stable_placement_dict, placement_list):
    index_list = []
    sample_id_list = []
    placement_end_list = stable_placement_dict['placement_end_id'][placement_list[0]]
    sample_ids_list = stable_placement_dict['sample_ids'][placement_list[0]]
    counter = 0
    for placement_end, sample_id in zip(placement_end_list, sample_ids_list):
        if placement_end==placement_list[-1]:
            index_list.append(counter)
            sample_id_list.append(sample_id)
        counter += 1
    return index_list, sample_id_list

def get_gripper_poses_from_samples(stable_placement_dict, placement_list):
    index_list, sample_id_list = identify_placement_ids(stable_placement_dict, placement_list)
    #get all samples that can perform a given placement
    gripper_pose_element_list = []
    for index in index_list:
        gripper_pose_element_list.append(stable_placement_dict['gripper_poses'][placement_list[0]][index])

    return gripper_pose_element_list[0], index_list[0]
