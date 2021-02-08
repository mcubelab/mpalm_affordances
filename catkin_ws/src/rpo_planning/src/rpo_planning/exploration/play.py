import os
import os.path as osp
import sys
import time
import argparse
import numpy as np
import rospy
import rospkg 
import signal
import threading
import pickle
import open3d
import copy

from airobot import Robot
from airobot.utils import common
import pybullet as p


from rpo_planning.utils import common as util
from rpo_planning.execution.motion_playback import OpenLoopMacroActions
from rpo_planning.config.base_skill_cfg import get_skill_cfg_defaults
from rpo_planning.config.explore_task_cfg import get_task_cfg_defaults
from rpo_planning.config.multistep_eval_cfg import get_multistep_cfg_defaults
from rpo_planning.robot.multicam_env import YumiMulticamPybullet 
from rpo_planning.utils.object import CuboidSampler
from rpo_planning.utils.pb_visualize import GoalVisual
from rpo_planning.utils.data import MultiBlockManager
from rpo_planning.utils.motion import guard
from rpo_planning.utils.planning.pointcloud_plan import PointCloudNode
from rpo_planning.utils.visualize import PCDVis, PalmVis

from rpo_planning.skills.samplers.pull import PullSamplerBasic, PullSamplerVAE
from rpo_planning.skills.samplers.push import PushSamplerBasic, PushSamplerVAE
from rpo_planning.skills.samplers.grasp import GraspSamplerBasic, GraspSamplerVAE
from rpo_planning.skills.primitive_skills import (
    GraspSkill, PullRightSkill, PullLeftSkill, PushRightSkill, PushLeftSkill
)
from rpo_planning.motion_planning.primitive_planners import (
    grasp_planning_wf, pulling_planning_wf, pushing_planning_wf
)


class PlayObjects(object):
    def __init__(self):
        self.name = None
        self.pose = None
        self.mesh = None
        self.obj_id = None


class PlayEnvironment(object):
    def __init__(self, pb_client, robot, cuboid_manager, cuboid_sampler, target_surface_ids, goal_visualization=False):
        self.pb_client = pb_client
        self.robot = robot
        self.cams = self.robot.cams
        self.robot_id = robot.yumi_ar.arm.robot_id
        self.cuboid_manager = cuboid_manager
        self.cuboid_sampler = cuboid_sampler
        self.goal_visualization = goal_visualization

        self.camera_inds = [0, 1, 2, 3]
        self.target_surface_ids = target_surface_ids

        # assume first target surface is the default
        self.default_surface_id = self.target_surface_ids[0]

        # table boundaries are [[x_min, x_max], [y_min, y_max]]
        self.table_boundaries = {}
        self.table_boundaries['x'] = np.array([0.1, 0.5])
        self.table_boundaries['y'] = np.array([-0.4, 0.4])

        self.clear_current_objects()

    @staticmethod
    def object_fly_away(obj_id):
        obj_pos = p.getBasePositionAndOrientation(obj_id)[0]
        return obj_pos[2] < -0.1

    @staticmethod
    def object_table_contact(robot_id, obj_id, table_id, pb_cl):
        table_contacts = p.getContactPoints(
            robot_id,
            obj_id,
            table_id,
            -1,
            pb_cl)

        n_list = []
        for pt in table_contacts:
            n_list.append(pt[-5])
        return len(table_contacts) > 0, n_list   

    @staticmethod
    def object_in_boundaries(obj_id, x_boundaries, y_boundaries, z_boundaries=None):
        obj_pos = p.getBasePositionAndOrientation(obj_id)[0]
        in_x = obj_pos[0] >= min(x_boundaries) and obj_pos[0] <= max(x_boundaries)
        in_y = obj_pos[1] >= min(y_boundaries) and obj_pos[1] <= max(y_boundaries)
        in_z = True if z_boundaries is None else obj_pos[2] >= min(z_boundaries) and obj_pos[2] <= max(z_boundaries)
        return in_x and in_y and in_z

    def _random_table_xy(self):
        x = np.random.random() * (max(self.table_boundaries['x']) - min(self.table_boundaries['x'])) + min(self.table_boundaries['x'])
        y = np.random.random() * (max(self.table_boundaries['y']) - min(self.table_boundaries['y'])) + min(self.table_boundaries['y'])
        return x, y

    def get_random_pose_mesh(self, tmesh):
        """Sample a random pose in the table top environment with a particular mesh object.
        This method computes a stable pose of the mesh using the internal trimesh function,
        then samples a position that is in the valid object region.

        Args:
            tmesh (Trimesh.T): [description]

        Returns:
            list: Random pose [x, y, z, qx, qy, qz, qw] in the tabletop environment
        """
        # TODO: implement functionality to be able to directly sample initial states which are not at z=0
        stable_poses = tmesh.compute_stable_poses()[0]
        pose = np.random.choice(stable_poses, 1)
        x, y = self._random_table_xy()
        pose[0] = x
        pose[1] = y
        return util.pose_stamped2list(util.pose_from_matrix(pose))

    def clear_current_objects(self):
        self._current_obj_list = []

    def get_current_obj_info(self):
        return copy.deepcopy(self._current_obj_list)

    def _sample_cuboid(self, obj_fname=None):
        if obj_fname is None:
            obj_fname = self.cuboid_manager.get_cuboid_fname()
        obj_id, _, mesh, goal_obj_id = self.cuboid_sampler.sample_cuboid_pybullet(
            obj_fname, goal=self.goal_visualization)
        return mesh, obj_fname, obj_id, goal_obj_id

    def sample_objects(self, n=1):
        self.clear_current_objects()
        for _ in range(n):
            mesh, cuboid_fname, obj_id, goal_obj_id = self._sample_cuboid()
            obj_dict = {}
            obj_dict['mesh'] = mesh
            obj_dict['fname'] = cuboid_fname
            obj_dict['obj_id'] = obj_id
            obj_dict['goal_obj_id'] = goal_obj_id
            self._current_obj_list.append(obj_dict)

    def initialize_object_states(self):
        # check if current set of objects is empty
        if not len(self._current_obj_list):
            raise ValueError('Must sample objects in the environment before '
                             'initializing states')
        for i, obj_dict in enumerate(self._current_obj_list):
            # get a random start pose and set to that pose
            start_pose = self.get_random_pose_mesh(obj_dict['mesh'])
            obj_id = obj_dict['obj_id']
            self.pb_client.reset_body(obj_id, start_pose[:3], start_pose[3:])

    def segment_object(self, obj_body_id, pts, seg):
        obj_inds = np.where(seg == obj_body_id)
        obj_pts = pts[obj_inds[0], :]
        return obj_pts

    def segment_surface(self, surface_link_id, pts, seg):
        # use PyBullet's weird way of converting the segmentation labels based on body/link id
        surface_val = self.robot_id + (surface_link_id + 1) << 24
        surface_inds = np.where(seg == surface_val)

        # filter
        surface_pts = pts[surface_inds[0], :]
        return surface_pts

    def get_observation(self):
        # TODO handle if we just want to get a point cloud of the table

        # check if current set of objects is empty
        if not len(self._current_obj_list):
            raise ValueError('Must sample objects in the environment before '
                             'initializing states')

        depths = []
        segs = []
        raw_pcd_pts = []
        # get the full, unsegmented point cloud from each camera
        for i, cam in enumerate(self.cams):
            rgb, depth, seg = cam.get_images(
                get_rgb=True,
                get_depth=True,
                get_seg=True
            )

            pts_raw, colors_raw = cam.get_pcd(
                in_world=True,
                filter_depth=False,
                depth_max=1.0
            )

            raw_pcd_pts.append(pts_raw)
            segs.append(seg.flatten())

        # combine all point clouds from each viewpoint together
        raw_pcd_pts = np.concatenate(np.asarray(raw_pcd_pts, dtype=np.float32), axis=0)
        segs = np.concatenate(np.asarray(segs, dtype=np.float32), axis=0)

        # go through random objects, and get the individual segmented object point clouds
        obj_pcd_pts = []

        for i, obj_dict in enumerate(self._current_obj_list):
            obj_id = obj_dict['obj_id']
            pcd_pts = self.segment_object(obj_id, raw_pcd_pts, segs)
            obj_pcd_pts.append(pcd_pts)

        # TODO: incorporate sampling a target surface
        # # and then go through static target surfaces, as if they were objects
        target_surface_pts = []
        for i, obj_id in enumerate(self.target_surface_ids):
            pcd_pts = self.segment_surface(obj_id, raw_pcd_pts, segs)
            target_surface_pts.append(pcd_pts)
        obs = {}
        obs['object_pcds'] = obj_pcd_pts
        obs['surface_pcds'] = target_surface_pts
        return obs

    def _valid_object_status(self, object_table_contact, object_fly_away, object_in_boundaries, *args):
        """Function to check if object state is valid, depending on a number of symbolic states we obtain
        from the simulator. All inputs are meant to be boolean values computed using separate methods

        Args:
            object_table_contact (bool): True if contact was detected between the object and the table
            object_fly_away (bool): True if object position was detected as below the nominal table surface position
            object_in_boundaries (bool): True if object is inside designated table boundaries

        Returns:
            bool: True if object state is valid
        """
        return object_table_contact and not object_fly_away and object_in_boundaries

    def check_environment_status(self):
        valid = True
        for i, obj_dict in enumerate(self._current_obj_list):
            # check if object is contacting table
            obj_table_contacting = self.object_table_contact(
                robot_id=self.robot.yumi_ar.arm.robot_id, 
                obj_id=obj_dict['obj_id'], 
                table_id=self.default_surface_id, 
                pb_cl=self.robot.yumi_ar.pb_client.get_client_id())

            # check if object position is below the table
            obj_below_table = self.object_fly_away(obj_dict['obj_id'])

            # check if object is within valid table boundaries (not stuck behind the robot)
            obj_in_boundaries = self.object_in_boundaries(
                obj_dict['obj_id'], self.table_boundaries['x'], self.table_boundaries['y'])
            
            valid = valid and self._valid_object_status(
                obj_table_contacting, 
                obj_below_table,
                obj_in_boundaries)
        return valid


class SkillExplorer(object):
    def __init__(self, skills):
        self.skills = skills

    def sample_skill(self, strategy=None):
        return np.random.choice(self.skills.keys(), 1)


def main(args):
    # example_config_path = osp.join(os.environ['CODE_BASE'], args.example_config_path)
    rospack = rospkg.RosPack()
    skill_config_path = osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/config/skill_cfgs')
    pull_cfg_file = osp.join(skill_config_path, 'pull') + ".yaml"
    pull_cfg = get_multistep_cfg_defaults()
    pull_cfg.merge_from_file(pull_cfg_file)
    pull_cfg.freeze()

    grasp_cfg_file = osp.join(skill_config_path, 'grasp') + ".yaml"
    grasp_cfg = get_multistep_cfg_defaults()
    grasp_cfg.merge_from_file(grasp_cfg_file)
    grasp_cfg.freeze()

    push_cfg_file = osp.join(skill_config_path, 'push') + ".yaml"
    push_cfg = get_multistep_cfg_defaults()
    push_cfg.merge_from_file(push_cfg_file)
    push_cfg.freeze()

    cfg = pull_cfg

    signal.signal(signal.SIGINT, util.signal_handler)
    rospy.init_node('PlayExplore')

    np.random.seed(args.np_seed)
    # initialize airobot and modify dynamics
    yumi_ar = Robot('yumi_palms',
                    pb=True,
                    pb_cfg={'gui': args.visualize,
                            'opengl_render': False},
                    arm_cfg={'self_collision': False,
                             'seed': args.np_seed})

    yumi_ar.arm.go_home(ignore_physics=True)

    yumi_gs = YumiMulticamPybullet(
        yumi_ar,
        cfg,
        exec_thread=False,
    )

    if args.sim:
        r_gel_id = cfg.RIGHT_GEL_ID
        l_gel_id = cfg.LEFT_GEL_ID
        table_id = cfg.TABLE_ID

        alpha = cfg.ALPHA
        K = cfg.GEL_CONTACT_STIFFNESS
        restitution = cfg.GEL_RESTITUION

        p.changeDynamics(
            yumi_ar.arm.robot_id,
            r_gel_id,
            lateralFriction=0.5,
            restitution=restitution,
            contactStiffness=K,
            contactDamping=alpha*K,
            rollingFriction=args.rolling
        )

        p.changeDynamics(
            yumi_ar.arm.robot_id,
            l_gel_id,
            lateralFriction=0.5,
            restitution=restitution,
            contactStiffness=K,
            contactDamping=alpha*K,
            rollingFriction=args.rolling
        )

        p.changeDynamics(
            yumi_ar.arm.robot_id,
            table_id,
            lateralFriction=0.1
        )

    if args.baseline:
        print('LOADING BASELINE SAMPLERS')
        pull_sampler = PullSamplerBasic()
        grasp_sampler = GraspSamplerBasic(None)
        push_sampler = PushSamplerVAE()
    else:
        print('LOADING LEARNED SAMPLERS')
        pull_sampler = PullSamplerVAE()
        push_sampler = PushSamplerVAE()
        grasp_sampler = GraspSamplerVAE(default_target=None)

    pull_right_skill = PullRightSkill(
        pull_sampler,
        yumi_gs,
        pulling_planning_wf,
        ignore_mp=False,
        avoid_collisions=True
    )

    pull_left_skill = PullLeftSkill(
        pull_sampler,
        yumi_gs,
        pulling_planning_wf,
        ignore_mp=False,
        avoid_collisions=True
    )

    push_right_skill = PushRightSkill(
        push_sampler,
        yumi_gs,
        pushing_planning_wf,
        ignore_mp=False,
        avoid_collisions=True
    )

    push_left_skill = PushLeftSkill(
        push_sampler,
        yumi_gs,
        pushing_planning_wf,
        ignore_mp=False,
        avoid_collisions=True
    )

    grasp_skill = GraspSkill(grasp_sampler, yumi_gs, grasp_planning_wf)
    grasp_pp_skill = GraspSkill(grasp_sampler, yumi_gs, grasp_planning_wf, pp=True)

    skills = {}
    skills['pull_right'] = pull_right_skill
    skills['pull_left'] = pull_left_skill
    # skills['grasp'] = grasp_skill
    # skills['grasp_pp'] = grasp_pp_skill
    # skills['push_right'] = push_right_skill
    # skills['push_left'] = push_left_skill

    # create exploring agent
    agent = SkillExplorer(skills)

    pb_info = None
    if args.sim:
        pb_info = {}
        pb_info['object_id'] = None
        pb_info['object_mesh_file'] = None
        pb_info['pb_client'] = yumi_ar.pb_client.get_client_id()

    action_planner = OpenLoopMacroActions(
        cfg,
        yumi_gs,
        pb=args.sim,
        pb_info=pb_info
    )

    cuboid_sampler = CuboidSampler(
        osp.join(
            os.environ['CODE_BASE'],
            'catkin_ws/src/config/descriptions/meshes/objects/cuboids/nominal_cuboid.stl'),
        pb_client=yumi_ar.pb_client)
    cuboid_fname_template = osp.join(
        os.environ['CODE_BASE'],
        'catkin_ws/src/config/descriptions/meshes/objects/cuboids/')
    cuboid_manager = MultiBlockManager(
        cuboid_fname_template,
        cuboid_sampler,
        robot_id=yumi_ar.arm.robot_id,
        table_id=cfg.TABLE_ID,
        r_gel_id=cfg.RIGHT_GEL_ID,
        l_gel_id=cfg.LEFT_GEL_ID)

    # visualization stuff
    palm_mesh_file = osp.join(os.environ['CODE_BASE'], cfg.PALM_MESH_FILE)
    table_mesh_file = osp.join(os.environ['CODE_BASE'], cfg.TABLE_MESH_FILE)
    viz_palms = PalmVis(palm_mesh_file, table_mesh_file, cfg)

    # create interface to make guarded movements
    guarder = guard.GuardedMover(robot=yumi_gs, pb_client=yumi_ar.pb_client.get_client_id(), cfg=cfg)

    # create manager
    env = PlayEnvironment(pb_client=yumi_ar.pb_client, robot=yumi_gs, cuboid_manager=cuboid_manager, cuboid_sampler=cuboid_sampler, target_surface_ids=[cfg.TABLE_ID])

    # sample some objects
    env.sample_objects(n=1)

    # instantiate objects in the world at specific poses
    env.initialize_object_states()

    # setup save stuff
    trasition_save_dir = osp.join(
        os.environ['CODE_BASE'], 
        'catkin_ws/src/primitives', 
        args.data_dir, args.save_data_dir, args.exp)
    if not osp.exists(trasition_save_dir):
        os.makedirs(trasition_save_dir)
    total_transitions = 0

    time.sleep(1.0)
    while True:
        yumi_ar.arm.go_home(ignore_physics=True)

        if not env.check_environment_status():
            env.initialize_object_states()
        # should we wait until everything in the environment is stable?
        # TODO: check if objects are stable
        # in the meantime, just use a sleep to wait long enough

        feasible = False
        current_obs_trial = 0
        # sample a skill type
        skill_type = agent.sample_skill()
        while not feasible:
            if current_obs_trial > 5:
                break
            current_obs_trial += 1
            # get an observation from the environment
            obs = env.get_observation()
            obj_info = env.get_current_obj_info()[0]
            if args.sim:
                action_planner.update_object(obj_id=obj_info['obj_id'], mesh_file=obj_info['fname'])
                guarder.set_object_id(obj_id=obj_info['obj_id'])
                p.changeDynamics(
                    obj_info['obj_id'],
                    -1,
                    lateralFriction=1.0
                )

            # sample an object from list of object point clouds
            ind = np.random.randint(len(obs['object_pcds']))
            obj_pcd = obs['object_pcds'][ind]

            # sample a target surface
            target_ind = np.random.randint(len(obs['surface_pcds']))
            target_surface = obs['surface_pcds'][target_ind]

            pointcloud_pts = obj_pcd[::int(obj_pcd.shape[0]/100.0)][:100]
            pointcloud_pts_full = obj_pcd

            # scale down point cloud to ensure some contact is made
            centroid = np.mean(pointcloud_pts_full, axis=0)
            centered_pts = pointcloud_pts - centroid
            centered_pts_full = pointcloud_pts_full - centroid
            centered_pts *= args.pcd_scalar 
            centered_pts_full *= args.pcd_scalar
            pointcloud_pts = centered_pts + centroid
            pointcloud_pts_full = centered_pts_full + centroid

            # construct input to agent
            start_sample = PointCloudNode()
            start_sample.set_pointcloud(
                pcd=pointcloud_pts,
                pcd_full=pointcloud_pts_full
            )

            # planes = pcd_segmenter.get_pointcloud_planes(pointcloud_pts_full)
            # start_sample.set_planes(planes)

            # sample an action
            new_state = agent.skills[skill_type].sample(
                        start_sample,
                        target_surface=target_surface,
                        final_trans=False
                    )

            primitive_name = skill_type
            trans_execute = util.pose_from_matrix(new_state.transformation)

            if args.trimesh_viz:
                pcd_data = {}
                pcd_data['start'] = start_sample.pointcloud_full
                pcd_data['object_pointcloud'] = start_sample.pointcloud_full
                pcd_data['transformation'] = np.asarray(util.pose_stamped2list(util.pose_from_matrix(new_state.transformation)))
                pcd_data['contact_world_frame_right'] = np.asarray(new_state.palms[:7])
                if 'pull' in skill_type or 'push' in skill_type:
                    pcd_data['contact_world_frame_left'] = np.asarray(new_state.palms[:7])
                else:
                    pcd_data['contact_world_frame_left'] = np.asarray(new_state.palms[7:])
                scene = viz_palms.vis_palms_pcd(pcd_data, world=True, centered=False, corr=False)
                scene.show()

            # try to execute the action
            if 'grasp' in skill_type:
                local_plan = grasp_planning_wf(
                    util.list2pose_stamped(new_state.palms[7:]),
                    util.list2pose_stamped(new_state.palms[:7]),
                    trans_execute
                )
            elif 'pull' in skill_type:
                local_plan = pulling_planning_wf(
                    util.list2pose_stamped(new_state.palms[:7]),
                    util.list2pose_stamped(new_state.palms[:7]),
                    trans_execute
                )
            elif 'push' in skill_type:
                local_plan = pushing_planning_wf(
                    util.list2pose_stamped(new_state.palms[:7]),
                    util.list2pose_stamped(new_state.palms[:7]),
                    trans_execute,
                    arm='r',
                    G_xy=(new_state.palms[:2] - np.mean(start_sample.pointcloud_full, axis=0)[:-1])
                )

            # get any objects that are left behind in the planning scene out of there
            action_planner.clear_planning_scene()

            # full motion planning check to see if the plan is feasible
            feasible = agent.skills[skill_type].feasible_motion(
                state=new_state,
                start_joints=None,
                nominal_plan=local_plan)

        if not feasible:
            continue
        active_arm = 'right' if 'right' in skill_type else 'left'
        action_planner.active_arm = active_arm

        try:
            if args.sim:
                if 'push' in skill_type:
                    p.changeDynamics(
                        yumi_ar.arm.robot_id,
                        r_gel_id,
                        rollingFriction=1.0
                    )

                    p.changeDynamics(
                        yumi_ar.arm.robot_id,
                        l_gel_id,
                        rollingFriction=1.0
                    )
                else:
                    p.changeDynamics(
                        yumi_ar.arm.robot_id,
                        r_gel_id,
                        rollingFriction=1e-4
                    )

                    p.changeDynamics(
                        yumi_ar.arm.robot_id,
                        l_gel_id,
                        rollingFriction=1e-4
                    )

            if 'grasp' in skill_type:
                # collision free motion to good start configuration
                if args.sim:
                    action_planner.add_remove_scene_object(action='add')
                    time.sleep(0.5)
                    _, _ = yumi_gs.move_to_joint_target_mp(grasp_cfg.RIGHT_INIT, grasp_cfg.LEFT_INIT, execute=True)
                    action_planner.add_remove_scene_object(action='remove')
                    time.sleep(0.5)

                # begin guarded move
                _, _ = action_planner.dual_arm_setup(local_plan[0], 0, pre=True)
                start_playback_time = time.time()
                if not guarder.still_grasping():
                    jj = 0
                    while True:
                        if guarder.still_grasping() or time.time() - start_playback_time > 20.0:
                            jj += 1
                        if jj > 2:
                            break
                        # TODO: deal with how this IK approach can break things
                        action_planner.dual_arm_approach()
                        time.sleep(0.075)
                        new_plan = grasp_planning_wf(
                            yumi_gs.get_current_tip_poses()['left'],
                            yumi_gs.get_current_tip_poses()['right'],
                            trans_execute
                        )
                    playback_plan = new_plan
                else:
                    playback_plan = local_plan
                for k, subplan in enumerate(playback_plan):
                    action_planner.playback_dual_arm('grasp', subplan, k, pre=False)
                    time.sleep(1.0)
            elif 'pull' in skill_type or 'push' in skill_type:
                skill_cfg = pull_cfg if 'pull' in skill_type else push_cfg
                # start at a good configuration
                if args.sim:
                    action_planner.add_remove_scene_object(action='add')
                    time.sleep(0.5)
                    _, _ = yumi_gs.move_to_joint_target_mp(skill_cfg.RIGHT_INIT, skill_cfg.LEFT_INIT, execute=True)
                    action_planner.add_remove_scene_object(action='remove')
                    time.sleep(0.5)

                
                # perform guarded move to make contact
                _, _ = action_planner.single_arm_setup(local_plan[0], pre=True)
                start_playback_time = time.time()
                n = True if 'pull' in skill_type else False
                if not guarder.still_pulling(arm=active_arm, n=n):
                    while True:
                        if guarder.still_pulling(arm=active_arm, n=n) or time.time() - start_playback_time > 20.0:
                            break
                        action_planner.single_arm_approach(arm=active_arm)
                        time.sleep(0.075)
                        replan_args = [
                            yumi_gs.get_current_tip_poses()['left'],
                            yumi_gs.get_current_tip_poses()['right'],
                            trans_execute
                        ]
                        if 'pull' in skill_type:
                            new_plan = pulling_planning_wf(*replan_args)
                        else:
                            new_plan = pushing_planning_wf(*replan_args)

                    playback_plan = new_plan
                else:
                    playback_plan = local_plan

                # execute open loop motion
                action_planner.playback_single_arm(skill_type, playback_plan[0], pre=False)
                action_planner.single_arm_retract(active_arm)
        except ValueError as e:
            print(e)
            continue

        time.sleep(1.0)
        yumi_ar.arm.go_home(ignore_physics=True)
        time.sleep(1.0)

        # check if environment state is still valid
        if env.check_environment_status():
            # get new obs
            obs_new = env.get_observation()

            o = obj_pcd
            To = np.asarray(util.pose_stamped2list(util.pose_from_matrix(new_state.transformation)))
            Tpc = np.asarray(new_state.palms)
            o_next = obs_new['object_pcds'][ind]

            # store the transition
            total_transitions += 1
            transition_fname = osp.join(trasition_save_dir, '%d.npz' % total_transitions)
            info_string = 'Saving transition number %d using action %s to fname: %s' % (total_transitions, skill_type, transition_fname)
            print(info_string)
            np.savez(transition_fname,
                observation = o,
                action_type = skill_type,
                contact = Tpc,
                subgoal = To,
                next_observation = o_next
            )
            # transition = (o, skill_type, To, Tpc, o_next)

        else:
            # reset if we're not valid
            env.initialize_object_states()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--save_data_dir', type=str, default='play_transitions')
    parser.add_argument('--exp', type=str, default='debug')
    parser.add_argument('--config_package_path', type=str, default='catkin_ws/src/config/')
    parser.add_argument('--example_config_path', type=str, default='catkin_ws/src/primitives/config')
    parser.add_argument('-v', '--visualize', action='store_true')
    parser.add_argument('--np_seed', type=int, default=0)
    parser.add_argument('--multi', action='store_true')
    parser.add_argument('--trimesh_viz', action='store_true')
    parser.add_argument('--ignore_physics', action='store_true')
    parser.add_argument('-r', '--rolling', type=float, default=0.0, help='rolling friction value for pybullet sim')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--camera_inds', nargs='+', type=int)
    parser.add_argument('--pcd_noise', action='store_true')
    parser.add_argument('--pcd_noise_std', type=float, default=0.0025)
    parser.add_argument('--pcd_noise_rate', type=float, default=0.00025)
    parser.add_argument('--pcd_scalar', type=float, default=0.9)

    args = parser.parse_args()
    main(args)
