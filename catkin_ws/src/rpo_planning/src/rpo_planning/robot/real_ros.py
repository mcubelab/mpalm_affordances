from rpo_planning.robot.yumi_ar_ros import YumiAIRobotROS


class YumiReal(YumiAIRobotROS):
    """
    Class for interfacing with Yumi in PyBullet
    with external motion planning, inverse kinematics,
    and forward kinematics, along with other helpers
    """
    def __init__(self, yumi_ar, cfg):
        """
        Class constructor. Sets up internal motion planning interface
        for each arm, forward and inverse kinematics solvers, and background
        threads for updating the position of the robot.

        Args:
            yumi_ar (airobot Robot): Instance of PyBullet simulated robot, from
                airobot library
            cfg (YACS CfgNode): Configuration parameters
            exec_thread (bool, optional): Whether or not to start the
                background joint position control thread. Defaults to True.
            sim_step_repeat (int, optional): Number of simulation steps
                to take each time the desired joint position value is
                updated. Defaults to 10
        """
        super(YumiReal, self).__init__(yumi_ar, cfg)

    def update_joints(self, pos, arm=None, egm=True):
        if arm is None:
            both_pos = pos
        elif arm == 'right':
            both_pos = list(pos) + self.yumi_ar.arm.arms['left'].get_jpos()
        elif arm == 'left':
            both_pos = self.yumi_ar.arm.arms['right'].get_jpos() + list(pos)
        else:
            raise ValueError('Arm not recognized')
        # if egm:
        #     self.yumi_ar.arm.set_jpos(both_pos, wait=True)
        # else:
        #     both_pos = [both_pos]
        #     self.yumi_ar.arm.set_jpos_buffer(both_pos, sync=True, wait=False)
        self.yumi_ar.arm.set_jpos(both_pos, wait=False)


def main(args):
    # get cfgs for each primitive
    from rpo_planning.config.base_skills_cfg import get_skill_cfg_defaults
    from airobot import Robot
    import os.path as osp
    import rospack

    skill_config_path = osp.join(rospack.get_ros_package_path('rpo_planning'), 'config/skill_cfgs')
    pull_cfg_file = osp.join(skill_config_path, 'pull') + ".yaml"
    pull_cfg = get_skill_cfg_defaults()
    pull_cfg.merge_from_file(pull_cfg_file)
    pull_cfg.freeze()

    grasp_cfg_file = osp.join(skill_config_path, 'grasp') + ".yaml"
    grasp_cfg = get_skill_cfg_defaults()
    grasp_cfg.merge_from_file(grasp_cfg_file)
    grasp_cfg.freeze()

    push_cfg_file = osp.join(skill_config_path, 'push') + ".yaml"
    push_cfg = get_skill_cfg_defaults()
    push_cfg.merge_from_file(push_cfg_file)
    push_cfg.freeze()

    # use our primitive for global config
    primitive_name = args.primitive
    if primitive_name == 'grasp':
        cfg = grasp_cfg
    elif primitive_name == 'pull':
        cfg = pull_cfg
    elif primitive_name == 'push':
        cfg = push_cfg
    else:
        raise ValueError('Primitive name not recognized')

    # create airobot interface
    yumi_ar = Robot('yumi_palms', pb=False, use_cam=False)
    yumi_ar.arm.right_arm.set_speed(100, 50)

    # create YumiReal interface
    yumi_gs = YumiReal(yumi_ar, cfg)

    from IPython import embed
    embed()
    # yumi_ar.arm.right_arm.set_jpos(yumi_ar.arm.right_arm.cfgs.ARM.HOME_POSITION, wait=False)
    # yumi_ar.arm.left_arm.set_jpos(yumi_ar.arm.left_arm.cfgs.ARM.HOME_POSITION, wait=False)
    # _, _ = yumi_gs.move_to_joint_target_mp(yumi_ar.arm.right_arm.cfgs.ARM.HOME_POSITION, yumi_ar.arm.left_arm.cfgs.ARM.HOME_POSITION, execute=True)
    # _, _ = yumi_gs.move_to_joint_target_mp(pull_cfg.RIGHT_INIT, pull_cfg.LEFT_INIT, execute=True)
    # _, _ = yumi_gs.move_to_joint_target_mp(push_cfg.RIGHT_INIT, push_cfg.LEFT_INIT, execute=True)
    # _, _ = yumi_gs.move_to_joint_target_mp(grasp_cfg.RIGHT_INIT, grasp_cfg.LEFT_INIT, execute=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--primitive',
        type=str,
        default='pull',
        help='which primitive to plan')

    args = parser.parse_args()
    main(args)
