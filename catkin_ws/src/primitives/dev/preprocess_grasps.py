for i in range(len(grasp_samples.collision_free_samples['gripper_poses'])):
    for j in range(len(grasp_samples.collision_free_samples['gripper_poses'][i])):
        palms = grasp_samples.collision_free_samples['gripper_poses'][i][j]

        palm_right, palm_left = palms[0], palms[1]
        normal_y_pose_prop = roshelper.transform_pose(normal_y_pose, palm_right)

        # print("x: " + str(normal_y_pose_prop.pose.position.x - palm_right.pose.position.x))
        # print("y: " + str(normal_y_pose_prop.pose.position.y - palm_right.pose.position.y))
        # print("z: " + str(normal_y_pose_prop.pose.position.z - palm_right.pose.position.z))
        # print("\n\n\n")

        theta = np.arccos(np.dot(roshelper.pose_stamped2list(normal_y_pose_prop)[:3], [0, 1, 0]))
        if theta > np.deg2rad(45) or theta < np.deg2rad(-45):
            print("x: " + str(normal_y_pose_prop.pose.position.x - palm_right.pose.position.x))
            print("y: " + str(normal_y_pose_prop.pose.position.y - palm_right.pose.position.y))
            print("z: " + str(normal_y_pose_prop.pose.position.z - palm_right.pose.position.z))
            print("\n\n\n")

            sample_yaw = np.random.random_sample() * np.deg2rad(90) - np.deg2rad(45)
            print("sample yaw: " + str(np.rad2deg(sample_yaw)))
            # sample_yaw = np.deg2rad(0)

            sample_q = common.euler2quat([0, 0, sample_yaw])

            sample_pose_world = roshelper.list2pose_stamped([0, 0, 0] + sample_q.tolist())
            palm_right_world = roshelper.convert_reference_frame(
                pose_source=palm_right,
                pose_frame_target=roshelper.unit_pose(),
                pose_frame_source=q0
            )
            palm_right_q = roshelper.pose_stamped2list(sample_pose_world)[3:]
            dq = common.quat_multiply(sample_q, common.quat_inverse(palm_right_q))

            sample_pose_trans = roshelper.convert_reference_frame(
                pose_source=sample_pose_world,
                pose_frame_target=palm_right_world,
                pose_frame_source=roshelper.unit_pose()
            )

            # dq = roshelper.pose_stamped2list(sample_pose_trans)[3:]

            print("rot: ", common.quat2euler(dq))

            q = common.quat_multiply(dq, roshelper.pose_stamped2list(sample_pose_world)[3:]).tolist()

            sample_palm_right = roshelper.list2pose_stamped(
                roshelper.pose_stamped2list(palm_right_world)[:3] + q
            )

            print(sample_palm_right)
            new_normal_y_pose_prop = roshelper.transform_pose(
                normal_y_pose, sample_palm_right)
            # print(new_normal_y_pose_prop)
            print("x: " + str(new_normal_y_pose_prop.pose.position.x -
                              sample_palm_right.pose.position.x))
            print("y: " + str(new_normal_y_pose_prop.pose.position.y -
                              sample_palm_right.pose.position.y))
            print("z: " + str(new_normal_y_pose_prop.pose.position.z -
                              sample_palm_right.pose.position.z))

            new_theta = np.arccos(
                np.dot(roshelper.pose_stamped2list(new_normal_y_pose_prop)[:3], [0, 1, 0]))
            
            print("new theta: " + str(np.rad2deg(new_theta)))
            print("\n\n\n")
