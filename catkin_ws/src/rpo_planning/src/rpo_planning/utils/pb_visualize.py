import time
import pybullet as p


class GoalVisual():
    def __init__(self, trans_box_lock, object_id, pb_client,
                 goal_init, show_init=True):
        self.trans_box_lock = trans_box_lock
        self.pb_client = pb_client
        self.update_goal_obj(object_id)

        if show_init:
            self.update_goal_state(goal_init)

    def visualize_goal_state(self):
        while True:
            self.trans_box_lock.acquire()
            p.resetBasePositionAndOrientation(
                self.object_id,
                [self.goal_pose[0], self.goal_pose[1], self.goal_pose[2]],
                self.goal_pose[3:],
                physicsClientId=self.pb_client)
            self.trans_box_lock.release()
            time.sleep(0.01)

    def update_goal_state(self, goal):
        self.trans_box_lock.acquire()
        self.goal_pose = goal
        p.resetBasePositionAndOrientation(
            self.object_id,
            [self.goal_pose[0], self.goal_pose[1], self.goal_pose[2]],
            self.goal_pose[3:],
            physicsClientId=self.pb_client)
        self.trans_box_lock.release()

    def update_goal_obj(self, obj_id):
        self.object_id = obj_id
        self.color_data = p.getVisualShapeData(obj_id)[0][7]

    def hide_goal_obj(self):
        color = [self.color_data[0],
                 self.color_data[1],
                 self.color_data[2],
                 0]
        p.changeVisualShape(self.object_id, -1, rgbaColor=color)

    def show_goal_obj(self):
        p.changeVisualShape(self.object_id, -1, rgbaColor=self.color_data)
