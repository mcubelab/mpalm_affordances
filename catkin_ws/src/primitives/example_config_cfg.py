from yacs.config import CfgNode as CN 

_C = CN()

# _C.SUBGOAL_TIMEOUT = 40
# _C.TIMEOUT = 45
_C.SUBGOAL_TIMEOUT = 120
_C.TIMEOUT = 120

_C.OBJECT_INIT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_C.OBJECT_FINAL = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

_C.PALM_RIGHT = [0.0, -0.08, 0.025, 1.0, 0.0, 0.0, 0.0]
_C.PALM_LEFT = [-0.0672, -0.250, 0.205, 0.955, 0.106, 0.275, 0.0]

_C.TIP_TO_WRIST_TF = [0.0, 0.071399, -0.14344421, 0.0, 0.0, 0.0, 1.0]
# _C.WRIST_TO_TIP_TF = [0.0, -0.071399, 0.14344421, 0.0, 0.0, 0.0, 1.0]
_C.WRIST_TO_TIP_TF = [0.0, -0.0714, 0.15, 0.0, 0.0, 0.0, 1.0]

# starter poses for the robot, default is robot home position
_C.RIGHT_INIT = [0.413, -1.325, -1.040, -0.053, -0.484, 0.841, -1.546]
_C.LEFT_INIT = [-0.473, -1.450, 1.091, 0.031, 0.513, 0.77, -1.669]

def get_cfg_defaults():
    return _C.clone()
