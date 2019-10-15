from yacs.config import CfgNode as CN 

_C = CN()

_C.OBJECT_INIT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_C.OBJECT_FINAL = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

_C.PALM_RIGHT = [0.0, -0.08, 0.025, 1.0, 0.0, 0.0, 0.0]
_C.PALM_LEFT = [-0.0672, -0.250, 0.205, 0.955, 0.106, 0.275, 0.0]

_C.TIP_TO_WRIST_TF = [0.0, 0.071399, -0.14344421, 0.0, 0.0, 0.0, 1.0]


def get_cfg_defaults():
    return _C.clone()
