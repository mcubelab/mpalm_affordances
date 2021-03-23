from yacs.config import CfgNode as CN
import numpy as np

_C = CN()
_C.EASY_PROBLEMS = 'easy_rearrangement_problems'
_C.MEDIUM_PROBLEMS = 'medium_rearrangement_problems'
_C.HARD_PROBLEMS = 'hard_rearrangement_problems'
_C.ASSETS_DIR = 'assets'

_C.DEFAULT_SCENE_POINTCLOUD_FILE = 'default_shelf_scene_pcd.npz'

def get_task_cfg_defaults():
    return _C.clone()
