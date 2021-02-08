from yacs.config import CfgNode as CN
import numpy as np

_C = CN()
_C.EASY_PROBLEMS = 'easy_rearrangement_problems'
_C.MEDIUM_PROBLEMS = 'medium_rearrangement_problems'
_C.HARD_PROBLEMS = 'hard_rearrangement_problems'

def get_task_cfg_defaults():
    return _C.clone()
