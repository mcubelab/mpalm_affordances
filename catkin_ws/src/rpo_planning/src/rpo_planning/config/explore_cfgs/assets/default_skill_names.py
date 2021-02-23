from yacs.config import CfgNode as CN

_C = CN()
_C.SKILL_NAMES = [
  'pull_right', 'pull_left', 'push_right', 'push_left', 'grasp', 'grasp_pp'
]
_C.SURFACE_NAMES = [
  'table', 'shelf'
]

def get_skillset_cfg():
    C = _C.clone()
    C.SKILL_SET = []
    for i in range(len(C.SKILL_NAMES)):
      for j in range(len(C.SURFACE_NAMES)):
        skill_name = C.SKILL_NAMES[i] + '_' + C.SURFACE_NAMES[j]
        C.SKILL_SET.append(skill_name)

    return C