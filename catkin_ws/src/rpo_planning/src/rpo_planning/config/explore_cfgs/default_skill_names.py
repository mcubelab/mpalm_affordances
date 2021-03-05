from yacs.config import CfgNode as CN

_C = CN()
# _C.SKILL_NAMES = [
#   'pull_right', 'pull_left', 'push_right', 'push_left', 'grasp', 'grasp_pp'
# ]
_C.WITHIN_SURFACE_SKILLS = [
  'pull_right', 'pull_left', 'push_right', 'push_left'
]
_C.BETWEEN_SURFACE_SKILLS = [
  'grasp', 'grasp_pp'
]
_C.SURFACE_NAMES = [
  'table', 'shelf'
]
_C.SKILL_NAMES = _C.WITHIN_SURFACE_SKILLS + _C.BETWEEN_SURFACE_SKILLS

def get_skillset_cfg():
    C = _C.clone()
    C.SKILL_SET = []
    for i in range(len(C.SKILL_NAMES)):
      for j in range(len(C.SURFACE_NAMES)):
        skill_name = C.SKILL_NAMES[i] 
        if skill_name in C.BETWEEN_SURFACE_SKILLS:
          skill_name = skill_name + '_' + C.SURFACE_NAMES[j]

        C.SKILL_SET.append(skill_name)

    return C