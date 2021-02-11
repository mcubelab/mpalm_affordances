from collections import namedtuple

rpo_transition = namedtuple(
    'RPOTransition',
    [
        'node',
        'observation',
        'action',
        'reward',
        'achieved_goal',
        'desired_goal',
        'done'
    ])