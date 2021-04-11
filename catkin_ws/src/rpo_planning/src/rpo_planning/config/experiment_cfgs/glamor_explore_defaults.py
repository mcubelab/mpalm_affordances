from yacs.config import CfgNode as CN

_C = CN()

_C.exploration = CN()
_C.exploration.mode = 'epsilon_greedy_decay'
_C.exploration.start_epsilon = 0.5
_C.exploration.decay_rate = 0.9

_C.relabeling = CN()
_C.relabeling.use_relabeling = True
_C.relabeling.relabeling_ratio = 0.5

_C.pretraining = CN()
_C.pretraining.use_pretraining = True
_C.pretraining.num_pretrain_epoch = 10

def get_glamor_explore_defaults():
    return _C.clone()
