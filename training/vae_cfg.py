from yacs.config import CfgNode as CN

_C = CN()

_C.ENCODER_HIDDEN_LAYERS_MLP = [64, 64]
_C.DECODER_HIDDEN_LAYERS_MLP = [64, 64]


def get_vae_defaults():
    return _C.clone()
