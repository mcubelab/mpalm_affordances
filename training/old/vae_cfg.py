from yacs.config import CfgNode as CN

_C = CN()

_C.ENCODER_HIDDEN_LAYERS_MLP = [512, 256, 128, 64]
_C.DECODER_HIDDEN_LAYERS_MLP = [512, 256, 128, 64]


# _C.ENCODER_HIDDEN_LAYERS_MLP = [64, 32]
# _C.DECODER_HIDDEN_LAYERS_MLP = [64, 32]


def get_vae_defaults():
    return _C.clone()
