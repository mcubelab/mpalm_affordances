import os, os.path as osp
import time
import numpy as np
import sys
import signal
import argparse
import lcm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm
import torch.nn as nn

from glamor.models import MultiStepDecoder, InverseModel
from skeleton_utils.utils import prepare_sequence_tokens
from skeleton_utils.language import SkillLanguage
from skeleton_utils.skeleton_globals import SOS_token, EOS_token, PAD_token
from lcm_inference.skeleton_predictor_lcm import GlamorSkeletonPredictorLCM

import rospkg
rospack = rospkg.RosPack()
sys.path.append(osp.join(rospack.get_path('rpo_planning'), 'src/rpo_planning/config/explore_cfgs'))
from default_skill_names import get_skillset_cfg

def signal_handler(sig, frame):
    """
    Capture exit signal from the keyboard
    """
    print('Exit')
    sys.exit(0)


def main(args):
    if args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load weights
    model_path = osp.join( 
        args.logdir, 
        args.model_path, 
        'model_' + str(args.model_number))

    print('Loading from model path: ' + str(model_path))
    checkpoint = torch.load(model_path)
    args_OLD = checkpoint['args']

    # create models
    # skill_lang = SkillLanguage('default')
    # skill_lang.index2skill = checkpoint['language_idx2skill']
    # skill_lang.skill2index = checkpoint['language_skill2idx']
    skill_lang = SkillLanguage('default')

    skillset_cfg = get_skillset_cfg()
    for skill_name in skillset_cfg.SKILL_SET:
        skill_lang.add_skill(skill_name)
    print('Skill Language: ')
    print(skill_lang.skill2index, skill_lang.index2skill)
    # server_params = SkeletonServerParams()
    server_params.set_skill2index(skill_lang.skill2index)
    server_proc = Process(target=serve_wrapper, args=(server_params,))
    server_proc.daemon = True
    server_proc.start()

    hidden_dim = args_OLD.latent_dim

    # in_dim is [x, y, z, x0, y0, z0]
    # out_dim mukst include total number of skills, pad token, and eos token
    in_dim = 6
    # out_dim = 9
    out_dim = len(skill_lang.index2skill.keys())

    inverse = InverseModel(in_dim, hidden_dim, hidden_dim)
    model = MultiStepDecoder(hidden_dim, out_dim)
    prior_model = MultiStepDecoder(hidden_dim, out_dim)

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        inverse.load_state_dict(checkpoint['inverse_model_state_dict'])
    except:
        print('NOT LOADING PRETRAINED WEIGHTS!!!')

    signal.signal(signal.SIGINT, signal_handler)
    model = model.eval()

    lc = lcm.LCM()
    lcm_predictor = GlamorSkeletonPredictorLCM(
        lc=lc,
        model=model,
        prior_model=prior_model,
        inverse_model=inverse,
        args=args_OLD,
        model_path=args.model_path,
        language=skill_lang)

    try:
        while True:
            lcm_predictor.predict_skeleton()
    except KeyboardInterrupt:
        pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='whether to use cuda or not')

    # Generic Parameters for Experiments
    parser.add_argument('--logdir', default='glamor_cachedir', type=str, help='location where log of experiments will be stored')
    parser.add_argument('--rundir', default='runs/glamor_runs')

    # model 
    parser.add_argument('--latent_dim', default=512, type=int, help='size of hidden representation')
    parser.add_argument('--max_seq_length', default=5, type=int, help='maximum sequence length')

    # loading weights
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--model_number', type=int, default=20000)

    args = parser.parse_args()

    main(args)

