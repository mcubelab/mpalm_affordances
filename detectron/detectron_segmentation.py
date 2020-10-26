import os, os.path as osp
import sys
import signal
import time
import argparse
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import numpy as np
import cv2
import random
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


def make_prediction(args, obs_file, predictor):
    while True:
        try:
            img = cv2.imread(obs_file)
            break
        except:
            pass
        time.sleep(0.01)
        
    outputs = predictor(img)

    # TODO process outputs
    pred = outputs

    # write outputs to filesystem
    pred_fname = osp.join(args.prediction_dir, obs_file.split('.jpg')[0] + '.npz')
    np.savez(
        pred_fname,
        pred=pred
    )
    os.remove(osp.join(args.observation_dir, obs_file))


def signal_handler(sig, frame):
    print('Exit')
    sys.exit(0)


def main(args):
    # setup segmentation model
    cfg = get_cfg()
    cfg.merge_from_file('./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
    cfg.MODELS.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = 'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'
    predictor = DefaultPredictor(cfg)

    # set up obs/pred dirs
    if not osp.exists(args.prediction_dir):
        os.makedirs(args.prediction_dir)
    if not osp.exists(args.observation_dir):
        os.makedirs(args.observation_dir)

    signal.signal(signal.SIGINT, signal_handler)

    while True:
        obs_fnames = os.listdir(args.observation_dir)
        observation_available = len(obs_fnames) > 0
        if observation_available:
            for fname in obs_fnames:
                if fname.endswith('.jpg'):
                    time.sleep(0.5)
                    make_prediction(args, fname, predictor)
        time.sleep(0.01)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--observation_dir', type=str, default='/tmp/detectron/observations')
    parser.add_argument('--prediction_dir', type=str, default='/tmp/detectron/predictions')

    args = parser.parse_args()
    main(args)