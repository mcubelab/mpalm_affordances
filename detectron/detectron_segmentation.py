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
# import cv2
# from scipy.misc import imread
from PIL import Image
import random
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog


def make_prediction(args, obs_file, predictor):
    print('Starting prediction, getting image')
    while True:
        try:
            # img = cv2.imread(obs_file)
            # img = imread(obs_file)
            img = np.asarray(Image.open(osp.join(args.observation_dir, obs_file)))
            break
        except:
            pass
        time.sleep(0.01)
        
    print('Got image, making prediction')
    outputs = predictor(img)
    masks = outputs['instances'].pred_masks.data.cpu().numpy()
    bboxes = outputs['instances'].pred_boxes.tensor.data.cpu().numpy()

    # TODO process outputs
    pred = masks

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
    # cfg.merge_from_file('/home/robot2/detectron2_repo/detectron2/config/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    # cfg.MODEL.WEIGHTS = 'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    # set up obs/pred dirs
    if not osp.exists(args.prediction_dir):
        os.makedirs(args.prediction_dir)
    if not osp.exists(args.observation_dir):
        os.makedirs(args.observation_dir)

    signal.signal(signal.SIGINT, signal_handler)

    print('Starting loop')
    while True:
        obs_fnames = os.listdir(args.observation_dir)
        observation_available = len(obs_fnames) > 0
        if observation_available:
            print('Observation available')
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
