# People detector for AVA-Google. Process many videos
__author__ = "Manuel J Marin Jimenez"

import detectron2
import os
import os.path as osp
import subprocess
import sys
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import pickle as pkl

import random
import argparse

from os.path import expanduser
homedir = expanduser("~")

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


def load_pkl(in_pkl):
    with open(in_pkl, 'rb') as fid:
        out_pkl = pkl.load(fid, encoding='latin1')

    return out_pkl


def save_pkl(in_pkl, mydict):
    output = open(in_pkl, 'wb')
    pkl.dump(mydict, output)
    output.close()


def make_if_not_exist(path):
    if not osp.exists(path):
        os.makedirs(path)


def make_if_not_exist_file(file):
    if not osp.isdir(osp.dirname(file)):
        os.system('mkdir -p ' + osp.dirname(file))


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]



# Input arguments
parser = argparse.ArgumentParser(description='Run a detector')
parser.add_argument('--case', type=str, required=False, default='val',
                    help='train|val')
parser.add_argument("--verbose", type=int,
                    nargs='?', required=False, default=1,
                    help="Whether to enable verbosity of output")
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
case = args.case
verbose = args.verbose
debug = args.debug

#im = cv2.imread("./input.jpg")

# cv2.imshow("Input image", im)
# cv2.waitKey(-1)

# Configure detector
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
# COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml
#cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.45  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

# Prepare directories
#case = 'train'

# framesdir = homedir+'/databases/AVAv2.2/Frames1s/'+case
# videosdir = homedir+'/databases/AVAv2.2/Clips1s/'+case
# outputdirbase = homedir+'/experiments/AVAv2.2/detectron2-detections/'+case
# #verbose = 1
# skip_if_done = True
framesdir = "/media/hdd/adrien/Ava_v2.2/frames/train"
outputdirbase = "/home/acances/Code/ava/new_detections"

# l_movies = get_immediate_subdirectories(framesdir) #(videosdir)
# if debug:
#     l_movies = ["-5KQ66BBWC4"]
l_movies = ["26V9UzqSguo"]

random.shuffle(l_movies)
nmovies = len(l_movies)

for mix in range(0,nmovies):
    moviename = l_movies[mix]

    print("+ Movie: {}".format(moviename))

    moviedir = os.path.join(framesdir, moviename)#(videosdir, moviename)
    print(moviedir)

    # Find clips in movie
    l_clips = get_immediate_subdirectories(moviedir)
    random.shuffle(l_clips)
    nclips = len(l_clips)
    print("Found {:03d} clips".format(nclips))

    if nclips == 0:
    	import pdb; pdb.set_trace()

    for vix in range(0,nclips):
        vidname = l_clips[vix]
        print("\t"+vidname+"\r")
        video_path = os.path.join(moviedir, vidname)

        outputdir = os.path.join(outputdirbase, moviename)
        if verbose > 1:
        	print("Results will be saved in: {}".format(outputdir))

        # Prepare output name
        video_name = os.path.split(video_path)[-1]
        video_name = os.path.splitext(video_name)[0]
        outpklname = os.path.join(outputdir, video_name+"_dets.pkl")

        if os.path.exists(outpklname):
            if verbose > 1:
                print("File already exists. Skipping it!")
            continue

        if False:
            vcap = cv2.VideoCapture(video_path)
            if not vcap.isOpened():
                print("Could not open input video file {}".format(video_path))
            if verbose:
                print("Reading video file {}".format(video_path))
    
            if outputdir is None or outputdir == "":
                export = False
            else:
                export = True
                #make_if_not_exist(outputdir) # TODO
    
            # Begin detection
            dets_dict = {}
            frame_idx = 0
            while vcap.grab():
                if verbose > 1:
                    print("Generating detections for frame %s..." % frame_idx)
    
                # Read frame from video
                ret, frame = vcap.retrieve()
                if ret:
                    im = frame
                    if export:
                        imgname = os.path.join(outputdir, "{:06d}.jpg".format(frame_idx))
                        cv2.imwrite(imgname, frame)
    
                    outputs = predictor(im)
    
                    # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
                    outputs["instances"].pred_classes
                    outputs["instances"].pred_boxes
    
                    #v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
                    #v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                    #cv2.imshow("Results", v.get_image()[:, :, ::-1])
                    #cv2.waitKey(-1)
    
                    if export and (verbose > 1):
                        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
                        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                        imgname = os.path.join(outputdir, l_classes[cix]+"_{:06d}_det.jpg".format(frame_idx))
                        cv2.imwrite(imgname, v.get_image()[:, :, ::-1])
    
    
                    inst = outputs["instances"]
                    classes = inst.pred_classes.to("cpu").numpy()
                    boxes = inst.pred_boxes
                    scores = inst.scores.to("cpu").numpy()
    
                    dets = []
                    ndets = 0
                    for i in range(0, len(classes)):
                        if classes[i] == 0:
                            box = boxes[i].to("cpu")
                            M = np.array([box.tensor[0][0].item(), box.tensor[0][1].item(),
                                          box.tensor[0][2].item(), box.tensor[0][3].item(), scores[i]])
                            # if ndets == 0:
                            #     dets= M
                            # else:
                            dets.append(M)
    
                            ndets = ndets + 1
    
                    dets = np.array(dets)
                    print(dets.shape)
    
                    dets_dict[frame_idx] = dets
    
                    #cv2.destroyAllWindows()
                    print("Done!")
                else:
                    print("Could not read frame {}".format(frame_idx))
                frame_idx += 1
            vcap.release()
        # Use existing frames        
        else:
            framesdir_vid = os.path.join(framesdir, moviename, vidname)
            
            if outputdir is None or outputdir == "":
                export = False
            else:
                export = True
                make_if_not_exist(outputdir) # TODO
    
            # Begin detection
            dets_dict = {}
            frame_idx = 1
            while True:
                if verbose > 1:
                    print("Generating detections for frame %s..." % frame_idx)
    
                # Read frame from video
                framename = os.path.join(framesdir_vid, "{:05d}.jpg".format(frame_idx))
                frame = cv2.imread(framename)
                if frame is None:
                    break
                if not frame is None:
                    im = frame
                    if False:
                        imgname = os.path.join(outputdir, "{:05d}.jpg".format(frame_idx))
                        cv2.imwrite(imgname, frame)
    
                    outputs = predictor(im)
    
                    # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
                    outputs["instances"].pred_classes
                    outputs["instances"].pred_boxes
    
                    #v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
                    #v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                    #cv2.imshow("Results", v.get_image()[:, :, ::-1])
                    #cv2.waitKey(-1)
    
                    if export and (verbose > 1):
                        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
                        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                        imgname = os.path.join(outputdir, moviename+"_{:05d}_det.jpg".format(frame_idx))
                        cv2.imwrite(imgname, v.get_image()[:, :, ::-1])
    
    
                    inst = outputs["instances"]
                    classes = inst.pred_classes.to("cpu").numpy()
                    boxes = inst.pred_boxes
                    scores = inst.scores.to("cpu").numpy()
    
                    dets = []
                    ndets = 0
                    for i in range(0, len(classes)):
                        if classes[i] == 0:
                            box = boxes[i].to("cpu")
                            M = np.array([box.tensor[0][0].item(), box.tensor[0][1].item(),
                                          box.tensor[0][2].item(), box.tensor[0][3].item(), scores[i]])
                            # if ndets == 0:
                            #     dets= M
                            # else:
                            dets.append(M)
    
                            ndets = ndets + 1
    
                    dets = np.array(dets)
                    if verbose > 1:
                        print(dets.shape)
    
                    dets_dict[frame_idx] = dets
    
                    #cv2.destroyAllWindows()
                    #print("Done!")
                else:
                    print("Could not read frame {}".format(frame_idx))
                frame_idx += 1
            


        # Save detections to disk        
        save_pkl(outpklname, dets_dict)
        if vix == 0: print("LONGUEUR DU DICT:", len(dets_dict))
