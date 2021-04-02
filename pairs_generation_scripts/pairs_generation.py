import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math as m
from torchvision import transforms, utils
import cv2
import sys


def get_boxes(bboxes_file):
    boxes = []
    with open(bboxes_file, "r") as f:
        for line in f:
            box = list(map(float, line.strip().split(",")))[:-1]
            boxes.append(box)
    return boxes


def get_enlarged_box(box, alpha):
    # Enlarge the box area by 100*alpha percent while preserving
    # the center and the aspect ratio
    beta = 1 + alpha
    x1, y1, x2, y2 = box
    dx = x2 - x1
    dy = y2 - y1
    x1 -= (np.sqrt(beta) - 1)*dx/2
    x2 += (np.sqrt(beta) - 1)*dx/2
    y1 -= (np.sqrt(beta) - 1)*dy/2
    y2 += (np.sqrt(beta) - 1)*dy/2
    return x1, y1, x2, y2


def get_processed_frame(frame, box, w, h):
    # frame : 3 * W * H
    # (w, h) : dimensions of new frame
    # bbox input has to be in normalized coords:
    # image top-left corner (0,0), bottom-right (1, 1).

    C, W, H = frame.shape
    print("frame: \t", frame.shape)

    x1, y1, x2, y2 = box

    # Turn normalized coord into integer coords
    X1 = max(0, m.floor(x1*W))
    X2 = max(0, m.floor(x2*W))
    Y1 = max(0, m.ceil(y1*H))
    Y2 = max(0, m.ceil(y2*H))
    
    dX = X2 - X1
    dY = Y2 - Y1

    # Get the cropped bounding box
    boxed_frame = transforms.functional.crop(frame, X1, Y1, dX, dY)
    dX, dY = boxed_frame.shape[1:]
    print("boxed: \t", boxed_frame.shape)
    print("dX, dY: \t", dX, dY)

    # Compute size to resize the cropped bounding box to
    if dY/dX >= h/w:
        w_tild = m.floor(dX/dY*h)
        h_tild = h
    else:
        w_tild = w
        h_tild = m.floor(dY/dX*w)
    assert w_tild <= w
    assert h_tild <= h

    # Get the resized cropped bounding box
    resized_boxed_frame = transforms.functional.resize(boxed_frame, [w_tild, h_tild])
    print("resiz: \t", resized_boxed_frame.shape)

    # Put the resized cropped bounding box on a gray canvas
    new_frame = 127*torch.ones(C, w, h)
    i = m.floor((w - w_tild)/2)
    j = m.floor((h - h_tild)/2)
    new_frame[:, i:i+w_tild, j:j+h_tild] = resized_boxed_frame
    print("new: \t", new_frame.shape)

    return new_frame






if __name__ == "__main__":

    # bbox = [-0.1, -0.2, 0.7, 1.1]
    # im = torch.rand(3, 64, 52)
    # im2 = get_processed_frame(im, bbox, 30, 30)

    frame_file = sys.argv[1]
    bboxes_file = sys.argv[2]

    boxes = get_boxes(bboxes_file)

    # frame : H * W * 3
    frame = cv2.imread(frame_file)

    print(type(frame))
    print(frame.shape)
    # frame : 3 * W * H
    frame = frame.transpose(2, 1, 0)
    print(frame.shape)
    frame = torch.from_numpy(frame)

    w = 224
    h = 224
    alpha = 0.1
    box = boxes[0]
    box = get_enlarged_box(box, alpha)
    # new_frame : 3 * w * h
    new_frame = get_processed_frame(frame, box, w, h)

    # new_frame : h * w * 3
    new_frame = new_frame.numpy().transpose(2, 1, 0)
    cv2.imwrite("formatted_frame.jpg", new_frame)













