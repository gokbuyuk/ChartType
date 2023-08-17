
import math
import json
import os
import cv2
import copy
from os.path import join, normpath
import numpy as np
import pickle
from chartparams import CHART_PARAMS as PARAMS

DEBUG = False
SHOW = False

IMG_RAW_DIR = normpath('../data/raw/train/images')
IMG_INTERIM_DIR = normpath('../data/interim/train/images')
ANNO_RAW_DIR = normpath('../data/raw/train/annotations')
ANNO_INTERIM_DIR = normpath('../data/interim/train/annotations')
PROC_DIR = normpath("../data/processed/train")
assert os.path.exists(ANNO_RAW_DIR)


def build_features(
    outfilebase,
    interimdir = ANNO_INTERIM_DIR,
    imginterimdir = IMG_INTERIM_DIR,
    procdir = PROC_DIR,
    overwrite=True,
    thumbx=int(PARAMS['thumbx']),
    thumby=int(PARAMS['thumby']),
    params=PARAMS):
    assert os.path.exists(interimdir)
    assert os.path.exists(imginterimdir)
    jsons = os.listdir(interimdir)
    if not os.path.exists(procdir):
        print("Creating directory", procdir)
        os.makedirs(procdir)
    if not os.path.exists(IMG_INTERIM_DIR):
        print("Creating directory", IMG_INTERIM_DIR)
        os.makedirs(IMG_INTERIM_DIR)
    n = len(jsons)
    X = np.zeros([n, thumbx, thumby])
    # outcome variable:
    # 0-3: fractional bounding box of x-axis labels
    # 4-7: bounding box of y-axis labels
    # 8: flag if x-axis is numeric (0) or categorical (1)
    # 9: flag if y-axis is numeric (0) or categorical (1)
    # 10: x coordinate of chart origin point where axis meet
    # 11: y coordinate of chart origin point where axis meet
    y = np.zeros([n,12]) 
    for i in range(len(jsons)):
        base = jsons[i][:-len('.json')]

        print(f"working on file {i} {round(100*i/len(jsons),1)}%: {base}")
        imginfile = join(imginterimdir, base + '.jpg')
        assert os.path.exists(imginfile)
        jsonfile = join(interimdir, jsons[i])
        assert os.path.exists(jsonfile)
        img = cv2.imread(imginfile)
        imgoutfile = join(IMG_INTERIM_DIR, base + '.jpg')
        if not overwrite and os.path.exists(imgoutfile):
            print(f"File {imgoutfile} already exists, no overwrite")
            continue
        # Convert the color image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = (gray - 127.5) / 127.5
        X[i,:,:] = gray
        # print(gray) between 0 and 255
        print('params:', params)
        with open(jsonfile, 'r') as f:
            anno = json.load(f)
            axis_x = anno['axes']['x-axis']
            axis_y = anno['axes']['y-axis']
            axis_x_type = axis_x['values-type']
            axis_y_type = axis_y['values-type']
            ny, nx = anno['img_shape'][:2]
            # looking at one point like the first point is sufficient to find y-intercept, bc perfectly horizontal
            axis_x_yintercept = axis_x['ticks'][0]['tick_pt']['y'] 
            axis_y_xintercept = axis_y['ticks'][0]['tick_pt']['x']
            axis_x_yintercept /= ny # unit: fraction of image size
            axis_y_xintercept /= nx
            assert axis_y_xintercept < 1.0
            assert axis_x_yintercept < 1.0
            axis_x_type = params['axis-types'][axis_x_type] # convert from string to integer code
            axis_y_type = params['axis-types'][axis_y_type] # 0 for 'numerical', 1 for 'categorical', error otherwise
            print('@axis-x',axis_x_type)
            print('@axis-y',axis_y_type)
            leftx, topx, rightx, bottomx = anno['xbox']
            lefty, topy, righty, bottomy = anno['ybox']

            assert leftx <= nx and leftx >= 0
            assert rightx <= nx
            assert bottomx <= ny
            assert righty <= nx
            assert bottomy <= ny
            # rny, rnx = anno['img_resized_shape'][:2]
            # red_y, red_x = iny/rny, inx/rnx # reduction factors from original image to reduce image
            # reduced coordints of bounding box of x-axis:
            lx2 = leftx / nx
            rx2 = rightx / nx
            tx2 = topx / ny
            bx2 = bottomx / ny
            # reduced coordints of bounding box of y-axis:
            ly2 = lefty / nx # adjust left coordinate for reduction in x-resolution
            ry2 = righty / nx # adjust right coordinate for reduction in x-resolution
            ty2 = topy / ny
            by2 = bottomy / ny
            lst = [lx2, tx2, rx2, bx2, ly2, ty2, ry2, by2,axis_x_type, axis_y_type, axis_y_xintercept, axis_x_yintercept]
            print(lst)
            assert max(lst[:8]) <= 1.1
            assert min(lst[:8]) >= -0.1
            y[i,:] = lst
    assert X.min() >= -1
    assert X.max() <= 1.0
    assert y.min() >= 0.0
    assert y.max() <= 1.0
    X_out_file = join(PROC_DIR, outfilebase + '_X_train.pkl')
    y_out_file = join(PROC_DIR, outfilebase + '_y_train.pkl')
    print("writing X and y to files", X_out_file, y_out_file)
    X.tofile(X_out_file)
    y.tofile(y_out_file)
    with open(X_out_file, 'wb') as file:
        # use the pickle module to serialize the array and write it to the file
        pickle.dump(X, file)
    with open(y_out_file, 'wb') as file:
        # use the pickle module to serialize the array and write it to the file
        pickle.dump(y, file)

build_features(outfilebase='axislabelbounds')