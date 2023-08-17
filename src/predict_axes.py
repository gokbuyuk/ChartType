from os.path import join, normpath
from keras.models import load_model
import os
import cv2
import numpy as np
from chartparams import CHART_PARAMS as PARAMS

MODEL_DIR = '../models'
IMG_DIR = '../data/raw/train/images'
ANNO_DIR = '../data/interim/train/annotations'


def predict_axis_bb_images_tf(
    img,
    basename = 'axislabelbounds',
    seriesname = PARAMS['series'],
    modeldir = MODEL_DIR,
    thumbx=PARAMS['thumbx'],
    thumby=PARAMS['thumby'],
    show=True):
    '''
    Convenience function that reads compiled data, creates, trains and saves an ML model,
    '''
    modelinfile = join(modeldir, basename + '_' + seriesname + ".h5")
    print(f"Reading model from {modelinfile}")
    model = load_model(modelinfile)
    resized_image = cv2.resize(img, (thumbx, thumby))
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = (gray - 127.5) / 127.5
    gray = np.expand_dims(gray, axis=-1) # add channel dimension
    gray = np.expand_dims(gray, axis=0) # add batch dimension
    # Make a prediction using the model
    pred = model.predict(gray)[0]
    # Extract the predicted bounding box coordinates
    lefthoriz, tophoriz, righthoriz, bottomhoriz, \
        leftvert, topvert, rightvert, bottomvert, \
        axis_x_type, axis_y_type = pred
    axis_x_type = round(axis_x_type)
    axis_y_type = round(axis_y_type)
    # Draw a rectangle on the image
    # resized_image = cv2.rectangle(resized_image, (lefthoriz, tophoriz), (righthoriz, bottomhoriz), (255, 0, 0), 2)
    # resized_image = cv2.rectangle(resized_image, (leftvert, topvert), (rightvert, bottomvert), (255, 0, 0), 2)
    # if show:
    #     cv2.imshow("axis " + bname, resized_image)
    #     cv2.waitKey(0)

    lefthoriz *= img.shape[1] # / thumbx
    tophoriz *= img.shape[0] # / thumby
    righthoriz *= img.shape[1] # / thumbx
    bottomhoriz *= img.shape[0] # / thumby
    leftvert *= img.shape[1] # / thumbx
    topvert *= img.shape[0] # / thumby
    rightvert *= img.shape[1] # / thumbx
    bottomvert *= img.shape[0] # / thumby
    # print("x-axis rectangle:", (int(lefthoriz), int(tophoriz)),  (int(righthoriz), int(bottomhoriz)))
    # print("y-axis rectangle:", (int(leftvert), int(topvert)),  (int(rightvert), int(bottomvert)))
    colors = {
        0:(0,255,0),
        1:(0,0,255)}
    img2 = cv2.rectangle(img, (int(lefthoriz), int(tophoriz)), (int(righthoriz), int(bottomhoriz)), colors[axis_x_type], 2)
    img2 = cv2.rectangle(img2, (int(leftvert), int(topvert)), (int(rightvert), int(bottomvert)), colors[axis_y_type], 2)
    if show:
        cv2.imshow("predicted axes ", img2)
        cv2.waitKey(0)
    return {'horizontal':(lefthoriz, tophoriz, righthoriz, bottomhoriz),
        'vertical':(leftvert, topvert, rightvert, bottomvert)}

def run_predict_axis_bb_images_tf(
    imgdir = IMG_DIR,
    annodir = ANNO_DIR,
    basename = 'axislabelbounds',
    seriesname = PARAMS['series'],
    modeldir = MODEL_DIR,
    n=30,
    thumbx=PARAMS['thumbx'],
    thumby=PARAMS['thumby'],
    show=True):
    '''
    Convenience function that reads compiled data, creates, trains and saves an ML model,
    '''
    modelinfile = join(modeldir, basename + '_' + seriesname + ".h5")
    print(f"Reading model from {modelinfile}")
    model = load_model(modelinfile)
    print(model)

    imgnames = os.listdir(imgdir)
    for i in range(min(n,len(imgnames))):
        imgname = imgnames[i]
        bname = imgname[:-4] # remove ending .jpg
        annoname = bname + '.json'
        print(imgname, bname)
        imgfile = join(imgdir, imgname)
        annofile = join(annodir, annoname)
        assert os.path.exists(imgfile)
        assert os.path.exists(annofile)

        # Load the test image
        img = cv2.imread(imgfile)
        resized_image = cv2.resize(img, (thumbx, thumby))
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        gray = (gray - 127.5) / 127.5
        print(gray)
        gray = np.expand_dims(gray, axis=-1) # add channel dimension
        gray = np.expand_dims(gray, axis=0) # add batch dimension

        # Make a prediction using the model
        pred = model.predict(gray)[0]
        print('raw prediction:', pred)
        # Extract the predicted bounding box coordinates
        lefthoriz, tophoriz, righthoriz, bottomhoriz, \
            leftvert, topvert, rightvert, bottomvert, \
            axis_x_type, axis_y_type, origin_x, origin_y = pred
        # Draw a rectangle on the image
        # resized_image = cv2.rectangle(resized_image, (lefthoriz, tophoriz), (righthoriz, bottomhoriz), (255, 0, 0), 2)
        # resized_image = cv2.rectangle(resized_image, (leftvert, topvert), (rightvert, bottomvert), (255, 0, 0), 2)
        # if show:
        #     cv2.imshow("axis " + bname, resized_image)
        #     cv2.waitKey(0)

        lefthoriz *= img.shape[1] # / thumbx
        tophoriz *= img.shape[0] # / thumby
        righthoriz *= img.shape[1] # / thumbx
        bottomhoriz *= img.shape[0] # / thumby
        leftvert *= img.shape[1] # / thumbx
        topvert *= img.shape[0] # / thumby
        rightvert *= img.shape[1] # / thumbx
        bottomvert *= img.shape[0] # / thumby
        axis_x_type = round(axis_x_type)
        axis_y_type = round(axis_y_type)
        origin_x *= img.shape[1]
        origin_y *= img.shape[0] # scale from relative coordinates to absolute
        colors = {
            0:(0,255,0),
            1:(0,0,255),
            2:(255,0,0)}
        print("x-axis rectangle:", (int(lefthoriz), int(tophoriz)),  (int(righthoriz), int(bottomhoriz)))
        print("y-axis rectangle:", (int(leftvert), int(topvert)),  (int(rightvert), int(bottomvert)))
        img2 = cv2.rectangle(img, (int(lefthoriz), int(tophoriz)), (int(righthoriz), int(bottomhoriz)), colors[axis_x_type], 2)
        img2 = cv2.rectangle(img2, (int(leftvert), int(topvert)), (int(rightvert), int(bottomvert)), colors[axis_y_type], 2)
        img2 = cv2.circle(img2,(int(origin_x), int(origin_y)), 10, colors[2])
        if show:
            cv2.imshow("axis " + bname, img2)
            cv2.waitKey(0)

if __name__ == '__main__':
    run_predict_axis_bb_images_tf()