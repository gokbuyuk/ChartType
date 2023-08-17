import copy
import os
import sys
import pytesseract
import easyocr
from PIL import Image
from os.path import join, normpath
import pandas as pd
import statistics
import numpy as np
import math
import cv2
# import time
from imutils.object_detection import non_max_suppression
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import json
from pandas.api.types import is_numeric_dtype
from scipy.signal import savgol_filter
from predict_axes import predict_axis_bb_images_tf
from chartparams import CHART_PARAMS as AXIS_PARAMS

SHOW=True
WAITKEY = True
DEBUG = True

RAW_DIR = '../data/raw'
IMG_DIR = join(RAW_DIR,'train/images')


def pearsonr_safe(x,y):
    '''
    Computes Pearson correlation with better error
    handling
    '''
    if not is_numeric_dtype(x) or not is_numeric_dtype(y):
        return 0.0, 1.0
    flags = ~ (pd.isna(x) | pd.isna(y))
    x2 = x[flags]
    y2 = y[flags]
    if len(x2) < 3:
        return 0.0, 1.0
    # print("running pearson r with:")
    # print(x2)
    # print(y2)
    return pearsonr(x2,y2)


# class Axis:
#
#     def pixel_to_value(pixel) -> float:



# def east_detect(image):
#     layerNames = [
#     	"feature_fusion/Conv_7/Sigmoid",
#     	"feature_fusion/concat_3"]
    
#     orig = image.copy()
    
#     if len(image.shape) == 2:
#         image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
#     (H, W) = image.shape[:2]
    
#     # set the new width and height and then determine the ratio in change
#     # for both the width and height: Should be multiple of 32
#     (newW, newH) = (320, 320)
    
#     rW = W / float(newW)
#     rH = H / float(newH)
    
#     # resize the image and grab the new image dimensions
#     image = cv2.resize(image, (newW, newH))
    
#     (H, W) = image.shape[:2]
    
#     net = cv2.dnn.readNet("model/frozen_east_text_detection.pb")
    
#     blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
#     	(123.68, 116.78, 103.94), swapRB=True, crop=False)
    
#     start = time.time()
    
#     net.setInput(blob)
    
#     (scores, geometry) = net.forward(layerNames)
    
#     (numRows, numCols) = scores.shape[2:4]
#     rects = []
#     confidences = []
#     # loop over the number of rows
#     for y in range(0, numRows):
#         # extract the scores (probabilities), followed by the geometrical
#         # data used to derive potential bounding box coordinates that
#         # surround text
#         scoresData = scores[0, 0, y]
#         xData0 = geometry[0, 0, y]
#         xData1 = geometry[0, 1, y]
#         xData2 = geometry[0, 2, y]
#         xData3 = geometry[0, 3, y]
#         anglesData = geometry[0, 4, y]
    
#         for x in range(0, numCols):
#     		# if our score does not have sufficient probability, ignore it
#             # Set minimum confidence as required
#             if scoresData[x] < 0.5:
#                 continue
#     		# compute the offset factor as our resulting feature maps will
#             #  x smaller than the input image
#             (offsetX, offsetY) = (x * 4.0, y * 4.0)
#             # extract the rotation angle for the prediction and then
#             # compute the sin and cosine
#             angle = anglesData[x]
#             cos = np.cos(angle)
#             sin = np.sin(angle)
#             # use the geometry volume to derive the width and height of
#             # the bounding box
#             h = xData0[x] + xData2[x]
#             w = xData1[x] + xData3[x]
#             # compute both the starting and ending (x, y)-coordinates for
#             # the text prediction bounding box
#             endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
#             endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
#             startX = int(endX - w)
#             startY = int(endY - h)
#             # add the bounding box coordinates and probability score to
#             # our respective lists
#             rects.append((startX, startY, endX, endY))
#             confidences.append(scoresData[x])
                        
#     boxes = non_max_suppression(np.array(rects), probs=confidences)
#     # loop over the bounding boxes
#     for (startX, startY, endX, endY) in boxes:
#     	# scale the bounding box coordinates based on the respective
#     	# ratios
#     	startX = int(startX * rW)
#         startY = int(startY * rH)
#     	endX = int(endX * rW)
#     	endY = int(endY * rH)
#     	# draw the bounding box on the image
#     	cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
    
#     print(time.time() - start)
#     return orig

# def test_east_detect(imgdir=IMG_DIR):
#     infile = join(imgdir,'aaeeb3e6866d.jpg')
#     print("input file:", infile)
#     assert os.path.exists(infile)
#     image = cv2.imread(infile)
#     out_image = east_detect(image)
#     cv2.imshow('EAST output', out_image)
#     # cv2.waitKey(0)


def easy_ocr_to_table(x, filter=True) -> pd.DataFrame:
    '''
    Converts EasyOCR output (list with bounding box and content) to 
    data frame with columns left, top, right, bottom, content
    
    Args:
    x(list): A list data structure with bounding box and OCR information as
        returned by easyocr
    filter(bool): Iff true, filter output such that left coordinates are always
        smaller than right coordinates and top coordinates are alsoway smaller than bottom coordinates.
    
    Returns:
        pd.DataFrame: Returns a data frame with OCR result in typical fashion
            with columns left, top, right, bottom, width, height, text
    '''
    lefts = []
    rights = []
    tops = []
    bottoms = []
    contents = []
    confidences = []
    for i in range(len(x)):
        bb = x[i][0]
        if len(bb) != 4:
            raise ValueError("Cannot parse content from EasyOCR at item " + str(i))
        bb0 = bb[0]
        bb1 = bb[1]
        bb2 = bb[2]
        bb3 = bb[3]
        lefts.append(int(bb0[0]))
        tops.append(int(bb0[1]))
        rights.append(int(bb2[0]))
        bottoms.append(int(bb2[1]))
        contents.append(x[i][1])
        confidences.append(x[i][2])
        if lefts[len(lefts)-1] > rights[len(rights)-1]:
            print("# warning: strange bounding box:")
            print(bb, ':', x[i][1])
    result = pd.DataFrame(data={'left':lefts,'top':tops,'right':rights,"bottom":bottoms, 'text':contents, 'confidence':confidences})
    result['width'] = result['right'] - result['left']
    result['height'] = result['bottom'] - result['top']
    if filter:
        result = result[(result['width'] > 0) & (result['height'] > 0)].reset_index(drop=True)
    return result  



def overlap_length(min1, max1, min2, max2):
    '''
    Returns A number greater zero if two intervals [min1,max1), [min2,max2) are overlapping corresponding to length of overlapping region.
    min and max coordinates are INCLUSIVE counting
    '''
    r1 = max1 - min1 + 1
    r2 = max2 - min2 + 1
    result = 0
    if not is_overlapping(min1, max1, min2, max2):
        return result
    if min1 <= min2 and max1 >= max2:  # 1 encloses 2: return smaller length (region 2)
        result = abs(max2-min2+1)
    elif min2 <= min1 and max2 >= max1:  # 2 encloses 1: return smaller length (region 1)
        result = abs(max1-min1+1)
    else:
        result = min(max1-min2+1, max2-min1+1)
    assert result >= 0
    assert result <= r1
    assert result <= r2
    return result



def overlap_area(
    left1, top1, right1, bottom1,
    left2, top2, right2, bottom2):
    '''
    Computes area of overlap between 2 bounding boxes
    '''
    if np.isnan(left1) or np.isnan(left2) or np.isnan(right1) or np.isnan(right2) or np.isnan(top1) or np.isnan(top2) or not np.isnan(bottom1) or np.isnan(bottom2):
        return 0
    if left1 > right1 or left2 > right2 or top1 > bottom1 or top2 > bottom2:
        return 0
    assert not np.isnan(left1)
    assert not np.isnan(left2)
    assert not np.isnan(right1)
    assert not np.isnan(right2)
    assert not np.isnan(top1)
    assert not np.isnan(top2)
    assert not np.isnan(bottom1)
    assert not np.isnan(bottom2)
    # #Assert(left1 <= right1)
    # #Assert(left2 <= right2)
    # #Assert(top1 <= bottom1)
    # #Assert(top2 <= bottom2)
    return overlap_length(left1, right1, left2, right2) \
        * overlap_length(top1, bottom1, top2, bottom2)


def overlap_score_left_right(
    left1, top1, right1, bottom1,
    left2, top2, right2, bottom2,
    vertical_overlap_ths=0.5):
    '''
    Computes a score for how well two bounding boxes are plausible
    to be overlapping rectangles corresponding to reading from
    left to right
    '''
    h1 = bottom1 - top1
    h2 = bottom2 - top2
    area = overlap_area(
        left1, top1, right1, bottom1,
        left2, top2, right2, bottom2)
    if area <= 0.0:
        return 0.0
    if left2 <= left1:
        return 0.0
    voverlap = overlap_length(top1, bottom1, top2, bottom2)
    assert voverlap >= 0.0
    vrel1 = voverlap/h1
    vrel2 = voverlap/h2
    assert  vrel1 <= 1.0
    assert vrel2 <= 1.0
    if min(vrel1, vrel2) < vertical_overlap_ths:
        return 0.0
    return area


def ocr_to_chains(lefts, tops, rights, bottoms, fontheight,
        score_cutoff=0.0, chain_len_min=2, ids=None, id_values=None):
    '''
    Returns id chains of characters forming phrases
    
    Parameters:
    ocr(pd.DataFrame): dataframe with OCR results. Expects columns names: left, right, top, bottom
    '''
    chains = []
    n = len(lefts)
    if ids is not None:
        if len(ids) != n:
            raise ValueError("Internal Error: number of provided ids must match number of provided bounding boxes.")
    member_of = [-1]*n # for each bounding box, shows with chain it is part of
    successor_of = [-1]*n # id of bounding box that is immediate reading order succesor of a certain bounding box
    predecessor_of = [-1]*n
    if n < 2:
        return {'chains':chains}
    dmtx = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            dmtx[i,j] = overlap_score_left_right(
                lefts[i], tops[i], rights[i], bottoms[i],
                lefts[j], tops[j], rights[j], bottoms[j])
    df = pd.DataFrame(dmtx).reset_index().melt('index')
    df.columns = ['row', 'column', 'value']    
    df.sort_values(by='value', ascending=False, inplace=True, ignore_index=True)
    df = df[df['value'] > score_cutoff]
    # df.to_csv("tmp_ocr_to_chains.csv")
    used = set()
    chain = []
    for i in range(len(df)):
        id1 = df.at[i,'row']
        id2 = df.at[i,'column']
        assert lefts[id1] < lefts[id2]
        if successor_of[id1] >= 0: # left bounding box already used
            continue  # id
        if predecessor_of[id2] >= 0:
            continue  # id2 is already used
        successor_of[id1] = id2
        predecessor_of[id2] = id1
    
    for i in range(n):
        if predecessor_of[i] < 0: # no predecessor, must be start:
            chain = []
            k = i
            while k >= 0:
                chain.append(k)
                korig=k
                k = successor_of[k]
                assert k < 0 or lefts[k] > lefts[korig]
            if len(chain) >= chain_len_min:
                chains.append(chain)
    assert isinstance(chains, list)
    # print('to_chains lefts, tops, rights, bottoms',lefts, tops, rights, bottoms)
    # print(chains)
    for i in range(len(chains)):
        for j in range(len(chains[i])):
            # assert j == 0 or lefts[chains[i][j-1]] < lefts[chains[i][j]]
            if ids is not None:
                if chains[i][j] >= len(ids):
                    raise ValueError(f"Provided ids of length {len(ids)} are not covering needed length:", chains[i][j])
                chains[i][j] = ids[chains[i][j]]  # convert to external ids
    for i in range(len(chains)):
        # print("Chain",i,":", chains[i])
        # if id_values is not None:
        #    print("Number of provide external values:", len(id_values))
        # if ids is not None:
        #    print("Number of provide external ids:", len(ids))
        for j in range(1,len(chains[i])):
            if ids is None:
                assert lefts[chains[i][j-1]] < lefts[chains[i][j]]
            elif id_values is not None:
                id1 = chains[i][j-1]
                v1 = id_values[id1]
                id2 = chains[i][j]
                v2 = id_values[id2]
                assert v1 < v2

    return {'chains':chains}


def is_overlapping(min1, max1, min2, max2, allow_reorder=True):
    '''
    Returns True iff two intervals [min1,max1), [min2,max2) are overlapping.
    Coordinates (even max values) are in INCLUSIVE but zero-based counting
    '''
    assert min1 <= max1
    assert min2 <= max2
    if allow_reorder:
        if min1 > max1:
            return is_overlapping(max1,min1,min2, max2, allow_reorder=allow_reorder)
        if min2 > max2:
            return is_overlapping(min1,max1,max2, min2, allow_reorder=allow_reorder)          
    if (max2 < min1) or (max1 < min2):
        return False
    return True


def df_is_overlapping(tops, bottoms, mini, maxi):
    '''
    This function computes a list of boolean values that 
    indicate which rows are overlapping.
    '''
    n = len(tops)
    # mtx = np.array([n,2])
    # mtx[:,0] = tops
    # mtx[:,1] = bottoms
    # pairs = mtx.tolist()
    result=pd.Series([False]*n)
    for i in range(n):
        result[i] = is_overlapping(tops[i],bottoms[i],mini,maxi)
    return result


def tilts_from_ocr_poly(ocr, dist_rel_ths=2.0,
            overlap_rel_ths_max = 0.8,
            overlap_rel_ths_min = -0.5,
            lcol='left', rcol='right', tcol='top', bcol='bottom',
            chain_len_min=2):
    '''
    Computes approximated angle for each word of OCR result using
    polynomial fit of stringed together character boxes.
    If finds closest neighbor to the right and loeft of the current
    bounding box and computes the average tilt angle.

    Parameters:
    ocr(pd.DataFrame) A Pandas data frame with columns left, top, right, bottom, content
    row: An integer number for the row (bounding box) from table for which tilt angle to be computed
    dist_rel_ths: Relative distance threshold in terms of character height
    
    Returns (list[float]): Computed tilt angles (in radians) for bounding box
    '''
    n = len(ocr)
    angle = 0
    lefts = ocr[lcol]
    tops = ocr[tcol]
    rights = ocr[rcol]
    bottoms = ocr[bcol]
    heights = ocr[bcol] - ocr[tcol]
    nx = max(rights)
    ny = max(bottoms)    
    medheights = statistics.median(heights)  # median character height
    # find leftmost starts
    nygrid = int(ny/medheights)
    idcol = '_INDEX_'
    dcol  = '_DIST_'
    ocr[idcol] = range(len(ocr))
    # find potential chain starts
    chains = []
    ocr['angle'] = np.nan
    ocr['chain'] = np.nan
    for i in range(ny):
        ytop = i * medheights
        ybot = ytop + medheights
        overlaps = df_is_overlapping(tops, bottoms, ytop, ybot)
        ocr2 = ocr[overlaps].copy(deep=True).sort_values(by=[lcol])
        ocr2.index = range(len(ocr2))
        chains2 = ocr_to_chains(ocr2[lcol], ocr2[tcol],
            ocr2[rcol], ocr2[bcol], fontheight=medheights,
                ids=ocr2[idcol], id_values=ocr[lcol]).get('chains', [])
        assert isinstance(chains2,list)
        for j in range(len(chains2)):
            chain = chains2[j]
            for k in range(1,len(chain)):
                assert ocr.at[chain[k-1],lcol] < ocr.at[chain[k],lcol]
            chains.append(chains2[j])
    xvals = []
    yvals = []
    ocr2 = copy.deepcopy(ocr)
    angleresults = [np.nan]*len(ocr2)
    for i in range(len(chains)):
        xv = []
        yv = []
        idv = []
        tv=[]
        rv=[]
        bv=[]
        lv=[]
        contents = []
        for j in range(len(chains[i])):
            id = chains[i][j]
            x = ocr.at[id, 'midx']
            y = ocr.at[id, 'midy']
            tv.append(ocr.at[id, tcol])
            bv.append(ocr.at[id, bcol])
            lv.append(ocr.at[id, lcol])
            rv.append(ocr.at[id, rcol])
            contents.append(ocr.at[id, 'text'])
            idv.append(id)
            xv.append(x)
            yv.append(y)
        est = angle_estimates_poly(xv, yv)
        est_angles = est['angle'].round(3)
        est_y = est['y2']
        assert len(est_angles) == len(xv)
        assert len(est_angles) == len(idv)
        assert 'angle' in ocr2.columns
        tmpdf = pd.DataFrame(data={'x':xv, 'y':yv, 'y2':est_y, 'angle':est_angles, 'chain':i,'text':contents,
        lcol:lv,tcol:tv,rcol:rv, bcol:bv})
        # tmpdf.to_csv("tmp_chain_" + str(i) + ".csv")
        for j in range(len(idv)):
            id = idv[j]
            assert id < len(ocr2)
            assert j < len(est_angles)
            assert 'angle' in ocr2.columns
            ocr2.at[id,'angle'] = est_angles[j]
            angleresults[id] = est_angles[j]
    return angleresults


def _ocr_to_y_labels_from_id(x, y, id, xtol=10, texts=None) -> pd.Series:
    '''
    Returns indices corresponding to bounding boxes that appear
    to be labels of a y-axis

    Args:
        x(pd.Series): Vector of x-values
        y(pd.Series): Vector of y-values
        id(int): Start index

    Returns:
        pd.Series: Series of indices corresponding to x-values
    '''
    xi = x[id]
    yi = y[id]

    df = pd.DataFrame(data={'x':x, 'y':y})
    df = df.sort_values(by='y', ascending=True)
    df = df[(df['x'] >= xi-xtol)&(df['x'] <= xi+xtol)]
    df['index'] = df.index
    df = df.reset_index(drop=True)
    df['y2'] = df['y'].copy()
    n=len(df)
    df['y2'][:(n-1)] = df['y'][1:]
    df['dy'] = df['y2'] - df['y']
    if texts is not None:
        if DEBUG:
            print("Starting y-chain with", texts[id])
            print(df)
    return df['index'].iloc[::-1]


def _ocr_to_y_labels(x, y, texts=None) -> pd.Series:
    '''
    Returns indices corresponding to bounding boxes that appear
    to be labels of a y-axis

    Args:
        x(pd.Series): Vector of x-values
        y(pd.Series): Vector of y-values

    Returns:
        pd.Series: Series of indices corresponding to x-values
    '''
    result = []
    for i in range(len(x)):
        tmp = _ocr_to_y_labels_from_id(x, y, i, texts=None)
        if len(tmp) > len(result):
            result = tmp
    return result


def ocr_to_y_labels(lefts, tops, rights, bottoms, texts=None) -> pd.Series:
    '''
    Returns indices corresponding to bounding boxes that appear
    to be labels of a y-axis
    '''
    result = _ocr_to_y_labels(lefts, tops, texts=texts)
    result2 = _ocr_to_y_labels(rights, tops, texts=texts)
    if len(result2) > len(result):
        if DEBUG:
            print('top right better than top left:')
            print(result2)
            result = result2
    return result


def _ocr_to_x_labels_from_id(x, y, id, index=None, ytol=20) -> pd.Series:
    '''
    Returns indices corresponding to bounding boxes that appear
    to be labels of a x-axis

    Args:
        x(pd.Series): Vector of x-values
        y(pd.Series): Vector of y-values
        id(int): Start index

    Returns:
        pd.Series: Series of indices corresponding to x-values
    '''
    yi = y[id]
    df = pd.DataFrame(data={'x':x, 'y':y})
    if index is not None:
        df['index'] = index
    df = df.sort_values(by='x', ascending=True)
    df = df[(df['y'] >= yi-ytol)&(df['y'] <= yi+ytol)]
    df['index'] = df.index
    df = df.reset_index(drop=True)
    df['x2'] = df['x'].copy()
    n=len(df)
    tmp = df['x'][1:].tolist()
    tmp.append(pd.NA)
    df['x2'] = tmp # [:(n-1)] = df['x'][1:].tolist()
    df['dx'] = df['x2'] - df['x']
    return df # ['index'].iloc[::-1]


def _ocr_to_x_labels(x, y, index=None) -> pd.Series:
    '''
    Returns indices corresponding to bounding boxes that appear
    to be labels of a x-axis

    Args:
        x(pd.Series): Vector of x-values
        y(pd.Series): Vector of y-values

    Returns:
        pd.Series: Series of indices corresponding to x-values
    '''
    result = []
    for i in range(len(x)):
        tmp = _ocr_to_x_labels_from_id(x, y, i, index=index)
        if len(tmp) > len(result):
            result = tmp
    return result


def ocr_to_x_labels(df, bottom_frac=None) -> pd.Series:
    '''
    Returns indices corresponding to bounding boxes that appear
    to be labels of a x-axis

    Args:
        df: Input data frame with ocr results, containing
            columns like 'text', 'tlx', 'tly', 'trx', 'tly' and so forth
        bottom_frac(float): y-coordinates have to be higher than this fraction
            of the maximum y-coordinate. Defaults to None; try 0.7; still needs code fix if not None
    '''
    # result = _ocr_to_x_labels(lefts, tops)
    # filter such that x-labels are near bottom of image
    # find lowest position (highest y values):
    bottom_y = max(df['try'].max(), df['tly'].max(), df['try'].max(), df['bly'].max(), df['bry'].max())
    if bottom_frac is not None:
        df = df[df['try'] >= bottom_y * bottom_frac].reset_index(drop=False) # we keep index
    print("df before _ocr_to_x_labels:")
    print(df)
    if 'index' in df.columns:
        result = _ocr_to_x_labels(df['trx'], df['try'], index=df['index'])
    else:
        result = _ocr_to_x_labels(df['trx'], df['try'])  
    print("result of _ocr_to_x_labels:")
    print(result)
    # if len(result2) > len(result):
    #    result = result2
    return result


def rotate_image(image, angleInDegrees):
    '''
    rotates image
    See: <https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point>
    '''
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)

    rad = math.radians(angleInDegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])
    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return outImg, rot


def rotate_coordinates(x, y, angleInDegrees, center_x, center_y):
    '''
    rotates image
    See: <https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point>
    '''
    # h, w = image.shape[:2]
    img_c = (center_x, center_y)

    rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)

    rad = math.radians(angleInDegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    # b_w = int((h * abs(sin)) + (w * abs(cos)))
    # b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += - img_c[0] # ((b_w / 2) 
    rot[1, 2] += - img_c[1] # ((b_h / 2) 

    # Define a homogeneous transformation matrix
    homogeneous_matrix = np.vstack((rot, [0, 0, 1]))

    # Convert the 2D points to homogeneous coordinates
    homogeneous_coords = np.vstack((x, y, np.ones_like(x)))

    # Apply the rotation using matrix multiplication
    rotated_coords = np.dot(homogeneous_matrix, homogeneous_coords)
    # Convert back to 2D points
    x_rotated = rotated_coords[0]
    y_rotated = rotated_coords[1]
    # outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return x_rotated, y_rotated


def polygon_overlap_area(x1, y1, w, h, vertices):
    """
    Calculate the overlap area between a 2D quadrilateral defined by its vertices and a bounding box specified by its top-left 
    corner coordinates and dimensions.

    Args:
        x1 (float): The x-coordinate of the top-left corner of the bounding box.
        y1 (float): The y-coordinate of the top-left corner of the bounding box.
        w (float): The width of the bounding box.
        h (float): The height of the bounding box.
        vertices (List[Tuple[float, float]]): A list of four tuples representing the vertices of the quadrilateral. Each tuple
            should contain two float values, representing the x and y coordinates of the vertex.

    Returns:
        float: The overlap area between the quadrilateral and the bounding box, or 0 if there is no overlap.

    Raises:
        ValueError: If the number of vertices in the quadrilateral is not equal to 4.
        ValueError: If any vertex of the quadrilateral is outside of the bounding box.

    Example:
        # Define a bounding box with top-left corner at (0, 0), width 5 and height 5
        x1, y1, w, h = 0, 0, 5, 5

        # Define a quadrilateral with vertices at (1, 1), (3, 1), (4, 3), and (2, 3)
        vertices = [(1, 1), (3, 1), (4, 3), (2, 3)]

        # Calculate the overlap area between the quadrilateral and the bounding box
        overlap = overlap_area(x1, y1, w, h, vertices)

        # The expected overlap area is 3
        assert overlap == 3
    """
    # Determine bounding box vertices
    x2 = x1 + w
    y2 = y1 + h

    # Initialize overlap area
    area = 0

    # Check if any quadrilateral vertex is inside bounding box
    for vertex in vertices:
        if x1 <= vertex[0] <= x2 and y1 <= vertex[1] <= y2:
            area = abs(0.5 * sum((
                vertices[i][0] * vertices[(i+1)%4][1] - vertices[(i+1)%4][0] * vertices[i][1]
                for i in range(4)
            )))
            return area

    # Calculate area of trapezoids formed by intersecting line segments and bounding box
    for i in range(4):
        x1_, y1_ = vertices[i]
        x2_, y2_ = vertices[(i+1)%4]

        if x1_ < x1 and x2_ < x1:
            continue
        if x1_ > x2 and x2_ > x2:
            continue
        if y1_ < y1 and y2_ < y1:
            continue
        if y1_ > y2 and y2_ > y2:
            continue

        if x1_ == x2_:
            # Vertical line segment
            y3 = max(y1_, y1)
            y4 = min(y2_, y2)
            area += (y4 - y3) * (x2_ - x1_)

        elif y1_ == y2_:
            # Horizontal line segment
            x3 = max(x1_, x1)
            x4 = min(x2_, x2)
            area += (x4 - x3) * (y2_ - y1_)

        else:
            # Sloped line segment
            slope = (y2_ - y1_) / (x2_ - x1_)
            y3 = max(y1_, y1, (x1 - x1_) * slope + y1_)
            y4 = min(y2_, y2, (x2 - x1_) * slope + y1_)
            area += (y4 - y3) * (x2_ - x1_)

    return area


def test_polygon_overlap_area():
    # Test case 1: No overlap
    x1, y1, w, h = 0, 0, 2, 2
    vertices = [(4, 4), (4, 6), (6, 6), (6, 4)]
    expected_area = 0
    assert polygon_overlap_area(x1, y1, w, h, vertices) == expected_area

    # Test case 2: Entire quadrilateral is inside bounding box
    x1, y1, w, h = 0, 0, 5, 5
    vertices = [(2, 2), (2, 3), (3, 3), (3, 2)]
    expected_area = 1
    assert polygon_overlap_area(x1, y1, w, h, vertices) == expected_area

    # Test case 3: Quadrilateral overlaps partially with bounding box
    x1, y1, w, h = 2, 2, 5, 5
    vertices = [(3, 3), (3, 4), (4, 4), (4, 3)]
    expected_area = 1
    assert polygon_overlap_area(x1, y1, w, h, vertices) == expected_area

    # Test case 4: Quadrilateral overlaps with only one side of bounding box
    x1, y1, w, h = 2, 2, 5, 5
    vertices = [(1, 3), (1, 4), (4, 4), (4, 3)]
    expected_area = 2
    # assert polygon_overlap_area(x1, y1, w, h, vertices) == expected_area

    # Test case 5: Quadrilateral overlaps with two sides of bounding box
    x1, y1, w, h = 2, 2, 5, 5
    vertices = [(1, 2), (1, 4), (4, 4), (4, 1)]
    expected_area = 4.5
    # assert polygon_overlap_area(x1, y1, w, h, vertices) == expected_area


def image_to_data_with_angle(
    image:np.ndarray,
    angle:float,
    engine:str='kerasocr',
    config:str='--psm 11', output_type:str='data.frame',
    show:bool=SHOW,
    line_color:tuple=(255,0,0),
    thickness:int=1) -> pd.DataFrame:
    '''
    Runs ocr but first rotates image, then reverses rotation of OCR result
    
    Args:

    image: Image in PIL format
    angle: rotation angle in degree
    engine(str): The OCR engine (i.e. package) to use. One of:
        'easyocr', 'kraken', 'pytesseract'
    '''
    if not isinstance(image, np.ndarray):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image2, rot = rotate_image(image, angle)
    if show:
        cv2.imshow('rotated image', image2)
        if WAITKEY:
            cv2.waitKey(0)
    # results = pytesseract.image_to_data(image2, config=config, output_type='data.frame') # pytesseract.Output.DICT)
    # results = results[~results['text'].isna()].reset_index(drop=True)
    reader = easyocr.Reader(['en'])
    if engine == 'easyocr':
        ocr = reader.readtext(image2, paragraph=False) # , link_threshold=0.99, width_ths=0.01, batch_size=batch_size)
        ocr = easy_ocr_to_table(ocr)
    elif engine == 'kerasocr':
        import kerasocr # inhouse file
        ocr = kerasocr.image_to_kerasocr(image2)
        ocr = kerasocr.kerasocr_to_table(ocr)
    elif engine == 'kraken':
        import kraken
        # print(kraken.__version__)
        # Perform OCR
        ocr = kraken.rpred(image2) # .recognize
        print(ocr)
        assert False
    elif engine == 'pytesseract':
        pass # TODO
        assert False
    else:
        raise ValueError("The OCR engine specified via variable 'engine' needs to be one of: 'easyocr', 'kraken', 'pytesseract'")  
    if 'tlx' not in ocr.columns: # is already there for some engines like kerasocr
        ocr['tlx'] = ocr['left'].copy() # top left x coordinate
        ocr['blx'] = ocr['left'].copy()  # bottom left x coordinate
        ocr['trx'] = ocr['right'].copy() # top right x coordinate
        ocr['brx'] = ocr['right'].copy()  # bottom right x coordinate
        ocr['tly'] = ocr['top'].copy() # top left y coordinate
        ocr['bly'] = ocr['bottom'].copy()  # bottom left y coordinate
        ocr['try'] = ocr['top'].copy() # top right y coordinate
        ocr['bry'] = ocr['bottom'].copy()  # bottom right y coordinate
    center_x = rot[0,2]
    center_y = rot[1,2]
    # rotate back to original position:
    x2, y2 = rotate_coordinates(ocr['tlx'], ocr['tly'], -angle, center_x, center_y )
    ocr['tlx'] = x2
    ocr['tly'] = y2
    x2, y2 = rotate_coordinates(ocr['trx'], ocr['try'], -angle, center_x, center_y )
    ocr['trx'] = x2
    ocr['try'] = y2
    x2, y2 = rotate_coordinates(ocr['blx'], ocr['bly'], -angle, center_x, center_y )
    ocr['blx'] = x2
    ocr['bly'] = y2
    x2, y2 = rotate_coordinates(ocr['brx'], ocr['bry'], -angle, center_x, center_y )
    ocr['brx'] = x2
    ocr['bry'] = y2
    assert 'text' in ocr.columns
    ocr = ocr[~ocr['text'].isna()].reset_index(drop=True)
    ocr = ocr[ocr['text']!=''].reset_index(drop=True)
    if show:
        for i in range(len(ocr)):
            tl = (int(ocr.at[i,'tlx']), int(ocr.at[i,'tly']))
            tr = (int(ocr.at[i,'trx']), int(ocr.at[i,'try']))
            bl = (int(ocr.at[i,'blx']), int(ocr.at[i,'bly']))
            br = (int(ocr.at[i,'brx']), int(ocr.at[i,'bry']))
            cv2.line(image,tl, tr, line_color, thickness)
            cv2.line(image,tr, br, line_color, thickness)
            cv2.line(image,bl, br, line_color, thickness)
            cv2.line(image,tl, bl, line_color, thickness)
        cv2.imshow('orig image with boxes', image)
        if WAITKEY:
            cv2.waitKey(0)
    return ocr


def test_ocr_to_x_labels(imgdir=join(RAW_DIR,'train/images'),
    batch_size=1):
    infile = join(imgdir,'aaeeb3e6866d.jpg')
    # print("input file:", infile)
    assert os.path.exists(infile)
    # Open the image file
    image = Image.open(infile)
    # Display the image
    # image.show()
    # Use pytesseract to extract the text from the image
    # results = pytesseract.image_to_data(=image, config='--psm 11', output_type='data.frame') # pytesseract.Output.DICT)
    angle = -45.0
    config='--psm 11'
    results = image_to_data_with_angle(image, angle, config=config, output_type='data.frame') # pytesseract.Output.DICT)
    
    results = results.drop(columns=['page_num', 'conf'], errors='ignore')
    results = results[~results['text'].isna()].reset_index(drop=True)
    results['right'] = results['left'] + results['width']
    results['bottom'] = results['top'] + results['height']
    # print(results.to_string())
    # print('detect x-labels:')
    result2 = ocr_to_x_labels(results)
    # print(result2)
    result3 = results.filter(items=result2['index'], axis=0)
    # print(result3['text'])
    assert len(result3) == 10
    # print(result3['text'].tolist())
    assert result3['text'].tolist()[0] == 'Less'
    assert result3['text'].tolist()[9] == 'Madagascar'


def test_ocr_to_y_labels(imgdir=join(RAW_DIR,'train/images'),
    batch_size=1):
    infile = join(imgdir,'aaeeb3e6866d.jpg')
    # print("input file:", infile)
    assert os.path.exists(infile)
    # Open the image file
    image = Image.open(infile)
    # Display the image
    # image.show()
    # Use pytesseract to extract the text from the image
    angle = 0.0
    config='--psm 11'
    results = pytesseract.image_to_data(image, config=config, output_type='data.frame') # pytesseract.Output.DICT)
    # results = image_to_data_with_angle(image, angle, config=config, output_type='data.frame') # pytesseract.Output.DICT)
    results = results.drop(columns=['block_num','par_num','line_num','word_num'], errors='ignore')
    results = results.drop(columns=['page_num', 'conf'], errors='ignore')
    results = results[~results['text'].isna()].reset_index(drop=True)
    results['right'] = results['left'] + results['width']
    results['bottom'] = results['top'] + results['height']
    # print("pytess results:")
    # print(results.to_string())
    # print('detect y-labels:')
    result2 = ocr_to_y_labels(results['left'], results['top'],
        results['right'], results['bottom'], texts=results['text'])
    # print('result2:')
    # print(result2)
    result3 = results.filter(items=result2, axis=0)
    # print(result3['text'])
    assert len(result3) == 6
    # print(result3['text'].tolist())
    assert result3['text'].tolist()[0] == '20'
    assert result3['text'].tolist()[5] == '120'


def get_image_dimensions(image):
    '''
        Returns the dimensions (width, height) of an image irrespective of whether it is from PIL or numpy.

    Args:
        image: An image object, either a numpy.ndarray or PIL.Image.Image.

    Returns:
        A tuple containing the dimensions (width, height) of the image.

    Raises:
        TypeError: If the image object is not of type numpy.ndarray or PIL.Image.Image.
    '''
    if isinstance(image, np.ndarray):
        return image.shape[:2][::-1]
    elif isinstance(image, Image.Image):
        return image.size[::-1]
    else:
        raise TypeError("Unsupported image type. Only numpy.ndarray and PIL.Image.Image are supported.")


def crop_pil_image(image, box):
    '''
    Crop image assuming PIL type of image. Ensures that bounding
    box within limits of image
    '''
    width, height = image.size
    left, top, right, bottom = box
    left = max(0, int(left))
    top = max(0, int(top))
    if right > width:
        right = width
    if height > bottom:
        bottom = height # min(height, int(bottom))
    print(f'Crop the image with dimensions {width},{height} to the specified region',
            (left, top, right, bottom))
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image

def posint_tuple(x, bounds = None) -> tuple:
    '''
    Converts array or list to tuple of same length
    with non-negative integers
    '''
    # int_tuple = tuple(int(x) for x in float_list)
    my_tuple = tuple([int(num) if num >= 0 else 0 for num in x])
    return my_tuple


def expand_bounding_box(bbox, margin, img_width, img_height):
    left, top, right, bottom = bbox
    left_margin = min(margin, left)
    top_margin = min(margin, top)
    right_margin = min(margin, img_width - right)
    bottom_margin = min(margin, img_height - bottom)
    
    return (
        left - left_margin,
        top - top_margin,
        right + right_margin,
        bottom + bottom_margin
    )


def crop_image(image, box):
    """
    Crops an image irrespective of whether it is a numpy array or a PIL image.

    Args:
        image: An image object, either a numpy.ndarray or PIL.Image.Image.
        box: A tuple of the form (left, upper, right, lower) that defines the region of the image to be cropped.

    Returns:
        A cropped image object of the same type as the input image.

    Raises:
        TypeError: If the image object is not of type numpy.ndarray or PIL.Image.Image.
    """
    print("starting crop-image with box", box)
    box = posint_tuple(box)
    print("for image with dimensions", get_image_dimensions(image))
    if isinstance(image, np.ndarray):
        return image[box[1]:box[3], box[0]:box[2], ...]
    elif isinstance(image, Image.Image):
        print("cropping PIL image with", box)
        return crop_pil_image(image, box)
    else:
        raise TypeError("Unsupported image type. Only numpy.ndarray and PIL.Image.Image are supported.")


def display_image(image, title:str='image', waitkey:bool=WAITKEY):
    '''
    Displays image to screen and waits for keystroke
    '''
    if isinstance(image, np.ndarray):
        # image = Image.fromarray(image)
        cv2.imshow(title, np.array(image))
        if waitkey:
            cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        image.show()
    # elif not isinstance(image, Image.Image):
    #     raise TypeError("Unsupported image type. Only numpy.ndarray and PIL.Image.Image are supported.")
    # cv2.imshow(title, np.array(image))
    # if waitkey:
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()

# def is_scalar(x):
#    return isinstance(x, numbers.Number)

def is_iterable(x):
    try:
        iter(x)
        return True
    except TypeError:
        return False


def calc_ocr_overlaps(ocr:pd.DataFrame, rectangle:tuple):
    '''
    Computes overlap areas between quadriliterals and 
    bounding box
    
    Args:
        ocr(pd.DataFrame): A dataframe with 8 columns defining
            quadriliterals detected by ocr:
            tlx, tly, trx, try, brx, bry, blx, bly
        rectangle(tuple): A tuple of length 4 defined in usual
            way: (left, top, right, bottom)
    Returns:
        Returns a pandas Series object with the per-row results
    '''
    n = len(ocr)
    areas = [0] * n
    left, top, right, bottom = rectangle
    w = right - left
    h = bottom - top
    for i in range(n):
        vertices = [
            (ocr.loc[i,'tlx'],ocr.loc[i,'tly']),
            (ocr.loc[i,'trx'],ocr.loc[i,'try']),
            (ocr.loc[i,'brx'],ocr.loc[i,'bry']),
            (ocr.loc[i,'blx'],ocr.loc[i,'bly']),
             ]
        areas[i] = polygon_overlap_area(left, top, w, h, vertices)
    return pd.Series(areas)


def ocr_2_axes(
    image,
    angles_x=[0,-45,-90],
    engine='easyocr',
    params=AXIS_PARAMS, show=SHOW):
    '''
    Extracts words from image and creates x and y axis
    for chart

    Args:
        image(np.numpy): Input image
        angles_x(float): Rotation angles in degrees used
            for running OCR for interpreting x-axis
        engine(str): OCR engine. One of 'easyocr', 'kraken', 'pytesseract'
        show(bool): If true, show image on screen
    '''
    # Open the image file
    if isinstance(image,str):
        # image = Image.open(image)
        image = cv2.imread(image)
    # Display the image
    # image.show()
    if show:
        print("image at start of ocr_2_axes")
        display_image(image)
    width, height = get_image_dimensions(image)
    axis_rectangles = predict_axis_bb_images_tf(image)
    box_bottom = axis_rectangles['horizontal']
    box_left = axis_rectangles['vertical']
    print("current parameters:")
    print(params)
    box_bottom = expand_bounding_box(box_bottom,
        margin=params['axislabel_bb_margin'],
        img_width=width, img_height=height)

    # bottom_frac = params.get('bottom_frac',0.3)
    # assert bottom_frac > 0.0 and bottom_frac <= 1.0
    # left_frac = params.get('left_frac',0.3)
    # assert left_frac > 0.0 and left_frac <= 1.0
    # top_of_bottom = int(height * (1.0-bottom_frac))
    # assert top_of_bottom >= 0 and top_of_bottom < height
    # box_bottom = (0, top_of_bottom, width, height)

    # crop bottom of image - for detecting x axis
    image_bottom = crop_image(image, box_bottom)
    print('datatype of image:', type(image))
    print("cropped bottom part of image")
    display_image(image_bottom)

    # right_of_left = int(width * left_frac)
    # box_left = (0, 0, right_of_left, height)
    # crop left of image - for detecting y axis
    image_left = crop_image(image, box_left)
    box_left = expand_bounding_box(box_left,
        margin=params['axislabel_bb_margin'],
        img_width=width, img_height=height)
    # x-axis:

    # Use pytesseract to extract the text from the image
    # results = pytesseract.image_to_data(image, config='--psm 11', output_type='data.frame') # pytesseract.Output.DICT)

    config='--psm 11'
    ocr_x = None
    best_angle_x = None
    if not is_iterable(angles_x):
        angles_x = [angles_x]
    for angle in angles_x:
        tmp = image_to_data_with_angle(image_bottom, angle,
            engine=engine, config=config, output_type='data.frame')
        if ocr_x is None:
            ocr_x = tmp
            best_angle_x = angle
        elif len(tmp) > len(ocr_x):
            ocr_x = tmp
            best_angle_x = angle

    # labels_x_rectangle = axis_rectangles['horizontal']
    # labels_y_rectangle = axis_rectangles['vertical']
    # ocr_x['overlap_x'] = calc_ocr_overlaps(ocr_x, labels_x_rectangle)
    
    # print('horizontal axis bounding box:', labels_x_rectangle)
    # print('vertical axis bounding box:', labels_y_rectangle)
    # print("filtering out unlikely x-axis labels:")
    # print(ocr_x[ocr_x['overlap_x'] ==  0.0])
    # ocr_x = ocr_x[ocr_x['overlap_x']> 0.0].reset_index(drop=True)
    ocr_y = image_to_data_with_angle(image_left, angle=0,
            engine=engine, config=config, output_type='data.frame')
    # ocr_y['overlap_y'] = calc_ocr_overlaps(ocr_y, labels_y_rectangle)
    # print("filtering out unlikely y-axis labels:")
    # print(ocr_y[ocr_y['overlap_y'] ==  0.0])
    # ocr_y = ocr_y[ocr_y['overlap_y']> 0.0].reset_index(drop=True)
    # results = results.drop(columns=['page_num', 'conf'], errors='ignore')
    # results = results[~results['text'].isna()].reset_index(drop=True)
    
    # results['right'] = results['left'] + results['width']
    # results['bottom'] = results['top'] + results['height']
    # print(results.to_string())
    print('detect x-labels:')
    resultx = ocr_to_x_labels(ocr_x)
    axis_x = ocr_x.filter(items=resultx['index'], axis=0)
    print("detected axis_x:")
    print(axis_x)

    # y-axis, ucing pytesseract:
    # config='--psm 11'
    # results = pytesseract.image_to_data(image, config=config, output_type='data.frame') # pytesseract.Output.DICT)
    # # results = image_to_data_with_angle(image, angle, config=config, output_type='data.frame') # pytesseract.Output.DICT)
    # results = results.drop(columns=['block_num','par_num','line_num','word_num'], errors='ignore')
    # results = results.drop(columns=['page_num', 'conf'], errors='ignore')
    # results = results[~results['text'].isna()].reset_index(drop=True)
    
    # results['right'] = results['left'] + results['width']
    # results['bottom'] = results['top'] + results['height']
    # print("pytess results:")
    # print(results.to_string())

    resulty = ocr_to_y_labels(ocr_y['tlx'], ocr_y['tly'],
        ocr_y['brx'], ocr_y['bry'], texts=ocr_y['text'])
    #    labels=labels)
    print('detected axis_y:')
    print(resulty)
    axis_y = ocr_y.filter(items=resulty, axis=0)


    # reader = easyocr.Reader(['en'])
    # ocr = reader.readtext(infile, paragraph=False) # , link_threshold=0.99, width_ths=0.01, batch_size=batch_size)
    # ocr = easy_ocr_to_table(ocr)
    # ocr['angle'] = tilts_from_ocr_poly(ocr)
    # print(ocr.to_string())
    # print(axis_x)
    bb_cols_x = ['tlx','blx','trx','brx']
    bb_cols_y = ['try','bry']
    interpret_x = interpret_axis(axis_x, bb_cols=bb_cols_x)
    interpret_y = interpret_axis(axis_y, bb_cols=bb_cols_y)
    d = {'axis_x':axis_x, 'axis_y':axis_y, 'angle_x':best_angle_x}
    for item in interpret_x:
        d[item + '_x'] = interpret_x[item]
    for item in interpret_y:
        d[item + '_y'] = interpret_y[item]
    assert 'chart2px_x' in d
    assert 'chart2px_y' in d

    g = find_grid_lines(np.asarray(image))

    # code for combining OCR-based axes and grid-line based axes goes here
    print('detected initial axes:', d)
    print("detected gridlines", g)
    cv2.waitKey(0)


    return d

def numeric_axis_labels(txt:pd.Series, na_frac_max:float=0.2) -> bool:
    '''
    Returns true if axis labels are mostly numeric, False otherwise.
    Returns False if given length zero.

    Args:
        txt: Series of axis label texts
        na_frac_max: Threshold for fraction of labels that cannot be converted to numbers. Default to 0.2
    '''
    txt = pd.Series(txt)
    if len(txt) == 0:
        return False
    nums = txt.apply(lambda x: pd.to_numeric(x, errors='coerce')) # txt.to_numeric(s, errors='coerce') # .to_numpy()astype(float,errors='ignore')
    frac = nums.isna().sum() / len(nums)
    return frac < na_frac_max



def test_numeric_axis_labels():
    assert numeric_axis_labels(['3','7'])
    assert not numeric_axis_labels(['fox','mouse'])

def interpret_numeric_axis(df:pd.DataFrame, bb_cols:list, txt_col='text') -> dict:
    '''
    Interprets numeric data axis labels. For example, if text labels
    '20', '40', '60' are given together with pixel coordiantes
    a linear regression is used to map pixel coordinates to chart coordinates
    
    Args:
        df: data frame with parsed coordinate texts
        bb_cols(list): column names corresponding to bounding box corners to be tested
    
    '''
    result = {}
    regression = None
    df['_NUMS_'] = df[txt_col].astype(float, errors='ignore')
    df = df[~df['_NUMS_'].isna()].reset_index(drop=True)
    if len(df) < 2:
        print(df)
        raise ValueError("Cannot find sufficient number of labels that can be converted to numbers")
    p_best = 1.0
    px2chart = None
    chart2px = None
    col_best = None
    x = df['_NUMS_'].to_numpy() # numerix axis labels; chart coordinates
    xt = x.reshape(-1, 1)
    for col in bb_cols:
        y = df[col].astype(float).to_numpy() # pixel coordinates
        yt = y.reshape(-1, 1)
        r, p = pearsonr_safe(x,y)
        if p < p_best:
            p_best = p
            col_best = col
            px2chart = LinearRegression()
            px2chart.fit(yt,x)
            chart2px = LinearRegression()
            chart2px.fit(xt,y)
    result['px2chart'] = px2chart
    result['chart2px'] = chart2px
    result['px_col'] = col_best
    return result


def interpret_categoric_axis(df:pd.DataFrame, bb_cols:list, txt_col='text') -> dict:
    '''
    Interprets numeric data axis labela. For example, if text labels
    '20', '40', '60' are given together with pixel coordiantes
    a linear regression is used to map pixel coordinates to chart coordinates
    
    Args:
        df: data frame with parsed coordinate texts
        bb_cols(list): column names corresponding to bounding box corners to be tested
    
    '''
    df['_LEVEL_'] = list(range(len(df)))
    result = interpret_numeric_axis(df, bb_cols=bb_cols, txt_col='_LEVEL_')
    result['levels'] = df[txt_col].tolist()
    return result


def predict_point(model, value):
    '''
    Convenience function to avoid array notation for single value
    Assumes model is similar to LinearRegression
    '''
    # print("called predict point with", value)
    if model is None:
        print("Warning: model is None")
        return np.NaN
    if np.isnan(value):
        return np.NaN
    result = model.predict(np.array([value]).reshape(-1,1) )[0]
    # print("predict point from", value, ":", result)
    return result


def test_interpret_numeric_axis():
    df = pd.DataFrame(data = {
        'text':['20', '40', '60', '80'],
        'x':[1010, 1020, 1030, 1040], # actual pixel column
        'y':[45, 46, 44, 47] # some other pixel column, ignore
    })
    bb_cols = ['x', 'y']
    result = interpret_numeric_axis(df=df,bb_cols=bb_cols)
    # print(result)
    assert 'px2chart' in result
    assert 'chart2px' in result
    assert 'px_col' in result
    assert result['px_col'] == 'x'
    # test if conversion functions have been created to 
    # convert from pixel to chart coordinates and vice versa
    assert abs(predict_point(result['px2chart'], 1020.0)  - 40.0) < 0.2
    assert abs(predict_point(result['chart2px'], 40.0) - 1020.0) < 0.2


def test_interpret_categoric_axis():
    df = pd.DataFrame(data = {
        'text':['Fox', 'Mouse', 'Cat', 'Dog'],
        'x':[1010, 1020, 1030, 1040], # actual pixel column
        'y':[45, 46, 44, 47] # some other pixel column, ignore
    })
    bb_cols = ['x', 'y']
    result = interpret_categoric_axis(df=df,bb_cols=bb_cols)
    # print(result)
    assert 'px2chart' in result
    assert 'chart2px' in result
    assert 'px_col' in result
    assert result['px_col'] == 'x'
    assert abs(predict_point(result['px2chart'], 1010.0)  - 0.0) < 0.2
    assert abs(predict_point(result['px2chart'], 1020.0)  - 1.0) < 0.2
    assert abs(predict_point(result['chart2px'], 1.0) - 1020.0) < 0.2
    assert result['levels'] == ['Fox', 'Mouse', 'Cat', 'Dog']


def interpret_axis(df, bb_cols, txt_col='text') -> dict:
    '''
    Uses raw axis information and creates
    mapping from pixiel coordinates to chart coordinates
    '''
    result = {}
    if numeric_axis_labels(df[txt_col]):
        # print("axis is numeric:")
        # print(df[txt_col])
        result= interpret_numeric_axis(df,
            bb_cols=bb_cols, txt_col=txt_col)
        result['kind'] = 'numeric'
    else:
        result = interpret_categoric_axis(df,
            bb_cols=bb_cols, txt_col=txt_col)
        result['kind'] = 'categoric'
    return result

def test_interpret_axis_numeric():
    df = pd.DataFrame(data = {
        'text':['20', '40', '60', '80'],
        'x':[1010, 1020, 1030, 1040], # actual pixel column
        'y':[45, 46, 44, 47] # some other pixel column, ignore
    })
    bb_cols = ['x', 'y']
    result = interpret_axis(df=df,bb_cols=bb_cols)
    # print(result)
    assert 'px2chart' in result
    assert 'chart2px' in result
    assert 'px_col' in result
    assert result['px_col'] == 'x'
    assert abs(predict_point(result['px2chart'], 1020.0)  - 40.0) < 0.2
    assert abs(predict_point(result['chart2px'], 40.0) - 1020.0) < 0.2


def test_interpret_axis_categoric():
    df = pd.DataFrame(data = {
        'text':['Fox', 'Mouse', 'Cat', 'Dog'],
        'x':[1010, 1020, 1030, 1040], # actual pixel column
        'y':[45, 46, 44, 47] # some other pixel column, ignore
    })
    bb_cols = ['x', 'y']
    result = interpret_axis(df=df,bb_cols=bb_cols)
    # print(result)
    assert 'px2chart' in result
    assert 'chart2px' in result
    assert 'px_col' in result
    assert result['px_col'] == 'x'
    assert abs(predict_point(result['px2chart'], 1010.0)  - 0.0) < 0.2
    assert abs(predict_point(result['px2chart'], 1020.0)  - 1.0) < 0.2
    assert abs(predict_point(result['chart2px'], 1.0) - 1020.0) < 0.2
    assert result['levels'] == ['Fox', 'Mouse', 'Cat', 'Dog']


def test_ocr_2_axes(imgfile= \
    join(RAW_DIR,'train','images','aaeeb3e6866d.jpg'),
    angles_x=[0,-45,-90]):
    # print("input file:", imgfile)
    assert os.path.exists(imgfile)
    result = ocr_2_axes(imgfile, angles_x=angles_x)
    # print(result)
    axis_x = result['axis_x']
    axis_y = result['axis_y']
    assert len(axis_x) == 10
    assert len(axis_y) == 6
    # print('x-axis:')
    # print(axis_x)
    # print(axis_x['text'].tolist())
    # print('y-axis:')
    # print(axis_y)
    # print(axis_y['text'].tolist())
    assert axis_y['text'].tolist()[0] == '20'
    assert axis_y['text'].tolist()[5] == '120'
    assert 'chart2px_x' in result
    assert 'chart2px_y' in result


def test_ocr_2_axes2(imgfile= \
    join(RAW_DIR,'train','images','45df1fe3293b.jpg'),
    engine='kerasocr',
    show=True):
    print("Input file:", imgfile)
    print('OCR engine:', engine)
    assert os.path.exists(imgfile)
    result = ocr_2_axes(imgfile, angles_x=[0], engine=engine,show=show)
    # print(result)
    axis_x = result['axis_x']
    axis_y = result['axis_y']
    assert len(axis_x) == 10
    assert len(axis_y) == 6
    # print('x-axis:')
    # print(axis_x)
    # print(axis_x['text'].tolist())
    # print('y-axis:')
    # print(axis_y)
    # print(axis_y['text'].tolist())
    assert axis_y['text'].tolist()[0] == '20'
    assert axis_y['text'].tolist()[5] == '120'
    assert 'chart2px_x' in result
    assert 'chart2px_y' in result


def find_lines(img, length_min=5.0, show=SHOW):
    '''
    Returns detected lines in image
    '''
    # Load the image
    if isinstance(img,str):
        img = cv2.imread(img)
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # Apply edge detection using Canny algorithm
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # Apply Hough transform to detect lines 
    # lines = cv2.HoughLinesP(thresh, rho=1, theta=np.pi/180, threshold=100) # or edges for thresh
    img2 = copy.deepcopy(img)
    # Find contours
    if show:
        cv2.imshow('thresh', thresh)
        if WAITKEY:
            cv2.waitKey(0)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE) # cv2.CHAIN_APPROX_TC89_L1) # cv2.CHAIN_APPROX_SIMPLE)
    x1v = []
    x2v = []
    y1v = []
    y2v = []
    angles = []
    assert len(contours) > 0
    # Loop over contours and filter for rectangles
    for i in range(len(contours)):
        for j in range(1, len(contours[i])):
            pt1 = tuple(contours[i][j-1][0])
            pt2 = tuple(contours[i][j][0])
            x1 = pt1[0] # contour1[0]
            y1 = pt1[1] # contour1[1]
            x2 = pt2[0] # contour2[0]
            y2 = pt2[1] # contour2[1]
            angle = math.atan2(y2 - y1, x2 - x1)
            # Convert angle to degrees
            angle_degrees = math.degrees(angle)
            x1v.append(x1)
            x2v.append(x2)
            y1v.append(y1)
            y2v.append(y2)
            angles.append(angle_degrees)
            # if abs(angle_degrees) < 5:
            cv2.line(img2, pt1, pt2, (255, 0, 0), 2)
            # perimeter = cv2.arcLength(contour, True)
            # approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            # if len(approx) in [2, 3,4,5]:
            #     x, y, w, h = cv2.boundingRect(contour)
            #     aspect_ratio = float(w) / h
            #     # if aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
            #     cv2.drawContours(img2, [approx], 0, (0, 0, 255), 2)


            # Draw the detected lines on the image

            # for line in lines:
            #     x1, y1, x2, y2 = line[0]
            #     x1v.append(x1)
            #     x2v.append(x2)
            #     y1v.append(y1)
            #     y2v.append(y2)
            #     # Calculate angle of line in radians
            #     angle = math.atan2(y2 - y1, x2 - x1)
            #     # Convert angle to degrees
            #     angle_degrees = math.degrees(angle)
            #     angles.append(angle_degrees)

            # rho, theta = line[0]
            # a = np.cos(theta)
            # b = np.sin(theta)
            # x0 = a * rho
            # y0 = b * rho
            # x1.append(int(x0 + 1000*(-b)))
            # y1.append(int(y0 + 1000*(a)))
            # x2.append(int(x0 - 1000*(-b)))
            # y2.append(int(y0 - 1000*(a)))
            # cv2.line(img2, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # angles.append(np.abs(theta * 180 / np.pi - 90))
    df = pd.DataFrame(data={'x1':x1v,'y1':y1v,'x2':x2v, 'y2':y2v, 'angle':angles})
    df['Length'] = np.sqrt((df['x2']-df['x1'])*(df['x2']-df['x1']) + (df['y2']-df['y1'])*(df['y2']-df['y1']))
    df = df[df['Length'] >= length_min].reset_index(drop=True)
    if show:
        cv2.imshow('detected lines', img2)
        if WAITKEY:
            cv2.waitKey(0)
    return df


def line_length(line) -> float:
    '''
    Computes Euclidian length of a line
    '''
    x0, y0, x1, y1 = line
    dx = x1 - x0
    dy = y1 - y0
    return math.sqrt(dx*dx + dy*dy)


def find_approximately_equidistant_subset_old1(arr):
    '''
    Uses 1D-Fourier transform to find most pronounced
    subset of approximately equi-distant points
    '''
    arr_fft = np.fft.fft(arr)  # Compute the FFT of the array
    power_spectrum = np.abs(arr_fft) ** 2  # Compute the power spectrum of the FFT
    
    # Find the indices of the peaks in the power spectrum
    peak_indices = np.argsort(power_spectrum)[::-1][:len(arr)//2]
    
    # Compute the frequencies corresponding to the peaks
    frequencies = np.fft.fftfreq(len(arr), d=1)
    peak_frequencies = frequencies[peak_indices]
    
    # Find the closest pair of peak frequencies
    min_diff = np.inf
    for i in range(len(peak_frequencies)):
        for j in range(i+1, len(peak_frequencies)):
            diff = abs(peak_frequencies[j] - peak_frequencies[i])
            if diff < min_diff:
                min_diff = diff
                best_indices = (i, j)
    
    # Find the subset of the array corresponding to the best pair of peak frequencies
    subset = []
    for i in range(len(arr)):
        if np.isclose(frequencies[i] % min_diff, 0) and (best_indices[0] <= i <= best_indices[1]):
            subset.append(arr[i])
    
    return subset


def find_approximately_equidistant_subset_old2(arr):
    arr.sort()  # Sort the array in ascending order
    n = len(arr)
    fft = np.fft.fft(arr)  # Compute the 1D Fourier transform of the sorted array
    
    # Find the peak in the Fourier spectrum (excluding the DC component)
    peak_idx = np.argmax(np.abs(fft[1:n//2])) + 1
    
    # Compute the frequency of the peak and the corresponding minimum difference
    freq = peak_idx / n
    min_diff = 1 / (freq * n)
    
    # Find all subsets of the array that have a difference close to the minimum difference
    subsets = []
    for i in range(n):
        for j in range(i+2, n):
            diff = arr[j] - arr[i]
            if abs(diff - (j - i) * min_diff) < 0.1 * min_diff:
                subsets.append(arr[i:j+1])
    
    return subsets


def find_approximately_equidistant_subset_old3(arr, tolerance=0.2):
    n = len(arr)
    arr_fft = np.fft.fft(arr)  # Compute the FFT of the input array
    arr_psd = np.abs(arr_fft) ** 2  # Compute the power spectral density of the FFT
    
    # Find the frequency with the highest power spectral density
    freqs = np.fft.fftfreq(n)
    idx_max = np.argmax(arr_psd[1:n//2]) + 1
    freq_max = freqs[idx_max]
    period_min = int(np.round(1 / (freq_max * n)))  # Compute the minimum period based on the frequency with the highest power spectral density
    print("freqs:", freqs, idx_max, freq_max, period_min)
    # Find all subsets of the array that have a period within the tolerance of the minimum period
    subsets = []
    for i in range(n):
        for j in range(i+2, n):
            period = j - i
            if abs(period - period_min) <= tolerance:
                subsets.append(arr[i:j+1])
    
    return subsets


def find_approximately_equidistant_subset_old4(arr, tolerance=0.3):
    arr = np.array(arr)  # Convert the input array to a numpy array
    n = len(arr)
    freq = np.fft.fftfreq(n)  # Compute the frequencies of the Fourier Transform
    y = np.fft.fft(arr)  # Compute the Fourier Transform of the input array
    amp = np.abs(y)  # Compute the amplitudes of the Fourier Transform
    print('freq:', freq, 'amp:', amp)
    amp[0] = 0  # Set the DC component to zero
    max_amp_idx = np.argmax(amp)  # Find the index of the largest amplitude
    
    # Compute the frequency corresponding to the largest amplitude
    if max_amp_idx <= n//2:
        freq_max_amp = freq[max_amp_idx]
    else:
        freq_max_amp = freq[max_amp_idx-n]
    
    # Find all subsets of the array that have a frequency within the tolerance of the frequency corresponding to the largest amplitude
    subsets = []
    for i in range(n):
        for j in range(i+2, n):
            if abs((j - i) / (arr[j] - arr[i]) - freq_max_amp) <= tolerance:
                subsets.append(arr[i:j+1])
    
    return subsets


def find_approximately_equidistant_subset_old6(arr, tolerance=0.5, window_size=3, poly_order=2):
    '''
    This function first converts the input array to a numpy array, and then applies a Savitzky-Golay filter to smooth the array. The savgol_filter() function from the scipy.signal module is used for this purpose, which fits a polynomial to a sliding window of the input data, and computes the filtered output as the value of the polynomial at the center of the window.
    Next, the function computes the differences between adjacent elements of the smoothed array using numpy.diff(). It then finds the mean difference, and loops through all pairs of elements in the input array, checking if the difference between the endpoints of the subset (computed as the difference between the smoothed values divided by the length of the subset) is within the tolerance of the mean difference. If it is, then the function adds the subset to a list of subsets that are approximately equidistant.
    Note that the choice of window_size and poly_order will affect the smoothing of the input array. A larger window_size will smooth the data over a longer timescale, but may also cause loss of detail in the data. A higher poly_order will fit a higher-degree polynomial to the data, which may better capture the shape of the data, but may also lead to overfitting. The default values of window_size=5 and poly_order=2 are reasonable starting points, but you may need to adjust them depending on the characteristics of your data.
    '''
    arr = np.array(arr)  # Convert the input array to a numpy array
    n = len(arr)
    
    # Smooth the input array using a Savitzky-Golay filter
    smoothed_arr = savgol_filter(arr, window_size, poly_order)
    
    # Compute the differences between adjacent elements of the smoothed array
    diff = np.diff(smoothed_arr)
    
    # Find all subsets of the array that have a difference within the tolerance of the mean difference
    mean_diff = np.mean(diff)
    subsets = []
    for i in range(n):
        for j in range(i+2, n):
            if abs((smoothed_arr[j] - smoothed_arr[i]) / (j - i) - mean_diff) <= tolerance:
                subsets.append(arr[i:j+1])
    
    return subsets


def find_approximately_equidistant_subset_7(arr, tolerance=0.5, window_size=3):
    arr = np.array(arr)  # Convert the input array to a numpy array
    n = len(arr)
    
    # Compute the pairwise distances between adjacent elements in the input array
    dist = np.diff(arr)
    print('distance matrix:', dist)

    # Compute histogram with 20 bins
    hist, edges = np.histogram(dist, bins=20)

    # Find bin with highest count
    max_count_idx = np.argmax(hist)

    # Compute mode as midpoint of bin with highest count
    mode = (edges[max_count_idx] + edges[max_count_idx+1]) / 2
    print("most common distance:", mode)

    # Smooth the distances using a moving average with a window size of `window_size`
    smooth_dist = np.convolve(dist, np.ones(window_size)/window_size, mode='same')
    # print("Smooth distances:", smooth_dist)
    # Compute the frequencies of the smoothed distances using Fourier Transform
    freq = np.fft.fftfreq(len(smooth_dist))
    y = np.fft.fft(smooth_dist)
    amp = np.abs(y)
    amp[0] = 0
    max_amp_idx = np.argmax(amp)
    
    # Compute the frequency corresponding to the largest amplitude
    if max_amp_idx <= n//2:
        freq_max_amp = freq[max_amp_idx]
    else:
        freq_max_amp = freq[max_amp_idx-n]
    
    # Find all subsets of the array that have a frequency within the tolerance of the frequency corresponding to the largest amplitude
    subsets = []
    for i in range(n):
        for j in range(i+2, n):
            if abs((j - i) / smooth_dist[i] - freq_max_amp) <= tolerance:
                subsets.append(arr[i:j+1])
    
    return subsets

def find_approximately_equidistant_subset(arr:list, tolerance=2.0, dist_min=2.0):
    """
    Finds a subset of elements in an array that are approximately equidistant.

    Parameters
    ----------
    arr : array_like
        Input array. Must be 1-dimensional.
    tolerance : float
        Tolerance for the difference between adjacent elements and the most common difference.

    Returns
    -------
    list
        A list of 1-dimensional numpy arrays, each of which contains 2 elements.
        Each pair of adjacent elements in the input array that are approximately equidistant,
        within the specified tolerance, is returned as a separate numpy array in the list.
        If no such pairs are found, the function returns an empty list.
    """
    arr = copy.deepcopy(arr)
    arr.sort()
    arr = np.array(arr)  # Convert the input array to a numpy array
    diffs = np.diff(arr)  # Compute the differences between adjacent elements
    # print("distances:", diffs)
    hist, bin_edges = np.histogram(diffs, bins=30,
        range=(max(diffs.min(),dist_min),diffs.max()) )  # Compute the histogram of the differences
    bin_width = bin_edges[1] - bin_edges[0]
    most_common_diff = bin_edges[np.argmax(hist)] + bin_width/2.0 # Find the most common difference
    # print("estimated most commond distance:", most_common_diff)
    # Find all subsets of the array that have a difference within the tolerance of the most common difference
    subsets = []
    n = len(arr)
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(arr[j] - arr[j-1] - most_common_diff) <= tolerance:
            j += 1
        if j - i > 1:
            subsets.append(arr[i:j])
        i = j
    
    result = subsets, most_common_diff
    if DEBUG:
        print("result of find_approximately_equidistant_subset:", result)
    return result


def test_find_approximately_equidistant_subset():
    arr = [5, 10, 40, 70, 100, 190]
    expected_subset = [10, 40, 70, 100] # distance of 30
    subset,_ = find_approximately_equidistant_subset(arr)
    subset = subset[0]
    if DEBUG:
        print("equi-subset of", arr, ':', expected_subset, ':', subset)
    np.testing.assert_array_equal(subset, expected_subset)
    # arr = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17]
    # expected_subset = [1, 5, 8, 11, 14, 17]
    # subset = find_approximately_equidistant_subset(arr)[0]
    # print("equi-subset of", arr, ':', expected_subset, ':', subset)
    # np.testing.assert_array_equal(subset, expected_subset)
    # assert False

    # arr = [1, 2, 3, 4, 5]
    # expected_subset = [1, 3, 5]
    # subset = find_approximately_equidistant_subset(arr)
    # print("equi-subset of", arr, ':', expected_subset, ':', subset)
    # np.testing.assert_array_equal(subset, expected_subset)

    # arr = [1, 2, 3, 4, 6]
    # expected_subset = [1, 4]
    # subset = find_approximately_equidistant_subset(arr)
    # print("equi-subset of", arr, ':', expected_subset, ':', subset)
    # np.testing.assert_array_equal(subset, expected_subset)

    # arr = [1, 2, 3, 4, 5, 7, 8, 9]
    # expected_subset = [1, 3, 5, 7, 9]
    # subset = find_approximately_equidistant_subset(arr)
    # print("equi-subset of", arr, ':', expected_subset, ':', subset)
    # np.testing.assert_array_equal(subset, expected_subset)


def find_grid_lines(img, length_min=30.0,
        img_title='Find grid lines',
        tolerance_x = 3.0,
        tolerance_y = 3.0,
        num_lines_min=4,
        show=SHOW):
    '''
    Returns detected grid lines in image
    TODO: skip lines that are near detected text
    Args:
        img: Numpy image
        show(bool): If true visualize found grid-lines
        Returns:
            pd.DataFrame with columns: x0,y0, x1,y1, 'kind'(with values 'h' or 'v')
    '''
    # Load the image
    if isinstance(img,str):
        if DEBUG:
            print("reading image", img)
        img = cv2.imread(img)
    ny, nx = img.shape[:2]
    dist_y_min = ny / 30
    dist_y_max = ny / 5
    dist_x_min = nx / 30
    dist_x_max = nx / 5
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # Apply edge detection using Canny algorithm
    edges = cv2.Canny(gray, 10, 50, apertureSize=3) # 50, 150, 3
    # print('edges:', edges) # Apply Hough transform to detect lines 
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=110, minLineLength=20, maxLineGap=5)
    #                              rho=1, theta=np.pi/180, threshold=100) # or edges for thresh
    img2 = copy.deepcopy(img)
    x0v = []
    y0v = []
    x1v = []
    y1v = []
    kindv = []
    # Draw lines on the image
    # print('houg lines:', lines)
    for line in lines:
        ll = line[0]
        length = line_length(ll)
        x0, y0, x1, y1 = ll
        angle = math.atan2(y1 - y0, x1 - x0)
        # Convert angle to degrees
        angle_degrees = math.degrees(angle)
        kind = 'other'
        # print('found line:', ll, angle)
        if length < length_min:
            continue # skip too short lines
        if abs(abs(angle_degrees)-90) < 5:
            kind = 'v'
        elif abs(angle_degrees) < 5:
            kind = 'h'
        # code for filtering
        # if kind in ['v','h','other']:
        x0v.append(x0)
        y0v.append(y0)
        x1v.append(x1)
        y1v.append(y1)
        kindv.append(kind)
        cv2.line(img2, (x0, y0), (x1, y1), (0, 0, 255), 2)
    if show:
        cv2.imshow(img_title, img2)
        if WAITKEY:
            cv2.waitKey(0)
    df = pd.DataFrame(data={
        'x0':x0v,
        'y0':y0v,
        'x1':x1v,
        'y1':y1v,
        'kind':kindv})
    # midpoints of lines:
    df['ym'] = (df['y0'] + df['y1']) * 0.5
    df['xm'] = (df['x0'] + df['x1']) * 0.5
    vlines = df[df['kind'] == 'v'].reset_index(drop=True)
    xmv = vlines['xm'].tolist()
    hlines = df[df['kind'] == 'h'].reset_index(drop=True)
    ymv = hlines['ym'].tolist()
    xsubsets, common_dx = find_approximately_equidistant_subset(xmv,dist_min=dist_x_min)
    # print('xsubsets:', xsubsets)
    xsubset = []
    # the following code is a bit tedious would be better to be part of
    # function find_approximately_equidistant_subset
    # what it is doing is to combine several subsets that are potentially
    # overlapping into one list
    if len(xsubsets) > 0:
        xsubset = xsubsets[0].tolist() # [item for sublist in ysubsets for item in sublist]
        for i in range(1,len(xsubsets)):
            nextset = xsubsets[i].tolist()
            if min(nextset) - max(xsubset) < abs(common_dx - tolerance_x):
                nextset = nextset[1:]
            xsubset = xsubset + nextset
    else:
        xsubset = None
    ysubsets, common_dy = find_approximately_equidistant_subset(ymv,dist_min=dist_y_min)
    # print('ysubsets:', ysubsets)
    ysubset = []
    # the following code is a bit tedious would be better to be part of
    # function find_approximately_equidistant_subset
    # what it is doing is to combine several subsets that are potentially
    # overlapping into one list
    if len(ysubsets) > 0:
        ysubset = ysubsets[0].tolist() # [item for sublist in ysubsets for item in sublist]
        for i in range(1,len(ysubsets)):
            nextset = ysubsets[i].tolist()
            if min(nextset) - max(ysubset) < abs(common_dy - tolerance_y):
                nextset = nextset[1:]
            ysubset = ysubset + nextset
    else:
        ysubset = None
    if show:
        if DEBUG:
            print('xsubset', xsubset)
        for x in xsubset:
            cv2.line(img,(int(x),50),(int(x),200), (0,200,255),1)
        if DEBUG:
            print('ysubset', ysubset)
        for y in ysubset:
            cv2.line(img,(10,int(y)),(200,int(y)), (200,0,255),1)
        cv2.imshow('lines',img)
        if WAITKEY:
            cv2.waitKey(0)
    if len(xsubset) < num_lines_min:
        xsubset = None
    if len(ysubset) < num_lines_min:
        ysubset = None  
    return {'x':xsubset, 'y':ysubset}


def test_find_grid_lines(imgfile= \
    join(RAW_DIR,'train','images','aaeeb3e6866d.jpg'),
    angles_x=[0,-45,-90]):
    # print("input file:", imgfile)
    assert os.path.exists(imgfile)
    df = find_grid_lines(imgfile)
    assert isinstance(df, dict)
    assert isinstance(df['y'], list)
    assert df['y'] == [50.0, 75.0, 101.0, 128.0, 155.0]
    assert df['x'] is None


def test_find_grid_lines2(imgfile= \
    join(RAW_DIR,'train','images','45df1fe3293b.jpg') ):
    '''
    Example of both x and y lines
    '''
    # print("input file:", imgfile)
    assert os.path.exists(imgfile)
    # print('parsing file', imgfile)
    d = find_grid_lines(imgfile)
    # print(d)
    assert isinstance(d, dict)
    assert isinstance(d['x'], list)
    assert isinstance(d['y'], list)
    assert d['x'] == [80.0, 110.0, 144.0, 176.0, 209.0, 242.0, 275.0, 308.0, 340.0, 373.0, 406.0, 439.0, 472.0]
    assert d['y'] == [67.0, 85.0, 106.0, 126.0, 146.0, 166.0, 187.0, 207.0, 228.0]


def create_difference_matrix(x, y):
    """
    Creates a matrix of all the possible differences between two 1D arrays x and y.

    Parameters:
    x (numpy.ndarray): A 1D numpy array.
    y (numpy.ndarray): A 1D numpy array.

    Returns:
    numpy.ndarray: A 2D numpy array with dimensions (len(x), len(y)), where the (i, j) element
    is the difference between the i-th element of x and the j-th element of y.

    Example:
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([4, 5, 6])
    >>> D = create_difference_matrix(x, y)
    >>> print(D)
    array([[-3, -4, -5],
           [-2, -3, -4],
           [-1, -2, -3]])
    """
    print("Started create_difference_matrix with")
    print(x)
    print(y)
    print(x.shape)
    print(y.shape)
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    X = np.tile(x, (len(y), 1)).T
    Y = np.tile(y, (len(x), 1))
    return X - Y


def test_create_difference_matrix():
    # Test case 1: x and y have the same length
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    D = create_difference_matrix(x, y)
    print("difference matrix:", D)
    assert np.all(D == np.array([[-3, -4, -5], [-2, -3, -4], [-1, -2, -3]])), f"Test case 1 failed: D = {D}"

    # Test case 2: x has more values than y
    x = np.array([1, 2, 3, 4])
    y = np.array([5, 6])
    D = create_difference_matrix(x, y)
    assert np.all(D == np.array([[-4, -5], [-3, -4], [-2, -3], [-1, -2]])), f"Test case 2 failed: D = {D}"

    # Test case 3: y has more values than x
    x = np.array([1, 2])
    y = np.array([3, 4, 5])
    D = create_difference_matrix(x, y)
    assert np.all(D == np.array([[-2, -3, -4], [-1, -2, -3]])), f"Test case 3 failed: D = {D}"

    print("All tests passed!")


def melt_matrix(X):
    """
    Converts a numeric matrix into a 'melted' pandas DataFrame.

    Parameters:
    X (numpy.ndarray): A 2D numpy array.

    Returns:
    pandas.DataFrame: A pandas DataFrame with columns 'row', 'col', and 'value',
    where 'row' and 'col' correspond to the row and column indices of the input matrix,
    respectively, and 'value' corresponds to the value of the input matrix at that index.

    Example:
    >>> X = np.array([[1, 2], [3, 4]])
    >>> df = melt_matrix(X)
    >>> print(df)
       row  col  value
    0    0    0      1
    1    0    1      2
    2    1    0      3
    3    1    1      4
    """
    print('started melt_matrix with:',X)
    assert len(X.shape) == 2
    print(np.ndindex(X.shape))
    # rows, cols = np.ndindex(X.shape)
    rows, cols = zip(*np.ndindex(X.shape))
    values = X.flatten()
    df = pd.DataFrame({'row': rows, 'col': cols, 'value': values})
    return df


def test_melt_matrix():
    # Test case 1: Identity matrix
    X = np.eye(3)
    df = melt_matrix(X)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (9, 3)
    assert all(df['value'] == [1, 0, 0, 0, 1, 0, 0, 0, 1])
    assert all(df['row'] == [0, 1, 2] * 3)
    assert all(df['col'] == [0, 1, 2] * 3)

    # Test case 2: Random matrix
    X = np.random.rand(2, 4)
    df = melt_matrix(X)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (8, 3)
    assert all(df['value'] == X.flatten())
    assert all(df['row'] == [0, 0, 0, 0, 1, 1, 1, 1])
    assert all(df['col'] == [0, 1, 2, 3] * 2)


def numeric_alignment(x, y) -> np.ndarray:
    '''
    Returns a data frame that maps elements of vector x
    to vector y
    Not all values have to be mapped. However, no
    element of x is mapped from more than once and no element of y
    is mapped to more than once.

    Args:
        x(np.ndarray): A numeric vector of values
        y(np.ndarray): A numeric vector of values

    Returns:
        Returns a data frame that maps elements of vector x
    to vector y.
    '''
    if not isinstance(x,np.ndarray):
        x = np.ndarray(x)
    assert len(x.shape) == 1
    if not isinstance(y,np.ndarray):
        y = np.ndarray(y)
    assert len(y.shape) == 1
    # Create a KDTree for Y
    D = create_difference_matrix(x,y)
    # sort such that lowest differences first:
    df = melt_matrix(D)
    df['value'] = df['value'].abs()
    df = df.sort_values(by=['value']).reset_index(drop=True)
    df['dup_row'] = df['row'].duplicated(keep='first')
    df['dup_col'] = df['col'].duplicated(keep='first')
    df['dup_either'] = (df['dup_row'] | df['dup_col'])
    print("melted matrix:")
    print(df)
    # keep only elements such that none is used twice
    df = df[~df['dup_either']].drop(columns=['dup_row','dup_col','dup_either'])
    df = df.sort_values(by='row').reset_index(drop=True)
    df.columns = ['xid','yid','diff']
    return df


def test_numeric_alignment():
    x = np.array([14.1, 1.5, 2.5, 3.5, 4.5]) # np.array([1, 2, 3, 4, 5])
    y = np.array([1.7, 2.7, 3.7,-9.0, 4.7]) # np.array([0, 2, 4, 6, 8, 10])
    expected_output = pd.DataFrame({'row': [1, 2, 3,4], 'col': [0, 1, 2, 4], 'value': [0.2, 0.2, 0.2, 0.2]})
    result = numeric_alignment(x, y)
    
    print('result:')
    print(result)
    assert np.array_equal(result['xid'],[1, 2, 3,4]) # expected_output)
    assert np.array_equal(result['yid'],[0, 1, 2, 4])


def improve_scale(x, y) -> np.ndarray:
    '''
    Given a vector x, shift x such that it aligns best with y
    Can handle missing values, finds alignment of x and y itself
    '''
    align = numeric_alignment(x, y)
    align['x'] = x[align['xid']]
    align['y'] = y[align['yid']]
    model = LinearRegression()
    # fit the model to the data
    model.fit(align[['x']].to_numpy(), align['y'].to_numpy())
    align['x2'] = model.predict(align[['x']].to_numpy())
    print(align)
    x2 = model.predict(x.reshape(-1, 1) )
    print(x2)
    return {'x2':x2, 'model':model, 'alignment':align}


def test_improve_scale():
    x = np.array([14.1, 1.5, 2.5, 3.5, 4.5]) # np.array([1, 2, 3, 4, 5])
    y = np.array([1.7, 2.7, 3.7,-9.0, 4.7]) # np.array([0, 2, 4, 6, 8, 10])
    result = improve_scale(x, y)
    assert np.allclose(result['x2'],[14.3, 1.7, 2.7, 3.7, 4.7])
    assert np.array_equal(result['alignment']['xid'],[1, 2, 3,4])
    assert np.array_equal(result['alignment']['yid'],[0, 1, 2, 4])


def read_annotation(file = join(RAW_DIR,'train','annotations','aaeeb3e6866d.json')):
    '''
    Reads chart annotations as provided by Kaggle competition
    '''
    # Load JSON file
    with open(file, 'r') as file:
        json_data = file.read()
    # Convert JSON data to dictionary
    my_dict = json.loads(json_data)
    data_series = my_dict['data-series']
    x = []
    y = []
    for i in range(len(data_series)):
        x.append(data_series[i]['x'])
        y.append(data_series[i]['y'])
    df = pd.DataFrame(data={'x':x, 'y':y})
    my_dict['data-series'] = df
    return my_dict


def run_all_tests():
    # test_polygon_overlap_area()
    # test_improve_scale()
    # assert False
    test_ocr_2_axes2()
    assert False
    test_create_difference_matrix()
    test_numeric_alignment()
    assert False
    test_create_difference_matrix()
    assert False

    test_find_grid_lines2()
    test_find_grid_lines()
    test_find_approximately_equidistant_subset()
    test_numeric_axis_labels()
    test_interpret_axis_numeric()
    test_interpret_axis_categoric()
    test_interpret_categoric_axis()
    test_interpret_numeric_axis()
    test_ocr_to_y_labels()
    test_ocr_to_x_labels()
    # test_east_detect()
    test_ocr_2_axes()


if __name__ == '__main__':
    run_all_tests()

