
import math
import json
import os
import cv2
import copy
from os.path import join, normpath
from chartparams import CHART_PARAMS as PARAMS

DEBUG = False
SHOW = False

IMG_RAW_DIR = normpath('../data/raw/train/images')
IMG_INTERIM_DIR = normpath('../data/interim/train/images')
ANNO_RAW_DIR = normpath('../data/raw/train/annotations')
ANNO_INTERIM_DIR = normpath('../data/interim/train/annotations')
assert os.path.exists(ANNO_RAW_DIR)
# assert os.path.exists(ANNO_INTERIM_DIR)


def boundingbox_distance(box1, box2) -> float:
    '''
    Returns distance between two bounding boxes. While there
    are many different ways of computing that, we use here the
    approach of the minimum distance between any two sides.
    The distance between overlapping bounding boxes is zero.

    Args:
        bb0(tuple): A 4-tuple corresponding to first bounding box
        bb1(tuple): A 4-tuple corresponding to second bounding box
    '''
    l1, t1, r1, b1 = box1
    l2, t2, r2, b2 = box2

    left = l2 - r1
    right = l1 - r2
    top = t2 - b1
    bottom = t1 - b2

    # If the boxes intersect, distance is zero
    if left < 0 and right < 0 and top < 0 and bottom < 0:
        return 0.0

    # Calculate the minimum distance between box edges
    distances = [left, right, top, bottom]
    distances = [d for d in distances if d > 0]
    if len(distances) == 0:
        return 0.0
    else:
        return math.sqrt(min([d**2 for d in distances]))
    

def test_boundingbox_distance():
    box1 = (10, 10, 30, 40)
    box2 = (30, 30, 50, 60)
    print(boundingbox_distance(box1, box2))
    assert math.isclose(boundingbox_distance(box1, box2), 0)

    box1 = (10, 10, 30, 40)
    box2 = (50, 50, 70, 90)
    print(boundingbox_distance(box1, box2))
    assert math.isclose(boundingbox_distance(box1, box2), 10)

    box1 = (10, 10, 30, 40)
    box2 = (0, 0, 5, 5)
    print(boundingbox_distance(box1, box2))
    assert math.isclose(boundingbox_distance(box1, box2), 5)


def closest_box(box_list, id):
    closest_distance = float('inf')
    closest_id = None
    for i in range(len(box_list)):
        if i == id:
            continue # skip self
        box = box_list[i]
        d = boundingbox_distance(box_list[id], box)
        if d < closest_distance:
            closest_distance = d
            closest_box = box
            closest_id = i

    return closest_id


def overlap(interval1, interval2):
    """
    Compute the overlap between two 1D intervals.

    Args:
        interval1 (tuple): A tuple representing the first interval as (left1, right1). Both bounds are inclusive
        interval2 (tuple): A tuple representing the second interval as (left2, right2). Both bounds are inclusive

    Returns:
        float: The length of the overlap between the two intervals. If the intervals do not overlap, the overlap length is zero.
    """
    left1, right1 = interval1
    left2, right2 = interval2

    overlap = max(0, min(right1, right2) - max(left1, left2))

    return overlap

def annotation_to_x_y_bounding_boxes(anno:dict) -> dict:
    '''
    Creates additional annotations of Kaggle images
    
    Args:
        anno(dict): Input annotation of a chart image that
        follows the annotation provided by Kaggle
    
    Returns:
        Returns the same annotation augmented with additional values
        such as the bounding box surreounding all x-labels and y-labels
    
    '''
    left, top, right, bottom = 10000, 10000, -1, -1
    txtary = anno['text'] # array of all text
    boxes = []
    topmost = None
    rightmost = None
    assert len(txtary) > 0
    found_tick = False
    for i in range(len(txtary)):
        txt = txtary[i]
        role = txt['role']
        if role == "tick_label":
            found_tick = True
            bb = polygon_boundingbox(txt['polygon'])
            boxes.append(bb)
            left = min(left, bb[0])
            if bb[1] < top:
                topmost = len(boxes)-1
            top = min(top, bb[1])
            if bb[2] > right:
                rightmost = len(boxes)-1
            right = max(right, bb[2])
            bottom = max(bottom, bb[3])
    # no we have overall bounding box of all x and y tick labels
    # but how can we sepate that into x alone and y alone?
    assert found_tick
    assert rightmost is not None
    assert topmost is not None

    xids = [rightmost]
    yids = [topmost]
    lx, tx, rx, bx = boxes[rightmost] # right most bounding box is start for bounding box over all x-tick-labels
    ly, ty, ry, by = boxes[topmost] # top-most bounding box is start for bounding box over all y-tick-labels
    if DEBUG:
        print("all boxes:")
        print(boxes)
        print("rightmost box", rightmost, ":", boxes[rightmost])
        print("topmost box", topmost, ":", boxes[topmost])
    for i in range(len(boxes)):
        if i in [rightmost, topmost]:
            continue # already included
        box = boxes[i]
        l,t,r,b = box
        # if overlapping with current x-axis bounding box but not y-axis one:
        if overlap((t, b), (tx, bx)) > 0 and overlap((l, r), (ly, ry)) == 0:
            lx = min(lx, l) # shift left bound
            bx = max(bx, b) # shift lower bound
            rx = max(rx, r)
            tx = min(tx, t)
            xids.append(i)
        elif overlap((t, b), (tx, bx)) == 0 and overlap((l, r), (ly, ry)) > 0:
            # if overlapping with current y-axis bounding box but not x-axis one:
            ly = min(ly, l) # shift left bound
            by = max(by, b) # shift lower bound
            ry = max(ry, r)
            ty = min(ty, t)
            yids.append(i)
    # special case of origin that may be used for box x and y:
    origin_count = 0
    for i in range(len(boxes)):
        box = boxes[i]
        if  overlap((t, b), (tx, bx)) > 0 and overlap((l, r), (ly, ry)) > 0:
            lx = min(lx, l) # shift left bound of x bb
            by = max(by, b) # shift lower bound of y bb
            # do not shift any other sides of x or y bounding boxes to avoid drift
            origin_count += 1
            # add to both x and y ids
            if i not in xids:
                xids.append(i)
            if i not in yids:
                yids.append(i)
            print("found origin label that is used for both x and y axis labels")
    xbox = (lx, tx, rx, bx)
    ybox = (ly, ty, ry, by)
    return {
        'xbox':xbox, 'ybox':ybox,
        'boxes': boxes,
        'x_ids':xids, 'y_ids':yids,
        'rightmost':rightmost, 'topmost':topmost
    }


def test_annotation_to_x_y_bounding_boxes(show=SHOW):
    annofile = '../fixtures/grid_barchart/45df1fe3293b.json'
    imgfile = '../fixtures/grid_barchart/45df1fe3293b.jpg'
    assert os.path.exists(annofile)
    with open(annofile, 'r') as f:
        anno = json.load(f)
    result = annotation_to_x_y_bounding_boxes(anno)
    print(result)
    xbox = result['xbox']
    ybox = result['ybox']
    boxes = result['boxes']
    topm = boxes[result['topmost']] # starting bb on top for y
    rightm = boxes[result['rightmost']] # starting bb on right for x
    if show:
        img = cv2.imread(imgfile)
        cv2.rectangle(img, (xbox[0], xbox[1]), (xbox[2], xbox[3]), (255, 0, 0), 2)
        cv2.rectangle(img, (ybox[0], ybox[1]), (ybox[2], ybox[3]), (0, 255, 0), 2)
        # show 'starter' box:
        # cv2.rectangle(img, (rightm[0], rightm[1]), (rightm[2], rightm[3]), (0, 0, 255), 2)
        cv2.imshow('recognized labels', img)
        cv2.waitKey(0)
    assert xbox[0] > ybox[2] # left side of x-labels must be to right of right side of y-labels
    assert xbox[1] > ybox[3] # top x-labels must be below bottom of y-labels


def test_annotation_to_x_y_bounding_boxes2(show=SHOW):
    '''
    Unit test for detecting bounding box of x and y labels
    Fails, but this is due to a malformed JSON file
    where all tick labels are incorrectly marked as 
    axis-title
    Remove this test from standard tests,
    do not require this test to pass
    '''
    annofile = '../fixtures/tricky1/733b9b19e09a.json'
    imgfile = '../fixtures/tricky1/733b9b19e09a.jpg'
    assert os.path.exists(annofile)
    with open(annofile, 'r') as f:
        anno = json.load(f)
    result = annotation_to_x_y_bounding_boxes(anno)
    print(result)
    xbox = result['xbox']
    ybox = result['ybox']
    boxes = result['boxes']
    topm = boxes[result['topmost']] # starting bb on top for y
    rightm = boxes[result['rightmost']] # starting bb on right for x
    if show:
        img = cv2.imread(imgfile)
        cv2.rectangle(img, (xbox[0], xbox[1]), (xbox[2], xbox[3]), (255, 0, 0), 2)
        cv2.rectangle(img, (ybox[0], ybox[1]), (ybox[2], ybox[3]), (0, 255, 0), 2)
        # show 'starter' box:
        # cv2.rectangle(img, (rightm[0], rightm[1]), (rightm[2], rightm[3]), (0, 0, 255), 2)
        cv2.imshow('recognized labels', img)
        cv2.waitKey(0)
    assert xbox[0] > ybox[2] # left side of x-labels must be to right of right side of y-labels
    assert xbox[1] > ybox[3] # top x-labels must be below bottom of y-labels


def augment_annotation(anno:dict) -> dict:
    '''
    Augments Kaggle annotation with additional computed values
    '''
    anno = copy.deepcopy(anno) # stick to call by value
    xyboxes = annotation_to_x_y_bounding_boxes(anno)
    anno['xbox'] = xyboxes['xbox'] # bounding boxes overall x-labels
    anno['ybox'] = xyboxes['ybox'] # bounding boxes overall x-labels
    return anno


def test_augment_annotation(show=SHOW):
    annofile = '../fixtures/grid_barchart/45df1fe3293b.json'
    imgfile = '../fixtures/grid_barchart/45df1fe3293b.jpg'
    assert os.path.exists(annofile)
    with open(annofile, 'r') as f:
        anno = json.load(f)
    result = augment_annotation(anno)
    print(result)
    xbox = result['xbox']
    ybox = result['ybox']
    if show:
        img = cv2.imread(imgfile)
        cv2.rectangle(img, (xbox[0], xbox[1]), (xbox[2], xbox[3]), (255, 0, 0), 2)
        cv2.rectangle(img, (ybox[0], ybox[1]), (ybox[2], ybox[3]), (0, 255, 0), 2)
        cv2.imshow('recognized labels', img)
        cv2.waitKey(0)
    assert xbox[0] > ybox[2] # left side of x-labels must be to right of right side of y-labels
    assert xbox[1] > ybox[3] # top x-labels must be below bottom of y-labels


def polygon_boundingbox(d:dict) -> tuple:
    '''
    Assumes dictionary contains keys 'x0', 'y0', 'x1', 'y1', ...
    returns tuple of left, top, right, bottom rectangular
    bounding box capturing whole quadriliteral

    Args:

        d(dict): Input dictionary corresponding to Kaggle
            annotation under -> text -> 0|1|2|... -> polygon
    
    Returns 4-tuple with bounding box values (float) that are 
        left, top, right, bottom of rectangle
    '''
    left = min(d['x0'], d['x1'], d['x2'], d['x3'])
    top = min(d['y0'], d['y1'], d['y2'], d['y3'])
    right = max(d['x0'], d['x1'], d['x2'], d['x3'])
    bottom = max(d['y0'], d['y1'], d['y2'], d['y3'])
    return (left, top, right, bottom)

def test_polygon_boundingbox():
    bb = polygon_boundingbox({
        'x0':1,'x1':2,'x2':3,'x3':4,
        'y0':10,'y1':11,'y2':12,'y3':13,
    })
    assert bb == (1,10,4, 13)


def make_annotation(anno:dict) -> dict:
    '''
    Creates additional annotations of Kaggle images
    
    Args:
        anno(dict): Input annotation of a chart image that
        follows the annotation provided by Kaggle
    
    Returns:
        Returns the same annotation augmented with additional values
        such as the bounding box surreounding all x-labels and y-labels
    
    '''
    # CODE
    return anno


def make_annotations(
    rawdir=ANNO_RAW_DIR,
    interimdir = ANNO_INTERIM_DIR,
    overwrite=True):
    '''
    Loops over all annotations and augments them
    with additional computed values and saves
    to interim
    '''
    jsons = os.listdir(rawdir)
    if not os.path.exists(interimdir):
        print("Creating directory", interimdir)
        os.makedirs(interimdir)
    for i in range(len(jsons)):
        base = jsons[i]
        print(f"working on file {i} {round(100*i/len(jsons),1)}%: {base}")
        infile = join(rawdir, base)
        assert os.path.exists(infile)
        with open(infile, 'r') as f:
            d = json.load(f)
        d2 = augment_annotation(d)
        outfile = join(interimdir, base)
        if not overwrite and os.path.exists(outfile):
            print(f"File {outfile} already exists, no overwrite")
            continue
        with open(outfile, 'w') as f:
            json.dump(d2, f)


def make_annotations(
    rawdir=ANNO_RAW_DIR,
    interimdir = ANNO_INTERIM_DIR,
    overwrite=True,
    thumbx=int(PARAMS['thumbx']),
    thumby=int(PARAMS['thumby'])):
    '''
    Loops over all annotations and augments them
    with additional computed values and saves
    to interim
    '''
    assert os.path.exists(rawdir)
    jsons = os.listdir(rawdir)
    if not os.path.exists(interimdir):
        print("Creating directory", interimdir)
        os.makedirs(interimdir)
    if not os.path.exists(IMG_INTERIM_DIR):
        print("Creating directory", IMG_INTERIM_DIR)
        os.makedirs(IMG_INTERIM_DIR)
    for i in range(len(jsons)):
        base = jsons[i][:-len('.json')]

        print(f"working on file {i} {round(100*i/len(jsons),1)}%: {base}")
        infile = join(rawdir, base + '.json')
        assert os.path.exists(infile)
        with open(infile, 'r') as f:
            d = json.load(f)
        d2 = augment_annotation(d)
        lx, tx, rx, bx = d2['xbox']
        ly, ty, ry, by = d2['ybox']
        outfile = join(interimdir, base + '.json')
        imgoutfile = join(IMG_INTERIM_DIR, base + '.jpg')
        if not overwrite and os.path.exists(outfile):
            print(f"File {outfile} already exists, no overwrite")
            continue
        imgfile = join(IMG_RAW_DIR, base + '.jpg')
        os.path.exists(imgfile)
        img = cv2.imread(imgfile)
        img_shape = img.shape
        d2['img_shape'] = img_shape
        ny, nx = img_shape[:2]
        if rx >= nx:
            rx = nx - 1
        if bx >= ny:
            bx = ny - 1
        if ry >= nx:
            ry = nx - 1
        if by >= ny:
            by = ny - 1
        assert lx >= 0
        assert tx >= 0
        assert ly >= 0
        assert ty >= 0
        assert rx <= nx
        assert bx <= ny
        assert ry <= nx # right side of y-axis labels should be less than x-resolution
        assert by <= ny
        d2['xbox'] = lx, tx, rx, bx # update clipped version
        assert tx < bx
        assert lx < rx
        d2['ybox'] = ly, ty, ry, by
        assert ly < ry
        assert ty < by
        # downscale
        # Resize the color image to 100x100 resolution
        resized_image = cv2.resize(img, (thumbx, thumby))
        resized_shape = img.shape
        d2['img_resized_shape'] = resized_shape
        # Convert the resized color image to grayscale
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        with open(outfile, 'w') as f:
            json.dump(d2, f)
        print("writing to files", outfile, imgoutfile)
        cv2.imwrite(imgoutfile, gray_image)

def run_tests():
    # test_annotation_to_x_y_bounding_boxes2(show=True)
    test_polygon_boundingbox()
    test_boundingbox_distance()
    test_augment_annotation()
    test_annotation_to_x_y_bounding_boxes()


if __name__ == '__main__':
    run_tests()
    make_annotations()

