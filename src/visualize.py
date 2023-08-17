'''
Visualize Chart Annotations
'''

import numpy as np
import cv2
import copy
import json
import os
from os.path import join, normpath
import random

random.seed=42

PARAMS = {
    'roles': {
        'chart_title':(0,255,0),
        'axis_title':(40,100,200),
        'tick_label':(255,0,0)
    },
    'text': {
        'line-thickness':1
    },
    'tick': {
        'color':(0,0,255),
        'radius':2
    },
    'extra': { # extra information like if x and y axis are numeric or categorical etc
        'color':(0,0,255),
        'font':cv2.FONT_HERSHEY_SIMPLEX,
        'scale':0.3
    }
}

TRAIN_DIR = normpath('../data/raw/train')
IMG_DIR = join(TRAIN_DIR, 'images')
ANNO_DIR = join(TRAIN_DIR, 'annotations')

assert os.path.exists(IMG_DIR)
assert os.path.exists(ANNO_DIR)


def _viz_chart_text(img:np.ndarray, anno:list, params:dict):
    '''
    Internal function for adding text annotations to image as bounding box
    '''
    for i in range(len(anno)):
        assert 'role' in anno[i] and 'polygon' in anno[i]
        role = anno[i]['role']
        poly = anno[i]['polygon']
        if not role in params['roles']:
            raise ValueError("Could not find role", role)
        color = params['roles'][role]
        for j in range(3):
            start = (poly['x' + str(j)], poly['y' + str(j)])
            end = (poly['x' + str(j+1)], poly['y' + str(j+1)])
            cv2.line(img, start, end, color, params['text']['line-thickness'])
        cv2.line(img, (poly['x3'],poly['y3']), (poly['x0'],poly['y0']), color, params['text']['line-thickness'])


def _viz_chart_axis(img:np.ndarray, anno:list, params:dict,
    extra_origin:tuple=None, extra_preface=''):
    '''
    Internal function for adding axis annotations to image
    '''
    # visualize tick points:
    for i in range(len(anno['ticks'])):
        assert 'tick_pt' in anno['ticks'][i]
        center = (anno['ticks'][i]['tick_pt']['x'],
                    anno['ticks'][i]['tick_pt']['y'])
        radius = params['tick']['radius']
        color = params['tick']['color']
        cv2.circle(img, center, radius, color, -1)
    if extra_origin is not None:
        assert isinstance(extra_origin, tuple)
        tick_type = 'default'
        if 'tick-type' in anno:
            tick_type = anno['tick-type'] # like 'markers'
        values_type = 'default'
        if 'values-type' in anno:
            values_type = anno['values-type'] # like 'numerical'
        s = f'{extra_preface}type: {tick_type} values: {values_type}'
        font = params['extra']['font']
        fontScale = params['extra']['scale']
        extra_color = params['extra']['color']
        extra_thickness = 1
        print(f'text: {s}, org:{extra_origin} font:{font} fontScale: {font}, color:{extra_color} thickness:{extra_thickness}) # , cv2.LINE_AA')
        cv2.putText(img, s, extra_origin, font, fontScale, extra_color, extra_thickness)


def viz_chart_annotations(img, anno:dict, params=PARAMS,
    img_title='chart image',
    show=True) -> np.ndarray:
    '''
    Display image combined with all Kaggle annotations
    '''
    if isinstance(img,str):
        print('Reading image', img)
        img = cv2.imread(img)
    img2 = copy.deepcopy(img)
    if 'text' in anno:
        _viz_chart_text(img2,anno['text'], params=params)
    else:
        raise ValueError("could not find field 'text' in annotation dictionary")
    if 'axes' in anno and 'x-axis' in anno['axes']:
        pass
        _viz_chart_axis(img2, anno['axes']['x-axis'], params=params,
            extra_origin=(5,10), extra_preface='x: ')
    else:
        raise ValueError("could not find field ['axes']['x-axis'] in annotation dictionary")
    if 'axes' in anno and 'y-axis' in anno['axes']:
        pass
        _viz_chart_axis(img2, anno['axes']['y-axis'], params=params,
            extra_origin=(5,20), extra_preface='y: ')
    else:
        raise ValueError("could not find field ['axes']['y-axis'] in annotation dictionary")
    if show:
        cv2.imshow(img_title, img2)
        cv2.waitKey(0)
    return img2


def test_viz_chart_annotations(id = 'aae9941fcd5a', show=True):
    imgfile = join(IMG_DIR, id + '.jpg')
    annofile = join(ANNO_DIR, id + '.json')
    assert os.path.exists(imgfile)
    assert os.path.exists(annofile)
    with open(annofile, 'r') as f:
        anno = json.load(f)
    viz_chart_annotations(imgfile, anno=anno, params=PARAMS, show=show)


def runall_viz_chart_annotations(imgdir= IMG_DIR,
    annodir= ANNO_DIR,
    n=-1,
    show=True):
    '''
    Extract data from all files in directories that correspond
    to vertical bar charts.
    '''
    dir_list = os.listdir(imgdir)
    nmax = len(dir_list)
    if n > 0: # do not compute all entries
        nmax = min(n, nmax)
        random.shuffle(dir_list)
    for i in range(nmax):
        file = dir_list[i]
        base = file[:-4]
        ifile = join(imgdir, file)
        assert os.path.exists(ifile)
        annofile = join(annodir, base + ".json")
        with open(annofile,'r') as f:
            anno = json.load(f)
        source = anno['source']
        print(f'ID: {base} source: {source}')
        print(anno.keys())
        img = cv2.imread(ifile)
        if show:
            cv2.imshow(base,img)
            cv2.waitKey(0)
        viz_chart_annotations(img=img, anno=anno, params=PARAMS,
            img_title=f'{base} annotated', show=show)
        cv2.destroyAllWindows()
    

if __name__ == '__main__':
    test_viz_chart_annotations()
    runall_viz_chart_annotations()

