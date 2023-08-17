
import keras_ocr
from matplotlib import pyplot as plt
import os
import pandas as pd


def image_to_kerasocr(imgfile):
    '''
    Applies keras-ocr to input image
    '''
    pipeline = keras_ocr.pipeline.Pipeline()
    print('Applying keras-ocr to image file:', imgfile)
    if isinstance(imgfile, str):
        assert os.path.exists(imgfile)
    images = []
    images.append(keras_ocr.tools.read(imgfile))
    predictions = pipeline.recognize(images)
    return predictions[0] # only one image


def kerasocr_to_table(prediction:list):
    '''
    Converts OCR results from keras_ocr from list/tuple
    representation to a dataframe
    
    Args:
        prediction(list): A list of OCR results as returned
        from keras_ocr corresponding to a single image.
        It is a list, and each list element is a 2-tuple
        consisting of a text string and a 4x2 numpy array
        corresponding to 4 corners of a bounding box, each
        with an x and y position
    
    Returns:
        pd.DataFrame: Returns a dataframe with column 'Text'
            for the OCR recognized text and 8 columns corresponding
            to x and y coordinates of the 4 corner points of the
            bounding box. Names are tlx, tly for top-left x and y
            coordiantes and so for (trx, try, brx,bry, blx, bly)
    '''
    texts, tlx, tly, trx, _try, brx, bry, blx, bly = [],[],[],[],[],[],[],[],[]

    for txt,bb in prediction:
        texts.append(txt)
        tlx.append(bb[0,0])
        tly.append(bb[0,1])
        trx.append(bb[1,0])
        _try.append(bb[1,1])
        brx.append(bb[2,0])
        bry.append(bb[2,1])
        blx.append(bb[3,0])
        bly.append(bb[3,1])
    df = pd.DataFrame(data={'text':texts,
        'tlx':tlx,'tly':tly,'trx':trx, 'try':_try, 'brx':brx,'bry':bry,
            'blx':blx, 'bly':bly})
    return df



def test_kerasocr_recognition(infile='../data/raw/train/images/45df1fe3293b.jpg'):
    prediction = image_to_kerasocr(infile)
    # print(prediction)
    df = kerasocr_to_table(prediction)
    print(df)
    assert False

    # Display the recognized text with bounding boxes
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap='gray')
    for pred, (l, t, r, b) in results:
        ax.add_patch(plt.Rectangle((l, t), r-l, b-t, fill=False, color='red', linewidth=2))
        ax.text(l, t-10, pred, color='red')
    plt.show()


if __name__ == '__main__':
    test_kerasocr_recognition()
