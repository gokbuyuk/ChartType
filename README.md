chartscrape
==============================

This project is centered around the task of extracting data from images of 2D charts and plots. Examples are images of bar charts, line charts and scatter plots where one wants to obtain the underlying data that was used to generate the plots. This project was motivated by an ongoing Kaggle competition of 'Making Charts Accessible'

## Usage

```
cd src
python make_dataset.py # create interim data under data/interim/train
python build_features.py # build dataset under data/processed for machine learning
python train_axes.py # train model for predicting bounding boxes of x and y axis
python predict_axes.py # predictions of axes model
```

Under `notebooks` are several Jupyter notebooks that demonstrate aspects of this package.


## Journey

So far:

* Extracting words using different types of OCR engines: tesseract, easyocr, keras-ocr
* Multi-angle OCR
* Predicting grid-lines using regular line detection
* Predicting tick
* deep learning to predict bounding box of x-labels and y-labels 

To do:

* increase accuracy of axis predictions
* predictions for separate chart types



## Installation


### cv2

Problems with OpenCV2 version and keras_ocr. Had to downgrade openCV2 version to
`pip install opencv-python==4.5.5.64`
See <https://stackoverflow.com/questions/72706073/attributeerror-partially-initialized-module-cv2-has-no-attribute-gapi-wip-gs>

### tensorflow

Only needed for keras_ocr
Needed to use special pip command for Mac OS:
`python -m pip install tensorflow-macos`

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io




--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
