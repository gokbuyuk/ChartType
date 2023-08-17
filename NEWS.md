## chartscrape 0.1.0 2023-04-01

* Initial commit
* This project is centered around the task of extracting data from images of 2D charts and plots. Examples are images of bar charts, line charts and scatter plots where one wants to obtain the underlying data that was used to generate the plots. This project was motivated by an ongoing Kaggle competition of 'Making Charts Accessible'


## chartscrape 0.2.0 2023-04-04

* first version for parsing bar charts
* known issue if all factors appear numeric


## chartscrape 0.2.1 2023-04-09

* implemented visualization of ground-truth data
* implemented detection of horizontal gridlines
* implemented detection of vertical gridlines


## chartscrape 0.2.2 2023-04-10

* integrated keras-ocr as OCR engine
* filtering by assumptions of location in image (not finalized)

## chartscrape 0.2.3 2023-04-15

* creation of interim data for JSON and images
* detection of bounding box of all x-labels and all y-labels in ground-truth data
* creation of 3D array representating all images (greyscale, low-res)

## chartscrape 0.3.0 2023-04-16

* Initial implementation of Tensorflow based model for predicting bounding boxes of axis labels
* Collection of hyperparameters in a common dictionary


## chartscrape 0.3.1 2023-04-19

* Prediction of numeric vs categorical axis types using Conv-Nets
* Code for generating images of random charts (horizonal bar charts)
* Prediction of chart origin in pixel coordinates

