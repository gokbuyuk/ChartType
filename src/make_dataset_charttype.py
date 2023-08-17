"""This file contains code to walk through the root_dir directory and read in each image file 
    that has a file extension of jpg, png, or jpeg.
It resizes each image to the desired size and appends the resulting image array to a list.
It also saves the file names in a separate list.
The list of image arrays is then converted to a numpy array and reshaped to have a 2D shape \
    with dimensions of (number of images, flattened image array). 
This 2D numpy array is then used to create a Pandas DataFrame with the file names as the index \
    and saved as a csv at output_dir.
"""  

import os
import cv2
import numpy as np
import pandas as pd
import json
# import sklearn
from sklearn.model_selection import train_test_split
from icecream import ic

mode = '' # '_test' for running only a sample of images to test and save them in 'data_ml_test' folder

img_dir = os.path.join('data','train','images')
json_dir = os.path.join('data','train','annotations') 
output_dir = os.path.join('data', f'data_ml{mode}', 'interim')

width, height = (136, 136)
img_size = (width, height)  # desired size of each image
img_array_list = []
file_names = []
chart_types = []
    
for subdir, dirs, files in os.walk(img_dir):
    n_files = len(files)
    for idx, file in enumerate(files):
        if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
            file_path = os.path.join(subdir, file)
            filename = file.split('.')[0]
            img = cv2.imread(file_path, 0) # 0 for grayscale
            img = cv2.resize(img, img_size, interpolation= cv2.INTER_CUBIC)
            
            # save image 
            output_path = os.path.join(output_dir, 'images', filename + '.jpg')
            # cv2.imshow(filename, img)
            cv2.imwrite(output_path, img)
            
            img_array_list.append(np.array(img))
            file_names.append(file)
            # if there are additional images generated, include the names and types here: 
            if 'horizbar' in file: 
                chart_types.append('horizontal_bar')
            elif 'scatter' in file: 
                chart_types.append('scatter')
            else: 
                with open(os.path.join(json_dir, filename + '.json'), 'r') as f:
                    json_data = json.load(f)
                    chart_type = json_data['chart-type']
                    chart_types.append(chart_type)
                
                    
            print(f"image {idx+1}/{n_files} added to the array. dimensions: {img.shape}")
        if mode != '':    
            if idx==100: 
                break

df_chart_types = pd.DataFrame({'filename': file_names, 'image_array': img_array_list, 'chart_type': chart_types})
# Split the dataframe into train and test dataframes
train_df, test_df = train_test_split(df_chart_types, test_size=0.10, stratify=df_chart_types['chart_type'] )


type_distribution = pd.DataFrame([train_df['chart_type'].value_counts().to_dict(), 
                                  train_df['chart_type'].value_counts(normalize=True).round(3).to_dict(), 
                                  test_df['chart_type'].value_counts().to_dict(), 
                                  test_df['chart_type'].value_counts(normalize=True).round(3).to_dict()], 
                                    index=['y_train_count', 'y_train_prop', 'y_test_count', 'y_test_prop']).transpose().reset_index().rename(columns={'index': 'chart_type'})
# type_distribution['chart_type'] = type_distribution['chart_type'].replace(['(', ')', ','], '')
ic(type_distribution)


# Extract the image arrays and class labels for train set
train_filenames = train_df['filename'].tolist()
train_labels = train_df[['filename', 'chart_type']]
train_images = np.array(train_df['image_array'].tolist())

# Extract the image arrays and class labels for test set
test_filenames = test_df['filename'].tolist()
test_labels = test_df[['filename', 'chart_type']]
test_images = np.array(test_df['image_array'].tolist())

# Save train and test sets to disk
np.save(os.path.join(output_dir,'x_train_charttype.npy'), train_images)
np.save(os.path.join(output_dir,'x_test_charttype.npy'), test_images)
train_labels.to_csv(os.path.join(output_dir,'y_train_charttype.csv'), index=False)
test_labels.to_csv(os.path.join(output_dir,'y_test_charttype.csv'), index=False)


