"""This file contains code to create training and test sets to predict tick type for x&y axes.
It walks through the root_dir directory and read in each image file 
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

def print_dict_structure(d, indent=0):
    for key, value in d.items():
        print('    ' * indent + str(key))
        if isinstance(value, dict):
            print_dict_structure(value, indent+1)


mode = '_test' # '_test' for running only a sample of images to test and save them in 'data_ml_test' folder
test_sample_size = 1000 

img_dir = os.path.join('data','train','images')
json_dir = os.path.join('data','train','annotations') 
output_img_dir = os.path.join('data', f'data_ml{mode}', 'interim')
output_dir = os.path.join('data', f'data_ml{mode}', 'processed')

if mode != '':
    print(f"Running in test mode. Only {test_sample_size} images will be processed. Outputs will be saved in {output_dir}")

width, height = (135, 135) # change this to edit the resolution
crop_fraction = 1/3
img_size = (width, height)  # desired size of each image
file_names = []
xtick_types = []
ytick_types = []


# initialize numpy arrays for xaxis and yaxis
if width == height:
    short = int(width*crop_fraction)
    long = int(width)
    xaxis = np.empty((0, short, long, 1))
    yaxis = np.empty((0, long, short, 1))
else: 
    xaxis = np.empty((0, int(width*crop_fraction), height, 1))
    yaxis = np.empty((0, width, int(height*crop_fraction), 1))

for subdir, dirs, files in os.walk(img_dir):
    n_files = len(files)
    
    for idx, file in enumerate(files):
        if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
            file_path = os.path.join(subdir, file)
            filename = file.split('.')[0]
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            resized = cv2.resize(img, img_size, interpolation= cv2.INTER_CUBIC)
            
            # crop bottom 1/3 and add to xaxis
            xaxis_crop = resized[(width-int(width*crop_fraction)): , :, np.newaxis]
            xaxis = np.append(xaxis, [xaxis_crop], axis=0)
            cv2.imwrite(os.path.join(output_img_dir, filename + '_xaxis.jpg'), xaxis_crop)

            # crop left 1/3 and add to yaxis
            yaxis_crop = resized[:, :int(height*crop_fraction), np.newaxis] 
            # print(yaxis_crop.shape, yaxis.shape)
            yaxis = np.append(yaxis, [yaxis_crop], axis=0)
            cv2.imwrite(os.path.join(output_img_dir, filename + '_yaxis.jpg'), yaxis_crop)
            
            # save the annotations
            file_names.append(file)
            
            with open(os.path.join(json_dir, filename + '.json'), 'r') as f:
                json_data = json.load(f)
                # print_dict_structure(json_data)
                xtick_type = json_data['axes']['x-axis']['tick-type']
                xtick_types.append(xtick_type)
                ytick_type = json_data['axes']['y-axis']['tick-type']
                ytick_types.append(ytick_type)
        
        print(f"image {idx+1}/{n_files} with dimensions: {(xaxis.shape, yaxis.shape)} added to the xtick and ytick arrays.")
                
        if idx==test_sample_size-1: 
            if mode != '':
                print(f"Ran in test mode, only {test_sample_size} images are processed") 
                break

# save numpy arrays
np.save(os.path.join(output_dir, 'xaxis_all'), xaxis)
np.save(os.path.join(output_dir, 'yaxis_all'), yaxis)

df_tick_types = pd.DataFrame({'filename': file_names, 
                              'xtick_array': list(xaxis), 
                              'ytick_array': list(yaxis), 
                              'xtick_type': xtick_types,
                              'ytick_type': ytick_types
                              })
df_tick_types.to_csv(os.path.join(output_dir,'tick_types_all.csv'), index=False)
# ic(df_tick_types)


# Extract the image arrays and class labels for train set
for tick in ['xtick', 'ytick']:
    label = f'{tick}_type'
    # Split the dataframe into train and test dataframes
    train_df, test_df = train_test_split(df_tick_types, 
                                         test_size=0.10, 
                                         stratify=df_tick_types[label] )

    type_distribution = pd.DataFrame([train_df[label].value_counts().to_dict(), 
                                    train_df[label].value_counts(normalize=True).round(3).to_dict(), 
                                    test_df[label].value_counts().to_dict(), 
                                    test_df[label].value_counts(normalize=True).round(3).to_dict()], 
                                        index=['y_train_count', 'y_train_prop', 'y_test_count', 'y_test_prop']).transpose().reset_index().rename(columns={'index': 'chart_type'})
    for subset in ['train', 'test']:
        ic(tick, subset, type_distribution)
        tick_df = locals()[f"{subset}_df"]
        tick_filenames = tick_df['filename'].tolist()
        tick_labels = tick_df[['filename', label]]
        tick_images = np.array(tick_df[f'{tick}_array'].tolist())
        tick_labels.to_csv(os.path.join(output_dir,f'y_{subset}_{tick}.csv'), index=False)

        np.save(os.path.join(output_dir,f'y_{subset}_{tick}.npy'), tick_images)

print(f'Train and test datasets are created and saved in {output_dir}')
