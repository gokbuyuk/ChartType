
import math
import json
import os
import cv2
import copy
from os.path import join, normpath
import numpy as np
import pickle

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow import keras

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from chartparams import CHART_PARAMS as PARAMS

# this ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available())
# this ensures that the current current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())

device = torch.device("mps") # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


def axis_bb_convnet_tf_model(
    input_shape=(
        int(PARAMS['thumbx']),
        int(PARAMS['thumby']), 1)):
    '''
    Construct tensorflow framework for convolutional neural network
    Assumes 12-output dimension (number of columns of y).
    The meaning of those 12 dimension is:
    0-3: bounding box of x-axis tick labels
    4-7: bounding box of y-axis tick labels
    8: flag if x-axis is numeric (0) or categorical (1)
    9: flag if y-axis is numeric (0) or categorical (1)
    10: x-coordinate of chart origin in pixel
    11: y-coordinate of chart origin in pixel
    '''
    # Define the model architecture
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
   
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(12, activation='sigmoid')) # two bounding boxes
    # if using classification it would look like this:
    # model.add(layers.Dense(5, activation='softmax')) # two bounding boxes
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    return model


# def bb_images_bb_to_mldata():

def end_to_end_mnist_tensorflow():
    '''
    Preprocessing, training and validation of the MNIST
    dataset using Tensorflow
    '''
    num_classes = 10
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # ds_train = ds_train.map(
    #     normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    # ds_train = ds_train.cache()
    # ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    # ds_train = ds_train.batch(128)
    # ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # ds_test = ds_test.map(
    #     normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    # ds_test = ds_test.batch(128)
    # ds_test = ds_test.cache()
    # ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
    # tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dense(10)
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
        ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        x_train, # ds_train,
        y_train,
        epochs=6,
        # validation_data=ds_test,
    )



def end_to_end_mnist_pytorch():
    '''
    Preprocessing, training and validation of the MNIST
    dataset using Tensorflow
    @see <https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118>
    '''

    train_data = datasets.MNIST(
        root = 'data',
        train = True,                         
        transform = ToTensor(), 
        download = True,            
    )
    test_data = datasets.MNIST(
        root = 'data', 
        train = False, 
        transform = ToTensor()
    )
    print(train_data)
    print(train_data.data.size())

    plt.imshow(train_data.data[0], cmap='gray')
    plt.title('%i' % train_data.targets[0])
    plt.show()
    figure = plt.figure(figsize=(10, 8))
    cols, rows = 5, 5
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_data), size=(1,)).item()
        img, label = train_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


    loaders = {
        'train' : torch.utils.data.DataLoader(train_data, 
                                            batch_size=100, 
                                            shuffle=True, 
                                            num_workers=1),
        
        'test'  : torch.utils.data.DataLoader(test_data, 
                                            batch_size=100, 
                                            shuffle=True, 
                                            num_workers=1),
    }
    loaders



    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Sequential(         
                nn.Conv2d(
                    in_channels=1,              
                    out_channels=16,            
                    kernel_size=5,              
                    stride=1,                   
                    padding=2,                  
                ),                              
                nn.ReLU(),                      
                nn.MaxPool2d(kernel_size=2),    
            )
            self.conv2 = nn.Sequential(         
                nn.Conv2d(16, 32, 5, 1, 2),     
                nn.ReLU(),                      
                nn.MaxPool2d(2),                
            )
            # fully connected layer, output 10 classes
            self.out = nn.Linear(32 * 7 * 7, 10)
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
            x = x.view(x.size(0), -1)       
            output = self.out(x)
            return output, x    # return x for visualization

    cnn = CNN()
    print(cnn)

    loss_func = nn.CrossEntropyLoss()   
    loss_func
            

    optimizer = optim.Adam(cnn.parameters(), lr = 0.01)   
    optimizer


 
    num_epochs = 5
    def train(num_epochs, cnn, loaders):
        
        cnn.train()
            
        # Train the model
        total_step = len(loaders['train'])
            
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(loaders['train']):
                
                # gives batch data, normalize x when iterate train_loader
                b_x = Variable(images)   # batch x
                b_y = Variable(labels)   # batch y
                output = cnn(b_x)[0]               
                loss = loss_func(output, b_y)
                
                # clear gradients for this training step   
                optimizer.zero_grad()           
                
                # backpropagation, compute gradients 
                loss.backward()    
                # apply gradients             
                optimizer.step()                
                
                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                pass
            
            pass
    
    
    pass
    train(num_epochs, cnn, loaders)

    def test():
        # Test the model
        cnn.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in loaders['test']:
                test_output, last_layer = cnn(images)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
                pass
        print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)
        

    test()

DEBUG = False
SHOW = False

PROC_DIR = normpath("../data/processed/train")
MODEL_DIR = normpath("../models")
assert os.path.exists(PROC_DIR)


def axis_bb_mldata(basename, procdir=PROC_DIR):
    '''
    Reads X and y values from processed image

    Returns X_train, y_train
    '''
    files = os.listdir(procdir)
    X_train_file = join(procdir, basename + '_X_train.pkl')
    y_train_file = join(procdir, basename + '_y_train.pkl')
    # open the binary file in read mode
    with open(X_train_file, 'rb') as file:
        # read the data from the file using the fromfile method
        X_train = pickle.load(file)
        print("read files X:",X_train.shape)
    with open(y_train_file, 'rb') as file:
        # read the data from the file using the fromfile method
        y_train = pickle.load(file)
        print("read files y:", y_train.shape)
        print('y_train first rows:', y_train[:10,:])
    for i in range(3):
        for j in range(8):
            assert y_train[i,j] >= 0.0
            assert y_train[i,j] <= 1.0
    return X_train, y_train


def train_bb_images_tf(basename):
    '''
    Generates X_train, y_train, Tensorflow model
    and trains the model
    '''
    X_train, y_train = axis_bb_mldata(basename)
    model = axis_bb_convnet_tf_model()
    print('ok got data and model!')
    model.fit(X_train, y_train)
    return model


def run_train_bb_images_tf(
    basename = 'axislabelbounds',
    seriesname = PARAMS['series'],
    modeldir = MODEL_DIR):
    '''
    Convenience function that reads compiled data, creates, trains and saves an ML model,
    '''
    model = train_bb_images_tf(basename)
    modeloutfile = join(modeldir, basename + '_' + seriesname + ".h5")
    print(f"Saving model to {modeloutfile}")
    model.save(modeloutfile)
    return model


if __name__ == '__main__':
    # train_axis_label_model(basename='axislabelbounds')
    run_train_bb_images_tf() # 'axislabelbounds'
    # end_to_end_mnist_pytorch()