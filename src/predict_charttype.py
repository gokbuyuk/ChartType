import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from icecream import ic
import os 
import tensorflow as tf
print(f"tensorflow version: {tf.__version__}")

# Save train and test sets to disk
data_dir = os.path.join('data','data_ml','interim')

x_train = np.load(os.path.join(data_dir,'x_train_charttype.npy'))
x_test = np.load(os.path.join(data_dir,'x_test_charttype.npy'))
y_train = pd.read_csv(os.path.join(data_dir,'y_train_charttype.csv'), header=0)
y_test = pd.read_csv(os.path.join(data_dir,'y_test_charttype.csv'), header=0)

print('shape:')
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(f"max pixel value before normalization: {x_train.max(), x_test.max()}")

type_distribution = pd.DataFrame([y_train['chart_type'].value_counts(normalize=True).round(3).to_dict(), 
                                  y_test['chart_type'].value_counts(normalize=True).round(3).to_dict()], 
                                 index=['y_train', 'y_test']).transpose().reset_index().rename(columns={'index': 'chart_type'})
type_distribution['chart_type'] = type_distribution['chart_type'].replace(['(', ')', ','], '')
print(type_distribution)

# Group by category_col and calculate the mean of column1 and column2
grouped = type_distribution.groupby('chart_type').mean()[['y_train', 'y_test']]

# Plot a barchart comparing column1 and column2 for each category
# x = np.arange(len(grouped.index))
# width = 0.35
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, grouped['y_train'], width, label='y_train')
# rects2 = ax.bar(x + width/2, grouped['y_test'], width, label='y_test')
# ax.set_xticks(x)
# ax.set_xticklabels(grouped.index)
# ax.legend()
# plt.show()
 
# Normalize so that the pixel values are between 0 and 1
norm_factor = x_train.max()
x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_train=x_train / norm_factor
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_test=x_test / norm_factor

input_shape = x_train.shape[1:]
ic(input_shape)
 
print(f"max pixel value after nnormalization: {x_train.max(), x_test.max()}")

# one-hot encode the 'category' column
one_hot_train = pd.get_dummies(y_train['chart_type'])
print('y train - one hot encoded')
print(one_hot_train.head(3))
# save one-hot encoded data as array of 1s and 0s
one_hot_array_train = one_hot_train.to_numpy()
y_train_arr = tf.constant(one_hot_array_train.astype(int))

one_hot_test = pd.get_dummies(y_test['chart_type'])
print('y test - one hot encoded')
print(one_hot_test.head(3))
one_hot_array_test = one_hot_test.to_numpy()
# add the one-hot encoded columns to the original dataframe
y_test_arr = tf.constant(one_hot_array_test.astype(int))

# plt.imshow(x_train[100][:,:,0])
# print(y_train_arr[100], x_train.shape)

## MODEL

## Define the Model

batch_size = 50
ic(batch_size)
num_classes = 5
ic(num_classes)
epochs = 3
ic(epochs)

ker_n = 7
kernel_size = (ker_n,ker_n)
ic(kernel_size)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size, strides=(2, 2), 
                           padding='same', activation='relu', 
                           input_shape=input_shape),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Dropout(0.20),
    tf.keras.layers.Conv2D(64, kernel_size, strides=(2, 2), 
                           padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(strides=(2,2)),
    tf.keras.layers.Dropout(0.20),
    tf.keras.layers.Conv2D(128, kernel_size, strides=(2, 2), 
                           padding='same', activation='relu'), 
    tf.keras.layers.MaxPool2D(strides=(2,2)),
    tf.keras.layers.Dropout(0.20),
    tf.keras.layers.Conv2D(256, kernel_size, strides=(2, 2), 
                           padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(strides=(2,2)),
    tf.keras.layers.Dropout(0.20),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model_label = f'model_{num_classes}_{batch_size}_{epochs}_{ker_n}'

model.save(os.path.join('models', f'chart_type_{model_label}.h5'))
# Print the model summary
model.summary()


## Fit the training data

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.995):
            print("\nReached 99.5% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()

## Testing the model on a validation dataset

history = model.fit(x_train, y_train_arr,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.1,
                    callbacks=[callbacks])

## Evaluate the loss and accuracy of our model

fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', 
           label="Training Loss")
ax[0].plot(history.history['val_loss'], color='r', 
           label="Validation Loss",
           axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training Accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation Accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.savefig(os.path.join('reports', 'figures', f"loss_accuracy_{model_label}.png"))

test_loss, test_acc = model.evaluate(x_test, y_test_arr)

class_labels = one_hot_test.columns.to_list()
# Predict the values from the testing dataset
Y_pred = model.predict(x_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred, axis = 1) 
# Map the class indices to class labels
y_pred_labels = [class_labels[i] for i in Y_pred_classes]

# Convert testing observations to one hot vectors
Y_true = np.argmax(y_test_arr,axis = 1)
# Map the class indices to class labels
# y_labels = [class_labels[i] for i in Y_true]

# compute the confusion matrix
confusion_mtx = tf.math.confusion_matrix(Y_true, Y_pred_classes) 

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='g')
plt.savefig(os.path.join('reports', 'figures', f"confusion_matrix_{model_label}.png"))
# plt.show()


# df_pred = pd.DataFrame({'y_true': Y_true, 'y_pred': Y_pred_classes})
test_pred = y_test.copy()
test_pred['predicted'] = Y_pred_classes
test_pred['predicted_labels'] = y_pred_labels
test_pred.to_csv(os.path.join('reports', f'predictions_{model_label}_test.csv'), index=False)