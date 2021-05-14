import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
from glob import glob
import random
import numpy as np
from tqdm.notebook import tqdm
import keras
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
# Code parts from https://www.tensorflow.org/tutorials/load_data/images
# https://machinelearningmastery.com/how-to-load-large-datasets-from-directories-for-deep-learning-with-keras/
# Main code from https://www.aicrowd.com/showcase/baseline-f1-smoke-elimination

data_directiory = "data"
train_clear_path = os.path.join(data_directiory, "train/clear")
train_smoke_path = os.path.join(data_directiory, "train/smoke")
val_clear_path = os.path.join(data_directiory, "val/clear")
val_smoke_path = os.path.join(data_directiory, "val/smoke")
test_data_path = os.path.join(data_directiory, "test/smoke")
test_submission_path = "clear"

# create a data generator
datagen = ImageDataGenerator()

# load and iterate training dataset
train_it_smoke = datagen.flow_from_directory('data/train', shuffle=False,batch_size=100)
# train_it_smoke = datagen.flow_from_directory('data/train/smoke', shuffle=False,batch_size=100)
# train_it_clear = datagen.flow_from_directory('data/train/clear', shuffle=False, batch_size=100)
# load and iterate validation dataset
val_it_smoke = datagen.flow_from_directory('data/val', shuffle=False, batch_size=100)
# val_it_smoke = datagen.flow_from_directory('data/val/smoke', shuffle=False, batch_size=100)
# val_it_clear = datagen.flow_from_directory('data/val/clear', shuffle=False, batch_size=100)
# load and iterate test dataset
test_it_smoke = datagen.flow_from_directory('data/test', shuffle=False, batch_size=100)

# Output Image Width & hight
image_width, image_height = 256, 256

# Training area

# Defining the model

model = Sequential()
model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(image_width, image_height, 3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()

opt = Adam(lr=0.000001)
model.compile(optimizer = opt, loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['mse'])

# fit model
model.fit(train_it_smoke, steps_per_epoch=200, validation_data=val_it_smoke, validation_steps=20)
# model.fit(train_it_smoke, train_it_clear, steps_per_epoch=200, validation_data=(val_it_smoke, val_it_clear), validation_steps=20)

y_test_pred = model.predict(test_it_smoke, steps=50)

print(y_test_pred)
i = 0
# Getting though corrupted images folder
for img_name in tqdm(os.listdir(test_data_path)):

    # Opening the smoke image
    # img = Image.open(os.path.join(test_data_path, f"{img_name}"))
    #  TODO: matching smoked image with clear image
    img = y_test_pred[i]
    # Saving the output image :)
    img.save(os.path.join(test_submission_path, f"{img_name}"))
    i = i + 1
#  sample_output_img = plt.imread(os.path.join(test_submission_path, f"0.jpg"))
#  plt.imshow(sample_output_img)
#  plt.pause(0.05)

#  np.mean((real_img - predicted_img)**2)#Mean squared error
sumOfMSE = 0
