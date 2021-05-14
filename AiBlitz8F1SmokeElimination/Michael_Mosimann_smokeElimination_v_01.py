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
# Code parts from https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/
# Main code from https://www.aicrowd.com/showcase/baseline-f1-smoke-elimination

data_directiory = "data"
train_clear_path = os.path.join(data_directiory, "train/clear")
train_smoke_path = os.path.join(data_directiory, "train/smoke")
val_clear_path = os.path.join(data_directiory, "val/clear")
val_smoke_path = os.path.join(data_directiory, "val/smoke")
test_data_path = os.path.join(data_directiory, "test/smoke")
test_submission_path = "clear"

#  img = plt.imread(test_data_path+"/0.jpg")
#  plt.imshow(img)
#  plt.pause(0.05)
#  Load data
x_train =[]
y_train =[]
x_val =[]
y_val =[]

#  Train part. # TODO:
for img_name in tqdm(os.listdir(train_smoke_path)):
    img_smoke = plt.imread(os.path.join(train_smoke_path, f"{img_name}"))
    img_clear = plt.imread(os.path.join(train_clear_path, f"{img_name}"))
    x_train.append(img_smoke)
    y_train.append(img_clear)
    #  print(np.mean((img_smoke - img_clear)**2))
x_train = np.array(x_train, dtype="uint8") / 255
y_train = np.array(y_train, dtype="uint8") / 255

#  Val part. # TODO:
for img_name in tqdm(os.listdir(val_smoke_path)):
    img_smoke = plt.imread(os.path.join(val_smoke_path, f"{img_name}"))
    img_clear = plt.imread(os.path.join(val_clear_path, f"{img_name}"))
    x_val.append(img_smoke)
    y_val.append(img_clear)
    #  print(np.mean((img_smoke - img_clear)**2))
x_val = np.array(x_val, dtype="uint8") / 255
y_val = np.array(y_val, dtype="uint8") / 255

# Output Image Width & hight
image_width, image_height = 256, 256

# plt.figure()
# plt.imshow(trainData[1][0])# smoke
# plt.pause(0.05)
# plt.figure()
# plt.imshow(trainData[1][1])# clear
# plt.pause(0.05)

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

model.fit(x_train, y_train, epochs = 500, validation_data = (x_val, y_val))

x_train =[]
y_train =[]
x_val =[]
y_val =[]

#  Test part. # TODO:
dataTest = []
for img_name in tqdm(os.listdir(test_data_path)):
    img_smoke = plt.imread(os.path.join(test_data_path, f"{img_name}"))
    dataTest.append(img_smoke)
    #  print(np.mean((img_smoke - img_clear)**2))
# x_Test = np.array(dataTest) / 255
x_Test = np.array(dataTest)

y_test_pred = model.predict(x_Test)

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
# sumOfMSE = 0
