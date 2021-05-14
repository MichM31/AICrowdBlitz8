import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
from glob import glob
import random
import numpy as np
from tqdm.notebook import tqdm
#import keras

data_directiory = "data"
train_clear_path = os.path.join(data_directiory, "train/clear")
train_smoke_path = os.path.join(data_directiory, "train/smoke")
val_clear_path = os.path.join(data_directiory, "val/clear")
val_smoke_path = os.path.join(data_directiory, "val/smoke")
test_data_path = os.path.join(data_directiory, "test/smoke")
test_submission_path = "clear"

img = plt.imread(test_data_path+"/0.jpg")
plt.imshow(img)
plt.pause(0.05)

#Train test part. # TODO:
#
#for img_name in tqdm(os.listdir(train_smoke_path)):
    #img_smoke = plt.imread(os.path.join(train_smoke_path, f"{img_name}"))
    #img_clear = plt.imread(os.path.join(train_clear_path, f"{img_name}"))
    #print(np.mean((img_smoke - img_clear)**2))

# Output Image Width & hight
image_width, image_height = 256, 256

# Getting though corrupted images folder
for img_name in tqdm(os.listdir(test_data_path)):

    # Opening the smoke image
    img = Image.open(os.path.join(test_data_path, f"{img_name}"))
    ## TODO: matching smoked image with clear image

    # Saving the output image :)
    img.save(os.path.join(test_submission_path, f"{img_name}"))

sample_output_img = plt.imread(os.path.join(test_submission_path, f"0.jpg"))
plt.imshow(sample_output_img)
plt.pause(0.05)

#np.mean((real_img - predicted_img)**2)#Mean squared error
sumOfMSE = 0
