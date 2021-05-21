import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
from glob import glob
import random
from tqdm.notebook import tqdm
import pandas as pd
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

data_directiory = "data"
train_path = os.path.join(data_directiory, "train")
train_smoke_path = os.path.join(data_directiory, "train/smoke")
train_clear_path = os.path.join(data_directiory, "train/clear")
val_path = os.path.join(data_directiory, "val")
val_smoke_path = os.path.join(data_directiory, "val/smoke")
val_clear_path = os.path.join(data_directiory, "val/clear")
test_data_path = os.path.join(data_directiory, "test/smoke")
test_submission_path = "clear"

# example of loading a pix2pix model and using it for one-off image translation
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot

# load an image
def load_image(filename, size=(256,256)):
	# load image with the preferred size
	pixels = load_img(filename, target_size=size)
	# convert to numpy array
	pixels = img_to_array(pixels)
	# scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5
	# reshape to 1 sample
	pixels = expand_dims(pixels, 0)
	return pixels

# load model
model = load_model('model_040000.h5')#model_040000.h5 , model_060000.h5
# load source image
for img_name in tqdm(os.listdir(test_data_path)):
  src_image = load_image(os.path.join(test_data_path, f"{img_name}"))
  #print('Loaded', src_image.shape)
  # generate image from source
  gen_image = model.predict(src_image)
  save_img(os.path.join(test_submission_path, f"{img_name}"),gen_image[0])
