# Code from the site : https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/ .
# load, split and scale the maps dataset ready for training
from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed

# load all images in a directory into memory
def load_images(path,path2, size=(256,256)):
  src_list, tar_list = list(), list()
  # enumerate filenames in directory, assume all are images
  i = 0
  for filename in listdir(path):
    if (i == 5000):
      break
    else:
      # load and resize the image
      clear_img = load_img(path + filename, target_size=size)
      # convert to numpy array
      clear_img = img_to_array(clear_img)
      tar_list.append(clear_img)
      i = i +1
      print(i)
  i = 0
  for filename in listdir(path2):
    if (i == 5000):
      break
    else:
      i = i +1
      # load and resize the image
      smoke_img = load_img(path2 + filename, target_size=size)
      # convert to numpy array
      smoke_img = img_to_array(smoke_img)
      src_list.append(smoke_img)
  return [asarray(src_list), asarray(tar_list)]

# dataset path
path = 'data/train/clear/'
path2 = 'data/train/smoke/'
# load dataset
[src_images, tar_images] = load_images(path,path2)
print('Loaded: ', src_images.shape, tar_images.shape)
# save as compressed numpy array
filename = 'cars_256.npz'
savez_compressed(filename, src_images, tar_images)
print('Saved dataset: ', filename)
