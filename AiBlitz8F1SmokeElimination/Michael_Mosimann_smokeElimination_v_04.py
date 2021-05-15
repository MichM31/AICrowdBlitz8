if __name__ == '__main__': #évite des problem de broken pipe, merci à tanujjain sur https://github.com/idealo/imagededup/issues/67

    import pandas as pd
    from fastai.vision.all import *
    from fastai.data.core import *
    import os
    import cv2
    import matplotlib.pyplot as plt
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

    # load data
    data_folder = "data"

    # converting the images to gray to see if results will improve.
    # Training phase, create model
    dls = ImageDataLoaders.from_folder(data_folder,train="train", valid="valid",test="test")
