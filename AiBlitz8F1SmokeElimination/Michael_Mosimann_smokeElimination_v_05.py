import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
from glob import glob
import random
from tqdm.notebook import tqdm 
import pandas as pd
from fastai.vision.all import *
from fastai.data.core import *
from fastai.vision.gan import *

data_directiory = "data"
train_path = os.path.join(data_directiory, "train")
train_smoke_path = os.path.join(data_directiory, "train/smoke")
train_clear_path = os.path.join(data_directiory, "train/clear")
val_path = os.path.join(data_directiory, "val")
val_smoke_path = os.path.join(data_directiory, "val/smoke")
val_clear_path = os.path.join(data_directiory, "val/clear")
test_data_path = os.path.join(data_directiory, "test/smoke")
test_submission_path = "clear"

dls = ImageDataLoaders.from_folder(data_directiory,train="train",valid="val")
#dls = ImageDataLoaders.from_folder(data_directiory,train="train", valid="val",test="test")
print(dls.items)
print(len(dls.items))
print(dls.valid_ds.items)
print(len(dls.valid_ds.items))

generator = basic_generator(64, n_channels=3, n_extra_layers=1)
critic    = basic_critic   (64, n_channels=3, n_extra_layers=1)
#critic    = basic_critic   (64, n_channels=3, n_extra_layers=1, act_cls=partial(nn.LeakyReLU, negative_slope=0.2))

#learn = unet_learner(dls, models.resnet34, loss_func=CrossEntropyLossFlat(axis=1), y_range=(0,1))
#learn = GANLearner.wgan(dls, generator, critic, opt_func = RMSProp)
learn = GANLearner.wgan(dls, generator, critic, metrics="mse")
print(learn)
learn.metrics
print(dls)

learn.fine_tune(1)

