"""
Author: Alex Nguyen
Gettysburg College
"""

# %%

import os
import time

import random

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16, MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory

from frcnn import Config as C
from frcnn.models import vgg_base

# These configs are to put into the config files, 
# but put here to easily testing
img_size = (800, 600)
img_shape = img_size + (3,)
anchor_ratios = []
anchor_sizes = []

# These configs are temporary configs
BATCH_SIZE = 32
EPOCHS = 1


# %%

vgg = build_vgg_base(img_shape=img_shape)




