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

import cv2

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16, MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import Progbar

from frcnn import Config
from frcnn.models import vgg_base, rpn_network, classifier_layer, rpn_to_roi
from frcnn.data import data_generator
from frcnn.utils import get_img_output_length, show_img, calc_iou
from frcnn.losses import rpn_loss_cls, rpn_loss_regr, class_loss_cls, class_loss_regr

from frcnn.data import *

# %%


## Evaluating notebook

