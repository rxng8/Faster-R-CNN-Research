
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16, MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory

def vgg_base(img_shape=(800, 600, 3), verbose=False):
    """ Generate a VGG model

    Args:
        img_shape (tuple, optional): [description]. Defaults to (800, 600, 3).
        verbose (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    base_model = VGG16(
        input_shape=img_shape,
        include_top=False,
        weights='imagenet'
    )
    # Get all the layers except for the last layer.
    model = Model(
        inputs=base_model.input,
        outputs=base_model.get_layer('block5_conv3').output,
        name='base_model'
    )
    if verbose:
        model.summary()
    return model

def rpn_network(n_anchors=9):
    """ Return 2 layers that are in the RPN network

    Args:
        n_anchors (int, optional): [description]. Defaults to 9.

    Returns:
        [type]: [description]
    """

    classify_layer = layers.Conv2D(
        n_anchors,
        (1, 1),
        padding='same',
        activation='sigmoid',
        kernel_initializer='uniform',
        name='rpn_out_classify'
    )

    regress_layer = layers.Conv2D(
        n_anchors * 4,
        (1, 1),
        padding='same',
        activation='linear',
        kernel_initializer='zero',
        name='rpn_out_regress'
    )

    return [classify_layer, regress_layer]


def RPN_to_ROI():
    pass

class ROIPooling(Model):
    def __init__(self):
        super().__init__()

class Classifier(Model):
    def __init__(self):
        super().__init__()