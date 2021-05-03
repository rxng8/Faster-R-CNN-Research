
import math
import numpy as np
import copy

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy


def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w*h


def iou(a, b):
    # a and b should be (x1,y1,x2,y2)

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)



lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4

def rpn_loss_regr(num_anchors):
    """Loss function for rpn regression
    Args:
        num_anchors: number of anchors (9 in here)
    Returns:
        Smooth L1 loss function 
            0.5*x*x (if x_abs < 1)
            x_abx - 0.5 (otherwise)
    """
    def rpn_loss_regr_fixed_num(y_true, y_pred):

        # x is the difference between true value and predicted vaue
        x = y_true[:, :, :, 4 * num_anchors:] - y_pred

        # absolute value of x
        x_abs = tf.math.abs(x)

        # If x_abs <= 1.0, x_bool = 1
        x_bool = tf.cast(tf.math.less_equal(x_abs, 1.0), tf.float32)

        return lambda_rpn_regr * tf.reduce_sum(
            y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) \
            / tf.reduce_sum(epsilon + y_true[:, :, :, :4 * num_anchors])

    return rpn_loss_regr_fixed_num


def rpn_loss_cls(num_anchors):
    """Loss function for rpn classification
    Args:
        num_anchors: number of anchors (9 in here)
        y_true[:, :, :, :9]: [0,1,0,0,0,0,0,1,0] means only the second and the eighth box is valid which contains pos or neg anchor => isValid
        y_true[:, :, :, 9:]: [0,1,0,0,0,0,0,0,0] means the second box is pos and eighth box is negative
    Returns:
        lambda * sum((binary_crossentropy(isValid*y_pred,y_true))) / N
    """
    def rpn_loss_cls_fixed_num(y_true, y_pred):
        # print(y_true.shape) # (1,33,18,18)
        # print(y_pred.shape) # (1,33,18,9)
        return lambda_rpn_class * tf.reduce_sum(y_true[:, :, :, :num_anchors] \
            * tf.expand_dims(binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:]), axis=-1)) \
            / tf.reduce_sum(epsilon + y_true[:, :, :, :num_anchors])

    return rpn_loss_cls_fixed_num


def class_loss_regr(num_classes):
    """Loss function for rpn regression
    Args:
        num_anchors: number of anchors (9 in here)
    Returns:
        Smooth L1 loss function 
            0.5*x*x (if x_abs < 1)
            x_abx - 0.5 (otherwise)
    """
    def class_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4*num_classes:] - y_pred
        x_abs = tf.math.abs(x)
        x_bool = tf.cast(tf.math.less_equal(x_abs, 1.0), tf.float32)
        return lambda_cls_regr * tf.reduce_sum(y_true[:, :, :4*num_classes] \
            * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) \
            / tf.reduce_sum(epsilon + y_true[:, :, :4*num_classes])
    return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
    return lambda_cls_class * tf.reduce_mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
    # return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
