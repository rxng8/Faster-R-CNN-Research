import math

from . import *

class Config:
    def __init__(self) -> None:
        
        self.verbose = False

        self.img_folder = None
        self.annotation_path = None
        self.img_extension = ".jpg"

        self.img_size = (200, 300)
        self.img_min_side = 300

        self.img_shape = self.img_size + (3,)
        self.anchor_box_scales = [128, 256, 512]
        self.anchor_box_ratios = [(1, 1), (1, 2*math.sqrt(2)), (2*math.sqrt(2), 1)]

        self.num_rois = 4 # Number of RoIs to process at once.

        # Augmentation flag
        self.use_horizontal_flips = True # Augment with horizontal flips in training. 
        self.use_vertical_flips = True   # Augment with vertical flips in training. 
        self.rot_90 = True           # Augment with 90 degree rotations in training. 

        self.record_path = None
        self.model_path = None
        self.num_rois = 4

        self.base_net_weights = None

        # image channel-wise mean to subtract
        self.img_channel_mean = [103.939, 116.779, 123.68]
        self.img_scaling_factor = 1.0

        # stride at the RPN (this depends on the network configuration)
        self.rpn_stride = 16

        # scaling the stdev
        self.std_scaling = 4.0
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

        # overlaps for RPN
        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7

        # overlaps for classifier ROIs
        self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.5

        # placeholder for the class mapping, automatically generated by the parser
        self.class_mapping = None