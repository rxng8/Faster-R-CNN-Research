from . import *

class Config:
    def __init__(self) -> None:
        
        self.img_size = (800, 600)
        self.img_shape = self.img_size + (3,)
        self.anchor_ratios = []
        self.anchor_sizes = []

        self.num_rois = 4 # Number of RoIs to process at once.

        # Augmentation flag
        self.horizontal_flips = True # Augment with horizontal flips in training. 
        self.vertical_flips = True   # Augment with vertical flips in training. 
        self.rot_90 = True           # Augment with 90 degree rotations in training. 

        self.record_path = None
        self.model_path = None
        self.num_rois = 4

        self.base_net_weights = None

        self.img_scaling_factor = 1


    