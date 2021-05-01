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

from tensorflow.keras.optimizers import Adam, SGD, RMSprop

from frcnn import Config
from frcnn.models import vgg_base
from frcnn.data import data_generator
from frcnn.utils import get_img_output_length, show_img

from frcnn.data import *

# These configs are to put into the config files, 
# but put here to easily testing

C = Config()
C.img_size = (800, 600)
C.img_min_side = 300
C.img_shape = C.img_size + (3,)
C.anchor_ratios = []
C.anchor_sizes = []

C.num_rois = 4 # Number of RoIs to process at once.

# Augmentation flag
C.horizontal_flips = True # Augment with horizontal flips in training. 
C.vertical_flips = True   # Augment with vertical flips in training. 
C.rot_90 = True           # Augment with 90 degree rotations in training. 

C.img_folder = "./dataset/open_image"
C.annotation_path = "./dataset/train-annotations-bbox.csv"
C.img_extension = ".jpg"

# These configs are temporary configs

BASE_PATH = "./dataset/open_image"
ANNOTATION_PATH = "./dataset/train-annotations-bbox.csv"

BATCH_SIZE = 32
EPOCHS = 1

train_path = ""

# %%

def get_data(annotation_path: str):
	"""
	Returns:
		List[Dict]: all_data: List[Dict[filepath, width, height, list(bboxes)]]. E.g:
			[{'filepath': 'c4879393a7637d4b',
			'width': 660,
			'height': 1024,
			'bboxes': [{'class': 'Shorts',
				'xmin': 0.201746,
				'xmax': 0.382153,
				'ymin': 0.448125,
				'ymax': 0.643125}, ...]
		Dict[str, int]: classes_count
	"""
	found_bg = False
	all_imgs = {}
	classes_count = {}
	class_mapping = {}
	visualise = True

	df = pd.read_csv(annotation_path)
	df_new = df[['ImageID', 'XMin', 'XMax', 'YMin', 'YMax', 'ClassName']]

	for i, row in df_new.iterrows():
		[filename, xmin, xmax, ymin, ymax, class_name] = row.to_numpy()
		if class_name not in classes_count:
			classes_count[class_name] = 1
		else:
			classes_count[class_name] += 1

		if filename not in all_imgs:
			all_imgs[filename] = {}
            
			img = cv2.imread(os.path.join(C.img_folder, filename + C.img_extension))
			# show_img(img)
			(rows, cols) = img.shape[:2]
			all_imgs[filename]['filepath'] = os.path.join(C.img_folder, filename + C.img_extension)
			all_imgs[filename]['width'] = cols
			all_imgs[filename]['height'] = rows
			all_imgs[filename]['bboxes'] = []
		
		all_imgs[filename]['bboxes'].append({
			'class': class_name,
			'xmin': xmin * float(cols),
			'xmax': xmax * float(cols),
			'ymin': ymin * float(rows),
			'ymax': ymax * float(rows)
		})
	
	all_data = []
	for key in all_imgs:
		all_data.append(all_imgs[key])  
	return all_data, classes_count, class_mapping

# all_data, classes_count, class_mapping = get_data(C.annotation_path)


# %%
# while True:
# for img_data in all_data:
#     # try:
#     # print(img_data)
#     img_data_aug, x_img = augment(img_data, C, augment=True)

#     (width, height) = (img_data_aug['width'], img_data_aug['height'])
#     (rows, cols, _) = x_img.shape

#     assert cols == width
#     assert rows == height

#     # get image dimensions for resizing
#     (resized_width, resized_height) = get_new_img_size(width, height, C.img_min_side)

#     print(width, height)
#     print(resized_width, resized_height)

#     # resize the image so that smaller side is length = 300px, the bboxes is resized accordingly in the cal_rpn method
#     x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
#     debug_img = x_img.copy()

#     # try:
#     y_rpn_cls, y_rpn_regr, num_pos = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height, get_img_output_length)
#     # except:
#     #     continue

#     # except Exception as e:
#     #     print(e)
#     #     continue
#     # print(num_pos)


#     break


# %%

#--------------------------------------------------------#
# This step will spend some time to load the data        #
#--------------------------------------------------------#
st = time.time()
train_imgs, classes_count, class_mapping = get_data(C.annotation_path)
print()
print('Spend %0.2f mins to load the data' % ((time.time()-st)/60) )

# %%

# Get train data generator which generate X, Y, image_data
data_gen_train = data_generator(train_imgs, C, get_img_output_length, mode='train')

# %%

# Explore 

X, Y, image_data, debug_img, debug_num_pos = next(data_gen_train)


print('Original image: height=%d width=%d'%(image_data['height'], image_data['width']))
print('Resized image:  height=%d width=%d C.im_size=%d'%(X.shape[1], X.shape[2], C.img_min_side))
print('Feature map size: height=%d width=%d C.rpn_stride=%d'%(Y[0].shape[1], Y[0].shape[2], C.rpn_stride))
print(X.shape)
print(str(len(Y))+" includes 'y_rpn_cls' and 'y_rpn_regr'")
print('Shape of y_rpn_cls {}'.format(Y[0].shape))
print('Shape of y_rpn_regr {}'.format(Y[1].shape))
print(image_data)

print('Number of positive anchors for this image: %d' % (debug_num_pos))
if debug_num_pos==0:
    gt_x1, gt_x2 = image_data['bboxes'][0]['xmin']*(X.shape[2]/image_data['height']), \
		image_data['bboxes'][0]['xmax']*(X.shape[2]/image_data['height'])
    gt_y1, gt_y2 = image_data['bboxes'][0]['ymin']*(X.shape[1]/image_data['width']), \
		image_data['bboxes'][0]['ymax']*(X.shape[1]/image_data['width'])
    gt_x1, gt_y1, gt_x2, gt_y2 = int(gt_x1), int(gt_y1), int(gt_x2), int(gt_y2)

    img = debug_img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    color = (0, 255, 0)
    cv2.putText(img, 'gt bbox', (gt_x1, gt_y1-5), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
    cv2.rectangle(img, (gt_x1, gt_y1), (gt_x2, gt_y2), color, 2)
    cv2.circle(img, (int((gt_x1+gt_x2)/2), int((gt_y1+gt_y2)/2)), 3, color, -1)

    plt.grid()
    plt.imshow(img)
    plt.show()
else:
    cls = Y[0][0]
    pos_cls = np.where(cls==1)
    print(pos_cls)
    regr = Y[1][0]
    pos_regr = np.where(regr==1)
    print(pos_regr)
    print('y_rpn_cls for possible pos anchor: {}'.format(cls[pos_cls[0][0],pos_cls[1][0],:]))
    print('y_rpn_regr for positive anchor: {}'.format(regr[pos_regr[0][0],pos_regr[1][0],:]))

    gt_x1, gt_x2 = image_data['bboxes'][0]['xmin']*(X.shape[2]/image_data['width']), \
		image_data['bboxes'][0]['xmax']*(X.shape[2]/image_data['width'])
    gt_y1, gt_y2 = image_data['bboxes'][0]['ymin']*(X.shape[1]/image_data['height']), \
		image_data['bboxes'][0]['ymax']*(X.shape[1]/image_data['height'])
    gt_x1, gt_y1, gt_x2, gt_y2 = int(gt_x1), int(gt_y1), int(gt_x2), int(gt_y2)

    img = debug_img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    color = (0, 255, 0)
    #   cv2.putText(img, 'gt bbox', (gt_x1, gt_y1-5), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
    cv2.rectangle(img, (gt_x1, gt_y1), (gt_x2, gt_y2), color, 2)
    cv2.circle(img, (int((gt_x1+gt_x2)/2), int((gt_y1+gt_y2)/2)), 3, color, -1)

    # Add text
    textLabel = 'gt bbox'
    (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,0.5,1)
    textOrg = (gt_x1, gt_y1+5)
    cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
    cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
    cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

    # Draw positive anchors according to the y_rpn_regr
    for i in range(debug_num_pos):

        color = (100+i*(155/4), 0, 100+i*(155/4))

        idx = pos_regr[2][i*4]/4
        anchor_size = C.anchor_box_scales[int(idx/3)]
        anchor_ratio = C.anchor_box_ratios[2-int((idx+1)%3)]

        center = (pos_regr[1][i*4]*C.rpn_stride, pos_regr[0][i*4]*C.rpn_stride)
        print('Center position of positive anchor: ', center)
        cv2.circle(img, center, 3, color, -1)
        anc_w, anc_h = anchor_size*anchor_ratio[0], anchor_size*anchor_ratio[1]
        cv2.rectangle(img, (center[0]-int(anc_w/2), center[1]-int(anc_h/2)), (center[0]+int(anc_w/2), center[1]+int(anc_h/2)), color, 2)
#         cv2.putText(img, 'pos anchor bbox '+str(i+1), (center[0]-int(anc_w/2), center[1]-int(anc_h/2)-5), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

print('Green bboxes is ground-truth bbox. Others are positive anchors')
plt.figure(figsize=(8,8))
plt.grid()
plt.imshow(img)
plt.show()

# %%

vgg = vgg_base(img_shape=C.img_shape)

def train_data_generator():
    pass



# %%

## Train models.


# %%


# Draft 

df = pd.read_csv("./dataset/train-annotations-bbox.csv")

# %%


