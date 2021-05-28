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

# These configs are temporary configs

BASE_PATH = "./dataset/open_image"
ANNOTATION_PATH = "./dataset/train-annotations-bbox.csv"
record_path = os.path.join("./models/", 'model/record.csv') # Record data (used to save the losses, classification accuracy and mean average precision)
BATCH_SIZE = 32
EPOCHS = 1

train_path = ""

C.record_path = record_path
C.model_path = ""
C.img_folder = "./dataset/open_image"
C.annotation_path = "./dataset/train-annotations-bbox.csv"
C.img_extension = ".jpg"

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
        Dict[str, int]: class_mapping
    """
    found_bg = False
    all_imgs = {}
    classes_count = {}
    class_mapping = {}
    visualize = True

    df = pd.read_csv(annotation_path)
    df_new = df[['ImageID', 'XMin', 'XMax', 'YMin', 'YMax', 'ClassName']]

    for i, row in df_new.iterrows():
        [filename, xmin, xmax, ymin, ymax, class_name] = row.to_numpy()
        if class_name not in classes_count:
            classes_count[class_name] = 1
        else:
            classes_count[class_name] += 1

        if class_name not in class_mapping:
            if class_name == 'bg' and found_bg == False:
                print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                found_bg = True
            class_mapping[class_name] = len(class_mapping)

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

    # make sure the bg class is last in the list
    if found_bg:
        if class_mapping['bg'] != len(class_mapping) - 1:
            key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
            val_to_switch = class_mapping['bg']
            class_mapping['bg'] = len(class_mapping) - 1
            class_mapping[key_to_switch] = val_to_switch
    else:
        class_mapping['bg'] = len(class_mapping)

    #! HOTFIX: have to add the class bg to the class count
    # because we also have to predict the class bg
    classes_count['bg'] = 0

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

# Get train data generator which generate X, Y, image_data
data_gen_train = data_generator(train_imgs, C, get_img_output_length, mode='train')

# %%

# Explore 

X, Y, image_data, debug_img, debug_num_pos = next(data_gen_train)

print('Original image: height=%d width=%d'%(image_data['height'], image_data['width']))
print('Resized image:  height=%d width=%d C.img_min_size=%d'%(X.shape[1], X.shape[2], C.img_min_side))
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
    print('the number of bboxes:', len(image_data['bboxes']))
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
        # cv2.putText(img, 'pos anchor bbox '+str(i+1), (center[0]-int(anc_w/2), center[1]-int(anc_h/2)-5), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

print('Green bboxes is ground-truth bbox. Others are positive anchors')
plt.figure(figsize=(8,8))
plt.grid()
plt.imshow(img)
plt.show()


# %%

## Train models.

input_shape_img = (None, None, 3)

img_input = layers.Input(shape=input_shape_img)
roi_input = layers.Input(shape=(None, 4))

# define the base network (VGG here, can be Resnet50, Inception, etc)
shared_layers = vgg_base()
shared_layers_tensor = shared_layers(img_input)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios) # 9
rpn_classify_layer, rpn_regress_layer = rpn_network(num_anchors)
rpn_cls_tensor = rpn_classify_layer(shared_layers_tensor)
rpn_regr_tensor = rpn_regress_layer(shared_layers_tensor)

classifier_cls_tensor, classifier_regr_tensor = classifier_layer(shared_layers_tensor, roi_input, C.num_rois, nb_classes=len(classes_count))


model_rpn = Model(img_input, [rpn_cls_tensor, rpn_regr_tensor])
model_classifier = Model([img_input, roi_input], [classifier_cls_tensor, classifier_regr_tensor])

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], [rpn_cls_tensor, rpn_regr_tensor] + [classifier_cls_tensor, classifier_regr_tensor])

# Because the google colab can only run the session several hours one time (then you need to connect again), 
# we need to save the model and load the model to continue training
if not os.path.exists(C.model_path):
    # If this is the begin of the training, load the pre-traind base network such as vgg-16
    try:
        print('This is the first time of your training')
        # print('loading weights from {}'.format(C.base_net_weights))
        # model_rpn.load_weights(C.base_net_weights, by_name=True)
        # model_classifier.load_weights(C.base_net_weights, by_name=True)
    except:
        print('Could not load pretrained model weights. Weights can be found in the keras application folder \
            https://github.com/fchollet/keras/tree/master/keras/applications')
    # Create the record.csv file to record losses, acc and mAP
    record_df = pd.DataFrame(columns=['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls', 'loss_rpn_regr', 'loss_class_cls', 'loss_class_regr', 'curr_loss', 'elapsed_time', 'mAP'])
else:
    # If this is a continued training, load the trained model from before
    print('Continue training based on previous trained model')
    print('Loading weights from {}'.format(C.model_path))
    model_rpn.load_weights(C.model_path, by_name=True)
    model_classifier.load_weights(C.model_path, by_name=True)
    
    # Load the records
    record_df = pd.read_csv(record_path)

    r_mean_overlapping_bboxes = record_df['mean_overlapping_bboxes']
    r_class_acc = record_df['class_acc']
    r_loss_rpn_cls = record_df['loss_rpn_cls']
    r_loss_rpn_regr = record_df['loss_rpn_regr']
    r_loss_class_cls = record_df['loss_class_cls']
    r_loss_class_regr = record_df['loss_class_regr']
    r_curr_loss = record_df['curr_loss']
    r_elapsed_time = record_df['elapsed_time']
    r_mAP = record_df['mAP']

    print('Already train %dK batches'% (len(record_df)))

# %%

optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_rpn.compile(optimizer=optimizer, loss=[rpn_loss_cls(num_anchors), rpn_loss_regr(num_anchors)])
model_classifier.compile(
    optimizer=optimizer_classifier, \
    loss=[class_loss_cls, class_loss_regr(len(classes_count)-1)], 
    metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'}
)
model_all.compile(optimizer='sgd', loss='mae')

# %%


# Training setting
# total_epochs = len(record_df)
# r_epochs = len(record_df)
total_epochs = 0
r_epochs = 0

epoch_length = 1000
num_epochs = 1
iter_num = 0

total_epochs += num_epochs

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []

# if len(record_df)==0:
#     best_loss = np.Inf
# else:
#     best_loss = np.min(r_curr_loss)
# print(len(record_df))

best_loss = np.Inf

# %%

start_time = time.time()
for epoch_num in range(num_epochs):
    progbar = Progbar(epoch_length)
    print('Epoch {}/{}'.format(r_epochs + 1, total_epochs))

    r_epochs += 1
    while True:
        # try:
        if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
            mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
            rpn_accuracy_rpn_monitor = []
            # print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
            if mean_overlapping_bboxes == 0:
                print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

        # Generate X (x_img) and label Y ([y_rpn_cls, y_rpn_regr])
        X, Y, img_data, debug_img, debug_num_pos = next(data_gen_train)

        # print("Generated data!")

        # Train rpn model and get loss value [_, loss_rpn_cls, loss_rpn_regr]
        loss_rpn = model_rpn.train_on_batch(X, Y)

        # print("Trained loss rpn!")

        # Get predicted rpn from rpn model [rpn_cls, rpn_regr]
        P_rpn = model_rpn.predict_on_batch(X)

        # R: bboxes (shape=(300,4))
        # Convert rpn layer to roi bboxes
        R = rpn_to_roi(P_rpn[0], P_rpn[1], C, use_regr=True, overlap_thresh=0.7, max_boxes=300)

        # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
        # X2: bboxes that iou > C.classifier_min_overlap for all gt bboxes in 300 non_max_suppression bboxes
        # Y1: one hot code for bboxes from above => x_roi (X)
        # Y2: corresponding labels and corresponding gt bboxes
        X2, Y1, Y2, IouS = calc_iou(R, img_data, C, class_mapping)
        
        # print("calc iou success")

        # If X2 is None means there are no matching bboxes
        if X2 is None:
            rpn_accuracy_rpn_monitor.append(0)
            rpn_accuracy_for_epoch.append(0)
            continue

        # Find out the positive anchors and negative anchors
        neg_samples = np.where(Y1[0, :, -1] == 1)
        pos_samples = np.where(Y1[0, :, -1] == 0)

        if len(neg_samples) > 0:
            neg_samples = neg_samples[0]
        else:
            neg_samples = []

        if len(pos_samples) > 0:
            pos_samples = pos_samples[0]
        else:
            pos_samples = []

        rpn_accuracy_rpn_monitor.append(len(pos_samples))
        rpn_accuracy_for_epoch.append((len(pos_samples)))

        if C.num_rois > 1:
            # If number of positive anchors is larger than 4//2 = 2, randomly choose 2 pos samples
            if len(pos_samples) < C.num_rois//2:
                selected_pos_samples = pos_samples.tolist()
            else:
                selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
            
            # Randomly choose (num_rois - num_pos) neg samples
            try:
                selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
            except:
                selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()
            
            # Save all the pos and neg samples in sel_samples
            sel_samples = selected_pos_samples + selected_neg_samples
        else:
            # in the extreme case where num_rois = 1, we pick a random pos or neg sample
            selected_pos_samples = pos_samples.tolist()
            selected_neg_samples = neg_samples.tolist()
            if np.random.randint(0, 2):
                sel_samples = random.choice(neg_samples)
            else:
                sel_samples = random.choice(pos_samples)

        # training_data: [X, X2[:, sel_samples, :]]
        # labels: [Y1[:, sel_samples, :], Y2[:, sel_samples, :]]
        #  X                     => img_data resized image
        #  X2[:, sel_samples, :] => num_rois (4 in here) bboxes which contains selected neg and pos
        #  Y1[:, sel_samples, :] => one hot encode for num_rois bboxes which contains selected neg and pos
        #  Y2[:, sel_samples, :] => labels and gt bboxes for num_rois bboxes which contains selected neg and pos
        loss_class = model_classifier.train_on_batch(
            [X, X2[:, sel_samples, :]], 
            [Y1[:, sel_samples, :], Y2[:, sel_samples, :]]
        )

        losses[iter_num, 0] = loss_rpn[1]
        losses[iter_num, 1] = loss_rpn[2]

        losses[iter_num, 2] = loss_class[1]
        losses[iter_num, 3] = loss_class[2]
        losses[iter_num, 4] = loss_class[3]

        iter_num += 1

        progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                ('final_cls', np.mean(losses[:iter_num, 2])), ('final_regr', np.mean(losses[:iter_num, 3]))])

        if iter_num == epoch_length:
            loss_rpn_cls = np.mean(losses[:, 0])
            loss_rpn_regr = np.mean(losses[:, 1])
            loss_class_cls = np.mean(losses[:, 2])
            loss_class_regr = np.mean(losses[:, 3])
            class_acc = np.mean(losses[:, 4])

            mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
            rpn_accuracy_for_epoch = []

            if C.verbose:
                print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                print('Loss RPN regression: {}'.format(loss_rpn_regr))
                print('Loss Detector classifier: {}'.format(loss_class_cls))
                print('Loss Detector regression: {}'.format(loss_class_regr))
                print('Total loss: {}'.format(loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr))
                print('Elapsed time: {}'.format(time.time() - start_time))
                elapsed_time = (time.time()-start_time)/60

            curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
            iter_num = 0
            start_time = time.time()

            if curr_loss < best_loss:
                if C.verbose:
                    print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
                best_loss = curr_loss
                model_all.save_weights(C.model_path)

            new_row = {'mean_overlapping_bboxes':round(mean_overlapping_bboxes, 3), 
                    'class_acc':round(class_acc, 3), 
                    'loss_rpn_cls':round(loss_rpn_cls, 3), 
                    'loss_rpn_regr':round(loss_rpn_regr, 3), 
                    'loss_class_cls':round(loss_class_cls, 3), 
                    'loss_class_regr':round(loss_class_regr, 3), 
                    'curr_loss':round(curr_loss, 3), 
                    'elapsed_time':round(elapsed_time, 3), 
                    'mAP': 0}

            record_df = record_df.append(new_row, ignore_index=True)
            record_df.to_csv(record_path, index=0)

            break

        # except Exception as e:
        #     print(f"Exception: {e}")
        #     continue

print('Training complete, exiting.')



# %%

