
import numpy as np
import random
import copy
import cv2

from tensorflow.keras import backend as K

from .utils import show_img, show_img_with_box, get_new_img_size, iou

def calc_rpn(C, img_data, width, height, resized_width, resized_height, img_length_calc_function, verbose=False):
    """ Copied from https://github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras/blob/master/frcnn_train_vgg.ipynb
    (Important part!) Calculate the rpn for all anchors 
        If feature map has shape 38x50=1900, there are 1900x9=17100 potential anchors
        #! TODO: Figure out the range of the bbox
    Args:
        C: config
        img_data: augmented image data, expecting x, y coordinates to have range [0, 1]
        width: original image width (e.g. 600)
        height: original image height (e.g. 800)
        resized_width: resized image width according to C.img_min_side (e.g. 300)
        resized_height: resized image height according to C.img_min_side (e.g. 400)
        img_length_calc_function: function to calculate final layer's feature map (of base model) size according to input image size

    Returns:
        y_rpn_cls: list(num_bboxes, y_is_box_valid + y_rpn_overlap)
            y_is_box_valid: 0 or 1 (0 means the box is invalid, 1 means the box is valid)
            y_rpn_overlap: 0 or 1 (0 means the box is not an object, 1 means the box is an object)
        y_rpn_regr: list(num_bboxes, 4*y_rpn_overlap + y_rpn_regr)
            y_rpn_regr: x1,y1,x2,y2 bunding boxes coordinates
        order of anchor size (s) and ratio (r): s1r1, s1r2, s1r3, (s1r...), s2r1, s2r2, ...
    """
    rpn_stride = float(C.rpn_stride) 
    anchor_sizes = C.anchor_box_scales   # 128, 256, 512
    anchor_ratios = C.anchor_box_ratios  # 1:1, 1:2*sqrt(2), 2*sqrt(2):1
    num_anchors = len(anchor_sizes) * len(anchor_ratios) # 3x3=9

    if verbose:
        print("rpn_stride: ", rpn_stride)
        print("anchor_sizes: ", anchor_sizes)
        print("anchor_ratios: ", anchor_ratios)
        print("num_anchors: ", num_anchors)

    # calculate the output map size based on the network architecture
    (output_width, output_height) = img_length_calc_function(resized_width, resized_height, downscale=C.downscale)
    if verbose:
        print(f"(output_height, output_width) = ({output_height}, {output_width})")

    n_anchratios = len(anchor_ratios)    # 3
    
    # initialise empty output objectives
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

    num_bboxes = len(img_data['bboxes'])

    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
    best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int)
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
    best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
    best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

    # get the GT box coordinates, and resize to account for image resizing
    gta = np.zeros((num_bboxes, 4))
    for bbox_num, bbox in enumerate(img_data['bboxes']):
    
        # get the GT box coordinates, and resize to account for image resizing
        gta[bbox_num, 0] = bbox['xmin'] * (resized_width / float(width)) # * float(width) # Multiply with float(height) for clear understanding
        gta[bbox_num, 1] = bbox['xmax'] * (resized_width / float(width)) # * float(width)    # because bbox ranged [0, 1], now we want it to range 
        gta[bbox_num, 2] = bbox['ymin'] * (resized_height / float(height)) # * float(height) # to the size (width or height of the image)
        gta[bbox_num, 3] = bbox['ymax'] * (resized_height / float(height)) # * float(height)
    
    #### VERIFIED TILL HERE

    if verbose:
        print("#### Building RPN from Ground truth box ####")

    # rpn ground truth
    for anchor_size_idx in range(len(anchor_sizes)):
        for anchor_ratio_idx in range(n_anchratios):
            anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
            anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]    
            
            # if verbose:
            #     print("anchor_x:", anchor_x)
            #     print("anchor_y:", anchor_y)

            for ix in range(output_width):                    
                # x-coordinates of the current anchor box   
                # ix + 0.5 originally 
                x1_anc = rpn_stride * (ix + 0.5) - anchor_x / 2
                x2_anc = rpn_stride * (ix + 0.5) + anchor_x / 2    
                
                # if verbose:
                #     print("x1_anc:", x1_anc)
                #     print("x2_anc:", x2_anc)

                # ignore boxes that go across image boundaries
                if x1_anc < 0 or x2_anc > resized_width:
                    # if verbose:
                    #     print("X: ignore boxes that go across image boundaries")
                    continue
                
                for jy in range(output_height):
                    # y-coordinates of the current anchor box
                    # jy + 0.5 originally 
                    y1_anc = rpn_stride * (jy + 0.5) - anchor_y / 2
                    y2_anc = rpn_stride * (jy + 0.5) + anchor_y / 2

                    # if verbose:
                    #     print("y1_anc:", y1_anc)
                    #     print("y2_anc:", y2_anc)

                    # ignore boxes that go across image boundaries
                    if y1_anc < 0 or y2_anc > resized_height:
                        # if verbose:
                        #     print("Y: ignore boxes that go across image boundaries")
                        continue

                    # See the width and height of groundtruth boxes and anchor boxes.
                    # if verbose:
                    #     print("gt height:", gta[bbox_num, 3] - gta[bbox_num, 2])
                    #     print("gt width:", gta[bbox_num, 1] - gta[bbox_num, 0])
                    #     print("anc height:", y2_anc - y1_anc)
                    #     print("anc width:", x2_anc - x1_anc)

                    # bbox_type indicates whether an anchor should be a target
                    # Initialize with 'negative'
                    bbox_type = 'neg'

                    # this is the best IOU for the (x,y) coord and the current anchor
                    # note that this is different from the best IOU for a GT bbox
                    best_iou_for_loc = 0.0

                    for bbox_num in range(num_bboxes):

                        # get IOU of the current GT box and the current anchor box
                        curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])
                        # calculate the regression targets if they will be needed
                        
                        if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:

                            cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
                            cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
                            cxa = (x1_anc + x2_anc) / 2.0
                            cya = (y1_anc + y2_anc) / 2.0

                            # See the width and height of groundtruth boxes and anchor boxes.
                            if verbose:
                                print(f"Feature number: (ix: {ix}, jy: {jy})")
                                print("gt xmin:", gta[bbox_num, 0])
                                print("gt ymin:", gta[bbox_num, 2])
                                print("gt xmax:", gta[bbox_num, 1])
                                print("gt ymax:", gta[bbox_num, 3])
                                print("gt height:", gta[bbox_num, 3] - gta[bbox_num, 2])
                                print("gt width:", gta[bbox_num, 1] - gta[bbox_num, 0])
                                print("current iou:", curr_iou)
                                print("xmin_anc:", x1_anc)
                                print("ymin_anc:", y1_anc)
                                print("xmax_anc:", x2_anc)
                                print("ymax_anc:", y2_anc)
                                print("anc height:", y2_anc - y1_anc)
                                print("anc width:", x2_anc - x1_anc)
                                print("cx:", cx)
                                print("cy:", cy)
                                print("cxa:", cxa)
                                print("cya:", cya)

                            # x,y are the center point of ground-truth bbox
                            # xa,ya are the center point of anchor bbox (xa=rpn_stride * (ix + 0.5); ya=rpn_stride * (iy+0.5))
                            # w,h are the width and height of ground-truth bbox
                            # wa,ha are the width and height of anchor bboxe
                            # tx = (x - xa) / wa
                            # ty = (y - ya) / ha
                            # tw = log(w / wa)
                            # th = log(h / ha)
                            tx = (cx - cxa) / (x2_anc - x1_anc)
                            ty = (cy - cya) / (y2_anc - y1_anc)
                            tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
                            th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))

                        if img_data['bboxes'][bbox_num]['class'] != 'bg':
                            
                            # all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
                            if curr_iou > best_iou_for_bbox[bbox_num]:
                                best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                                best_iou_for_bbox[bbox_num] = curr_iou
                                best_x_for_bbox[bbox_num,:] = [x1_anc, x2_anc, y1_anc, y2_anc]
                                best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th]
                                
                            # we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
                            if curr_iou > C.rpn_max_overlap:
                                bbox_type = 'pos'
                                num_anchors_for_bbox[bbox_num] += 1
                                # we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
                                if curr_iou > best_iou_for_loc:
                                    best_iou_for_loc = curr_iou
                                    best_regr = (tx, ty, tw, th)
                                    if verbose:
                                        print("Final chosen anchor (xmin, ymin, xmax, ymax):", x1_anc, y1_anc, x2_anc, y2_anc)

                            # if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
                            if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
                                # gray zone between neg and pos
                                if bbox_type != 'pos':
                                    bbox_type = 'neutral'

                    # turn on or off outputs depending on IOUs
                    if bbox_type == 'neg':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'neutral':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'pos':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
                        y_rpn_regr[jy, ix, start:start+4] = best_regr

    # we ensure that every bbox has at least one positive RPN region

    for idx in range(num_anchors_for_bbox.shape[0]):
        if num_anchors_for_bbox[idx] == 0:
            # no box with an IOU greater than zero ...
            if best_anchor_for_bbox[idx, 0] == -1:
                continue
            y_is_box_valid[
                best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
                best_anchor_for_bbox[idx,3]] = 1
            y_rpn_overlap[
                best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
                best_anchor_for_bbox[idx,3]] = 1
            start = 4 * (best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3])
            y_rpn_regr[best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], start:start+4] = best_dx_for_bbox[idx, :]

    y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

    y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

    y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

    if verbose:
        print("POS_LOC: ", pos_locs)
        print("neg_locs: ", neg_locs)

    num_pos = len(pos_locs[0])

    # one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
    # regions. We also limit it to 256 regions.
    num_regions = 256

    ### ???????????????????? ###
    if len(pos_locs[0]) > num_regions/2:
        val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
        num_pos = num_regions/2

    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

    return np.copy(y_rpn_cls), np.copy(y_rpn_regr), num_pos

def data_generator(all_img_data, C, img_length_calc_function, mode='train', verbose=False):
    """ Yield the ground-truth anchors as Y (labels)
        
    Args:
        all_img_data: list(filepath, width, height, list(bboxes))
        C: config
        img_length_calc_function: function to calculate final layer's feature map (of base model) size according to input image size
        mode: 'train' or 'test'; 'train' mode need augmentation

    Returns:
        x_img: image data after resized and scaling (smallest size = 300px)
        Y: [y_rpn_cls, y_rpn_regr]
        img_data_aug: augmented image data (original image with augmentation)
        debug_img: show image for debug
        num_pos: show number of positive anchors for debug
    """
    while True:

        for img_data in all_img_data:
            try:
                if verbose:
                    print(img_data)
                # read in image, and optionally add augmentation

                if mode == 'train':
                    img_data_aug, x_img = augment(img_data, C, augment=True)
                else:
                    img_data_aug, x_img = augment(img_data, C, augment=False)
                
                if verbose:
                    print("x_img:", "shape:", x_img.shape)
                    show_img(x_img)

                if verbose:
                    # draw ground truth bouding box
                    gt_x1, gt_x2 = img_data_aug['bboxes'][0]['xmin'], \
                        img_data_aug['bboxes'][0]['xmax']
                    gt_y1, gt_y2 = img_data_aug['bboxes'][0]['ymin'], \
                        img_data_aug['bboxes'][0]['ymax']
                    gt_x1, gt_y1, gt_x2, gt_y2 = int(gt_x1), int(gt_y1), int(gt_x2), int(gt_y2)
                    show_img_with_box(x_img, gt_x1, gt_y1, gt_x2, gt_y2)


                (width, height) = (img_data_aug['width'], img_data_aug['height'])
                (rows, cols, _) = x_img.shape

                assert cols == width
                assert rows == height

                # get image dimensions for resizing
                (resized_width, resized_height) = get_new_img_size(width, height, img_min_side=C.img_min_side)

                # resize the image so that smaller side is length = 300px
                x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
                debug_img = x_img.copy()

                if verbose:
                    print(f"Image after resize (height, width) = ({resized_height}, {resized_width})")
                    show_img(x_img)
                    num_bboxes = len(img_data['bboxes'])
                    # get the GT box coordinates, and resize to account for image resizing
                    gta = np.zeros((num_bboxes, 4))
                    for bbox_num, bbox in enumerate(img_data_aug['bboxes']):
                        # get the GT box coordinates, and resize to account for image resizing
                        gta[bbox_num, 0] = int(bbox['xmin'] * (resized_width / float(width))) # * float(width) # Multiply with float(height) for clear understanding
                        gta[bbox_num, 1] = int(bbox['xmax'] * (resized_width / float(width))) # * float(width)    # because bbox ranged [0, 1], now we want it to range 
                        gta[bbox_num, 2] = int(bbox['ymin'] * (resized_height / float(height))) # * float(height) # to the size (width or height of the image)
                        gta[bbox_num, 3] = int(bbox['ymax'] * (resized_height / float(height))) # * float(height)
                        gta = gta.astype(int)
                        print(f"bbox num {bbox_num}: Original: {bbox}. Resized: {gta[bbox_num]}")
                        print(f"resized image with bbox:")
                        show_img_with_box(x_img, gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3])

                y_rpn_cls, y_rpn_regr, num_pos = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height, img_length_calc_function, verbose=verbose)
                # try:
                #   y_rpn_cls, y_rpn_regr, num_pos = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height, img_length_calc_function)
                # except:
                #   continue

                # Zero-center by mean pixel, and preprocess image

                x_img = x_img[:,:, (2, 1, 0)]  # BGR -> RGB
                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]
                x_img /= C.img_scaling_factor

                x_img = np.transpose(x_img, (2, 0, 1))
                x_img = np.expand_dims(x_img, axis=0)

                y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= C.std_scaling

                x_img = np.transpose(x_img, (0, 2, 3, 1))
                y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
                y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

                yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug, debug_img, num_pos

            except Exception as e:
                print("Generator Error:", e)
                continue




def augment(img_data, config, augment=True):
    """ Copied and editted from https://github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras/blob/master/frcnn_train_vgg.ipynb

    Args:
        img_data ([type]): [description]
        config ([type]): [description]
        augment (bool, optional): [description]. Defaults to True.

    Returns:
        Dict: img_data_aug: dict_keys(['filepath', 'width', 'height', 'bboxes'])
        np.ndarray: img: The image. Shape(h, w, 3). Range [0, 255].
    """
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    assert 'width' in img_data
    assert 'height' in img_data

    img_data_aug = copy.deepcopy(img_data)

    img = cv2.imread(img_data_aug['filepath'])

    if augment:
        rows, cols = img.shape[:2]

        if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 1)
            for bbox in img_data_aug['bboxes']:
                x1 = bbox['xmin']
                x2 = bbox['xmax']
                bbox['xmax'] = cols - x1
                bbox['xmin'] = cols - x2

        if config.use_vertical_flips and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 0)
            for bbox in img_data_aug['bboxes']:
                y1 = bbox['ymin']
                y2 = bbox['ymax']
                bbox['ymax'] = rows - y1
                bbox['ymin'] = rows - y2

        if config.rot_90:
            angle = np.random.choice([0,90,180,270],1)[0]
            if angle == 270:
                img = np.transpose(img, (1,0,2))
                img = cv2.flip(img, 0)
            elif angle == 180:
                img = cv2.flip(img, -1)
            elif angle == 90:
                img = np.transpose(img, (1,0,2))
                img = cv2.flip(img, 1)
            elif angle == 0:
                pass

            for bbox in img_data_aug['bboxes']:
                x1 = bbox['xmin']
                x2 = bbox['xmax']
                y1 = bbox['ymin']
                y2 = bbox['ymax']
                if angle == 270:
                    bbox['xmin'] = y1
                    bbox['xmax'] = y2
                    bbox['ymin'] = cols - x2
                    bbox['ymax'] = cols - x1
                elif angle == 180:
                    bbox['xmax'] = cols - x1
                    bbox['xmin'] = cols - x2
                    bbox['ymax'] = rows - y1
                    bbox['ymin'] = rows - y2
                elif angle == 90:
                    bbox['xmin'] = rows - y2
                    bbox['xmax'] = rows - y1
                    bbox['ymin'] = x1
                    bbox['ymax'] = x2        
                elif angle == 0:
                    pass

    img_data_aug['width'] = img.shape[1]
    img_data_aug['height'] = img.shape[0]
    return img_data_aug, img