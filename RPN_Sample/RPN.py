import glob
import os
import traceback
import numpy as np
import numpy.random as npr
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
from RPN_Sample.utils import generate_anchors, bbox_overlaps, bbox_transform, loss_cls, smoothL1, parse_label, unmap, \
    parse_label_csv


def build_RPN():
    k = 9
    # region RPN Model
    feature_map_tile = Input(shape=(None, None, 512))
    convolution_3x3 = Conv2D(
        filters=512,
        kernel_size=(3, 3),
        padding='same',
        name="3x3"
    )(feature_map_tile)

    output_deltas = Conv2D(
        filters=4 * k,
        kernel_size=(1, 1),
        activation="linear",
        kernel_initializer="uniform",
        name="deltas1"
    )(convolution_3x3)

    output_scores = Conv2D(
        filters=1 * k,
        kernel_size=(1, 1),
        activation="sigmoid",
        kernel_initializer="uniform",
        name="scores1"
    )(convolution_3x3)

    model_rpn = Model(inputs=[feature_map_tile], outputs=[output_scores, output_deltas])
    model_rpn.compile(optimizer='adam', loss={'scores1': loss_cls, 'deltas1': smoothL1})
    return model_rpn
    # endregion


def produce_batch(pretrained_model, file_path, gt_boxes):
    bg_fg_frac = 2
    k = 9

    # region Get feature map from backbone network (VGG16)
    img = load_img(file_path)
    img_width = 224
    img_height = 224
    img = img.resize((int(img_width), int(img_height)))
    # feed image to pretrained model and get feature map
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    feature_map = pretrained_model.predict(img)
    height = np.shape(feature_map)[1]
    width = np.shape(feature_map)[2]
    num_feature_map = width * height
    # calculate output w, h stride
    w_stride = img_width / width
    h_stride = img_height / height
    # generate base anchors according output stride.
    # endregion

    # base anchors are 9 anchors wrt a tile (0,0,w_stride-1,h_stride-1)
    base_anchors = generate_anchors(w_stride, h_stride)
    # slice tiles according to image size and stride.
    # each 1x1x1532 feature map is mapping to a tile.
    shift_x = np.arange(0, width) * w_stride
    shift_y = np.arange(0, height) * h_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(),
                        shift_y.ravel())).transpose()
    # apply base anchors to all tiles, to have a num_feature_map*9 anchors.
    all_anchors = (base_anchors.reshape((1, 9, 4)) +
                   shifts.reshape((1, num_feature_map, 4)).transpose((1, 0, 2)))
    total_anchors = num_feature_map * 9
    all_anchors = all_anchors.reshape((total_anchors, 4))
    # only keep anchors inside image+borader.
    border = 0
    inds_inside = np.where(
        (all_anchors[:, 0] >= -border) &
        (all_anchors[:, 1] >= -border) &
        (all_anchors[:, 2] < img_width + border) &  # width
        (all_anchors[:, 3] < img_height + border)  # height
    )[0]
    anchors = all_anchors[inds_inside]
    # calculate overlaps each anchors to each gt boxes,
    # a matrix with shape [len(anchors) x len(gt_boxes)]
    overlaps = bbox_overlaps(anchors, gt_boxes)
    # find the gt box with biggest overlap to each anchors,
    # and the overlap ratio. result (len(anchors),)
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    # find the anchor with biggest overlap to each gt boxes,
    # and the overlap ratio. result (len(gt_boxes),)
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
    # labels, 1=fg/0=bg/-1=ignore
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)
    # set positive label, define in Paper3.1.2:
    # We assign a positive label to two kinds of anchors: (i) the
    # anchor/anchors with the highest Intersection-overUnion
    # (IoU) overlap with a ground-truth box, or (ii) an
    # anchor that has an IoU overlap higher than 0.7 with any gt boxes
    labels[gt_argmax_overlaps] = 1
    labels[max_overlaps >= .7] = 1
    # set negative labels
    labels[max_overlaps <= .3] = 0
    # subsample positive labels if we have too many
    #     num_fg = int(RPN_FG_FRACTION * RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]
    #     if len(fg_inds) > num_fg:
    #         disable_inds = npr.choice(
    #             fg_inds, size=(len(fg_inds) - num_fg), replace=False)
    #         labels[disable_inds] = -1
    # subsample negative labels if we have too many
    num_bg = int(len(fg_inds) * bg_fg_frac)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1
    #
    batch_inds = inds_inside[labels != -1]
    batch_inds = (batch_inds / k).astype(np.int)
    full_labels = unmap(labels, total_anchors, inds_inside, fill=-1)
    batch_label_targets = full_labels.reshape(-1, 1, 1, 1 * k)[batch_inds]

    pos_anchors = all_anchors[inds_inside[labels == 1]]
    bbox_targets = bbox_transform(pos_anchors, gt_boxes[argmax_overlaps, :][labels == 1])
    bbox_targets = unmap(bbox_targets, total_anchors, inds_inside[labels == 1], fill=0)
    batch_bbox_targets = bbox_targets.reshape(-1, 1, 1, 4 * k)[batch_inds]
    padded_fcmap = np.pad(feature_map, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant')
    padded_fcmap = np.squeeze(padded_fcmap)
    batch_tiles = []
    for ind in batch_inds:
        x = ind % width
        y = int(ind / width)
        fc_3x3 = padded_fcmap[y:y + 3, x:x + 3, :]
        batch_tiles.append(fc_3x3)
    return np.asarray(batch_tiles), batch_label_targets.tolist(), batch_bbox_targets.tolist()


def input_generator():
    voc_path = 'C:\\BaiduNetdiskDownload\\pascalvoc\\VOCdevkit\\VOC2007\\'
    img_path = voc_path + 'JPEGImages\\'
    annotation_path = voc_path + 'Annotations\\'
    batch_size = 64
    batch_tiles = []
    batch_labels = []
    batch_bounding_boxes = []
    backbone_network = VGG16(include_top=True, weights="imagenet")
    backbone_network = Model(inputs=backbone_network.input, outputs=backbone_network.layers[17].output)
    while 1:
        for file_name in glob.glob(voc_path + 'ImageSets\\Main\\*_train.txt'):
            with open(file_name, 'r') as f:
                for line in f:
                    if 'extra' not in line:
                        try:
                            gt_boxes = parse_label(annotation_path + line.split()[0] + '.xml')
                            if len(gt_boxes) == 0:
                                continue
                            tiles, labels, bounding_boxes = produce_batch(
                                backbone_network,
                                img_path + line.split()[0] + '.jpg',
                                gt_boxes
                            )
                        except Exception:
                            print('parse label or produce batch failed: for: ' + line.split()[0])
                            traceback.print_exc()
                            continue
                        for i in range(len(tiles)):
                            batch_tiles.append(tiles[i])
                            batch_labels.append(labels[i])
                            batch_bounding_boxes.append(bounding_boxes[i])
                            if len(batch_tiles) == batch_size:
                                a = np.asarray(batch_tiles)
                                b = np.asarray(batch_labels)
                                c = np.asarray(batch_bounding_boxes)
                                if not a.any() or not b.any() or not c.any():
                                    print("empty array found.")
                                    batch_tiles = []
                                    batch_labels = []
                                    batch_bounding_boxes = []
                                    continue
                                yield a, [b, c]
                                batch_tiles = []
                                batch_labels = []
                                batch_bounding_boxes = []


def input_gen_airplane():
    annotation = "..\\ProcessedData\\Airplanes_Annotations"
    images_path = "..\\ProcessedData\\Images"
    batch_size = 16
    batch_tiles = []
    batch_labels = []
    batch_bounding_boxes = []
    backbone_network = VGG16(include_top=True, weights="imagenet")
    backbone_network = Model(inputs=backbone_network.input, outputs=backbone_network.layers[17].output)
    while 1:
        for e, i in enumerate(os.listdir(annotation)):
            if i.startswith("airplane"):
                try:
                    gt_boxes = parse_label_csv(os.path.join(annotation, i))
                    image_filename = i.split(".")[0] + ".jpg"
                    tiles, labels, bounding_boxes = produce_batch(
                        backbone_network,
                        os.path.join(images_path, image_filename),
                        gt_boxes
                    )
                except Exception as e:
                    print("file: " + os.path.join(annotation, i) + " could not be parsed!")
                    print(repr(e))
                    continue
                for j in range(len(tiles)):
                    batch_tiles.append(tiles[j])
                    batch_labels.append(labels[j])
                    batch_bounding_boxes.append(bounding_boxes[j])
                    if len(batch_tiles) == batch_size:
                        a = np.asarray(batch_tiles)
                        b = np.asarray(batch_labels)
                        c = np.asarray(batch_bounding_boxes)
                        if not a.any() or not b.any() or not c.any():
                            print("empty array found.")
                            batch_tiles = []
                            batch_labels = []
                            batch_bounding_boxes = []
                            continue
                        yield a, [b, c]
                        batch_tiles = []
                        batch_labels = []
                        batch_bounding_boxes = []


def train_RPN(BiClassify=False):
    if BiClassify:
        file_path = '..\\TrainedModels\\RPN_Prototype.h5'
    else:
        file_path = '..\\TrainedModels\\RPN_Sample.h5'
    checkpoint = ModelCheckpoint(filepath=file_path,
                                 monitor='loss',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto',
                                 save_freq='epoch'
                                 )
    if os.path.exists(file_path):
        model_rpn = tf.keras.models.load_model(
            file_path,
            custom_objects={
                'loss_cls': loss_cls,
                'smoothL1': smoothL1
            }
        )
    else:
        model_rpn = build_RPN()
        model_rpn.save(file_path)
    with tf.device('/gpu:0'):
        if BiClassify:
            model_rpn.fit_generator(input_gen_airplane(), steps_per_epoch=100, epochs=800, callbacks=[checkpoint])
        else:
            model_rpn.fit_generator(input_generator(), steps_per_epoch=100, epochs=800, callbacks=[checkpoint])


gpu_list = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpu_list:
    tf.config.experimental.set_memory_growth(gpu, True)

train_RPN(BiClassify=True)
