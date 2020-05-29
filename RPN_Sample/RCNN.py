import glob
import os
import traceback
import numpy as np
import tensorflow as tf
import numpy.random as npr
import tensorflow.keras.backend as K
from random import randint
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Layer, Input
from RPN_Sample.utils import generate_anchors, bbox_overlaps, bbox_transform, \
    loss_cls, smoothL1, parse_label, filter_boxes, \
    clip_boxes, py_cpu_nms, bbox_transform_inv

labels_dict = {
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20
}
nb_classes = 20


# RoI Pooling layer
class RoIPooling(Layer):
    def __init__(self, size=(7, 7)):
        self.size = size
        super(RoIPooling, self).__init__()

    def build(self, input_shape):
        self.shape = input_shape
        super(RoIPooling, self).build(input_shape)

    def call(self, inputs, **kwargs):
        ind = K.reshape(inputs[2], (-1,))
        x = tf.image.crop_and_resize(inputs[0], inputs[1], ind, self.size)
        return x

    def compute_output_shape(self, input_shape):
        a = input_shape[1][0]
        b = self.size[0]
        c = self.size[1]
        d = input_shape[0][3]
        return a, b, c, d

    def get_config(self):
        config = {'size': self.size}
        base_config = super(RoIPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def build_RCNN():
    feature_map = Input(batch_shape=(None, None, None, 512))
    rois = Input(batch_shape=(None, 4))
    ind = Input(batch_shape=(None, 1), dtype='int32')
    p1 = RoIPooling()([feature_map, rois, ind])
    flat1 = Flatten()(p1)
    fc1 = Dense(
        units=1024,
        activation="relu",
        name="fc2"
    )(flat1)
    fc1 = BatchNormalization()(fc1)
    output_deltas = Dense(
        units=4 * nb_classes,
        activation="linear",
        kernel_initializer="uniform",
        name="deltas2"
    )(fc1)
    output_scores = Dense(
        units=1 * nb_classes,
        activation="softmax",
        kernel_initializer="uniform",
        name="scores2"
    )(fc1)
    classifier = Model(inputs=[feature_map, rois, ind], outputs=[output_scores, output_deltas])
    classifier.compile(optimizer='rmsprop',
                       loss={'deltas2': smoothL1, 'scores2': 'categorical_crossentropy'})
    return classifier


def produce_batch(rpn_model, backbone_network, file_path, gt_boxes, category):
    h_w = [224, 224]
    img = load_img(file_path)
    img_width = 224
    img_height = 224
    img = img.resize((int(img_width), int(img_height)))
    # feed image to backbone network and get feature map
    img = img_to_array(img)
    img /= 255
    img = np.expand_dims(img, axis=0)
    feature_map = backbone_network.predict(img)
    height = np.shape(feature_map)[1]
    width = np.shape(feature_map)[2]
    num_feature_map = width * height
    # calculate output w, h stride
    w_stride = h_w[1] / width
    h_stride = h_w[0] / height
    # generate base anchors according output stride.
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
    # feed feature map to RPN model, get proposal labels and bounding boxes.
    res = rpn_model.predict(feature_map)
    scores = res[0]
    scores = scores.reshape(-1, 1)
    deltas = res[1]
    deltas = np.reshape(deltas, (-1, 4))
    # proposals transform to bbox values (x1, y1, x2, y2)
    proposals, remove_valid = bbox_transform_inv(all_anchors, deltas)
    if remove_valid.any():
        scores = np.delete(scores, remove_valid, axis=0)
    proposals = clip_boxes(proposals, (h_w[0], h_w[1]))
    # remove small boxes, here threshold is 40 pixel
    keep = filter_boxes(proposals, 40)
    proposals = proposals[keep, :]
    scores = scores[keep]

    # sort scores and only keep top 6000.
    pre_nms_top_n = 6000
    order = scores.ravel().argsort()[::-1]
    if pre_nms_top_n > 0:
        order = order[:pre_nms_top_n]
    proposals = proposals[order, :]
    scores = scores[order]
    # apply NMS to to 6000, and then keep top 300
    post_nms_top_n = 300
    keep = py_cpu_nms(np.hstack((proposals, scores)), 0.7)
    if post_nms_top_n > 0:
        keep = keep[:post_nms_top_n]
    proposals = proposals[keep, :]

    # add gt_boxes to proposals.
    proposals = np.vstack((proposals, gt_boxes))
    # calculate overlaps of proposal and gt_boxes
    overlaps = bbox_overlaps(proposals, gt_boxes)
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    # labels = gt_labels[gt_assignment] #?

    # sub sample
    fg_inds = np.where(max_overlaps >= FG_THRESH)[0]
    fg_rois_per_this_image = min(int(BATCH * FG_FRACTION), fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    bg_inds = np.where((max_overlaps < BG_THRESH_HI) &
                       (max_overlaps >= BG_THRESH_LO))[0]
    bg_rois_per_this_image = BATCH - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)
    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:

    # labels = labels[keep_inds]
    rois = proposals[keep_inds]
    gt_rois = gt_boxes[gt_assignment[keep_inds]]
    targets = bbox_transform(rois, gt_rois)  # input rois
    rois_num = targets.shape[0]

    batch_box = np.zeros((rois_num, nb_classes, 4))
    for i in range(rois_num):
        batch_box[i, category] = targets[i]
    batch_box = np.reshape(batch_box, (rois_num, -1))
    # get gt category
    batch_categories = np.zeros((rois_num, nb_classes, 1))

    for i in range(rois_num):
        batch_categories[i, category] = 1

    batch_categories = np.reshape(batch_categories, (rois_num, -1))

    return rois, batch_box, batch_categories, feature_map


def boss(voc_path):
    print('worker start ' + voc_path + 'ImageSets\\Main')
    data_dict = {}
    for file_name in glob.glob(voc_path + 'ImageSets\\Main\\*_train.txt'):
        print(file_name)
        with open(file_name, 'r') as f:
            basename = os.path.basename(file_name)
            category = labels_dict[basename.split('_')[0]]
            content = []
            for line in f:
                if 'extra' not in line:
                    content.append(line)
            data_dict[category] = content
    print(len(data_dict))
    return data_dict


def worker(data_dict, voc_path):
    fc_index = 0
    batch_rois = []
    batch_feature_map_index = []
    batch_categories = []
    batch_bounding_boxes = []
    img_path = voc_path + 'JPEGImages\\'
    annotation_path = voc_path + 'Annotations\\'
    backbone_network = VGG16(include_top=True, weights="imagenet")
    backbone_network = Model(inputs=backbone_network.input, outputs=backbone_network.layers[17].output)
    rpn_model = load_model('..\\TrainedModels\\RPN_Sample.h5',
                           custom_objects={'loss_cls': loss_cls, 'smoothL1': smoothL1})
    while 1:
        try:
            category = randint(1, nb_classes)
            content = data_dict[category]
            n = randint(0, len(content)-1)
            line = content[n]
            gt_boxes = parse_label(annotation_path + line.split()[0] + '.xml')
            if len(gt_boxes) == 0:
                continue
            rois, bounding_boxes, categories, feature_map = produce_batch(
                rpn_model,
                backbone_network,
                img_path + line.split()[0] + '.jpg',
                gt_boxes,
                category-1
            )
        except Exception as e:
            print(repr(e))
            print('parse label or produce batch failed: for: ' + line.split()[0])
            traceback.print_exc()
            continue
        if len(rois) <= 0:
            continue
        for i in range(len(rois)):
            batch_rois.append(rois[i])
            batch_feature_map_index.append(fc_index)
            batch_categories.append(categories[i])
            batch_bounding_boxes.append(bounding_boxes[i])
        a = feature_map
        b = np.asarray(batch_rois)
        c = np.asarray(batch_feature_map_index)
        d = np.asarray(batch_categories)
        e = np.asarray(batch_bounding_boxes)
        f = np.zeros((len(rois), a.shape[1], a.shape[2], a.shape[3]))
        f[0] = feature_map[0]
        print("Yield!")
        # yield [f, b, c], [d, e]
        batch_rois = []
        batch_feature_map_index = []
        batch_categories = []
        batch_bounding_boxes = []
        fc_index = 0


BATCH = 16
FG_FRACTION = .25
FG_THRESH = .5
BG_THRESH_HI = .5
BG_THRESH_LO = .1
VOC_path = 'C:\\BaiduNetdiskDownload\\pascalvoc\\VOCdevkit\\VOC2007\\'
gpu_list = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpu_list:
    tf.config.experimental.set_memory_growth(gpu, True)
checkpoint = ModelCheckpoint(filepath='..\\TrainedModels\\RCNN_Sample.h5', monitor='loss', verbose=1,
                             save_best_only=True)
all_data = boss(VOC_path)
worker(all_data, VOC_path)
# model = build_RCNN()
# with tf.device('/gpu:0'):
#     model.fit_generator(worker(all_data, VOC_path), steps_per_epoch=10, epochs=100, callbacks=[checkpoint])
