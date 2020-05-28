import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

from RPN_Sample.utils import generate_anchors, bbox_transform_inv, loss_cls, smoothL1, py_cpu_nms, parse_label_csv, \
    bbox_overlaps, bbox_transform


def getAnchors():
    # region configure parameters
    img_width = 224
    img_height = 224
    width = 14
    height = 14
    num_feature_map = width * height
    w_stride = img_width / width
    h_stride = img_height / height
    # endregion

    base_anchors = generate_anchors(w_stride, h_stride)
    shift_x = np.arange(0, width) * w_stride
    shift_y = np.arange(0, height) * h_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack(
        (
            shift_x.ravel(),
            shift_y.ravel(),
            shift_x.ravel(),
            shift_y.ravel()
        )
    ).transpose()
    # apply base anchors to all tiles, to have a num_feature_map*9 anchors.
    all_anchors = (
            base_anchors.reshape((1, 9, 4)) + shifts.reshape((1, num_feature_map, 4)).transpose((1, 0, 2))
    )
    total_anchors = num_feature_map * 9
    all_anchors = all_anchors.reshape((total_anchors, 4))
    return all_anchors


def getDeviations(rpn, feature_map):
    res = rpn.predict(feature_map)
    scores = res[0]
    scores = scores.reshape(-1, 1)
    deltas = res[1]
    deltas = np.reshape(deltas, (-1, 4))
    # proposals transform to bbox values (x1, y1, x2, y2)
    return deltas, scores


def DA2ROI(deltas, all_anchors, scores):
    proposals, remove_valid = bbox_transform_inv(all_anchors, deltas)
    if remove_valid.any():
        scores = np.delete(scores, remove_valid, axis=0)
    return proposals, scores


def select_proposals(scores, proposals):
    # sort scores and only keep top 6000.
    pre_nms_top_n = 30
    order = scores.ravel().argsort()[::-1]
    if pre_nms_top_n > 0:
        order = order[:pre_nms_top_n]
    proposals = proposals[order, :]
    scores = scores[order]
    # apply NMS to to 6000, and then keep top 300
    post_nms_top_n = 10
    keep = py_cpu_nms(np.hstack((proposals, scores)), 0.7)
    if post_nms_top_n > 0:
        keep = keep[:post_nms_top_n]
    proposals = proposals[keep, :]
    scores = scores[keep, :]
    return proposals, scores


def calculate_overlap(proposals, gt_boxes):
    proposals = np.vstack((proposals, gt_boxes))
    overlaps = bbox_overlaps(proposals, gt_boxes)
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    return gt_assignment, max_overlaps


def sub_sample(max_overlaps):
    batch = 2
    fg_fraction = .25
    fg_thresh = .5
    bg_thresh_hi = .5
    bg_thresh_lo = .1
    # sub sample
    fg_inds = np.where(max_overlaps >= fg_thresh)[0]
    fg_rois_per_this_image = min(int(batch * fg_fraction), fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = np.random.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    bg_inds = np.where((max_overlaps < bg_thresh_hi) &
                       (max_overlaps >= bg_thresh_lo))[0]
    bg_rois_per_this_image = batch - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = np.random.choice(bg_inds, size=bg_rois_per_this_image, replace=False)
    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    return keep_inds


def RPN_forward(image, SelectBest=True):
    rpn_model = load_model(
        '..\\TrainedModels\\RPN_Prototype.h5',
        custom_objects={
            'loss_cls': loss_cls,
            'smoothL1': smoothL1
        }
    )
    backbone_network = VGG16(include_top=True, weights="imagenet")
    backbone_network = Model(inputs=backbone_network.input, outputs=backbone_network.layers[17].output)
    # 利用骨干网络对图像特征进行提取，得到14*14*512的特征图
    feature_map = backbone_network.predict(image)
    # 利用RPN从特征图得到预测的锚框偏移系数和对应的分数
    deltas, scores = getDeviations(rpn=rpn_model, feature_map=feature_map)
    # 根据预先制定的规则生成原始锚框
    anchors = getAnchors()
    # 根据锚框和偏移量生成实际框坐标（x1, y1, x2, y2）
    proposals, scores = DA2ROI(deltas=deltas, scores=scores, all_anchors=anchors)
    if SelectBest:
        proposals, scores = select_proposals(scores=scores, proposals=proposals)
    return proposals, scores


def make_batch_for_training(proposals):
    gt_boxes = parse_label_csv("..\\ProcessedData\\Airplanes_Annotations\\airplane_002.csv")
    gt_assignment, max_overlaps = calculate_overlap(proposals=proposals, gt_boxes=gt_boxes)
    keep_indices = sub_sample(max_overlaps=max_overlaps)
    rois = proposals[keep_indices]
    gt_rois = gt_boxes[gt_assignment[keep_indices]]
    targets = bbox_transform(rois, gt_rois)  # input rois


def getImage(file_path="..\\ProcessedData\\Images\\airplane_002.jpg"):
    image = load_img(file_path)
    img_width = 224
    img_height = 224
    image = image.resize((int(img_width), int(img_height)))
    image = img_to_array(image)
    image /= 255
    image = np.expand_dims(image, axis=0)
    return image


gpu_list = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpu_list:
    tf.config.experimental.set_memory_growth(gpu, True)
img = getImage()
P, S = RPN_forward(img, SelectBest=True)
img = img.reshape((224, 224, 3))
for i in range(P.shape[0]):
    x1 = P[i, 0]
    y1 = P[i, 1]
    x2 = P[i, 2]
    y2 = P[i, 3]
    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
plt.imshow(img)
plt.show()
print("Done")
