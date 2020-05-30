import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from RPN_Sample.utils import generate_anchors, bbox_transform_inv, loss_cls, smoothL1, py_cpu_nms


def getAnchors(img_width=224, img_height=224, width=14, height=14):
    # region configure parameters
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


def RPN_load(file_path='..\\TrainedModels\\RPN_Prototype.h5'):
    rpn_model = load_model(
        file_path,
        custom_objects={
            'loss_cls': loss_cls,
            'smoothL1': smoothL1
        }
    )
    return rpn_model


def select_proposals(scores, proposals, AutoSelection=0.5, pre_nms_top_n=30, post_nms_top_n=10):
    if AutoSelection:
        pre_nms_top_n = int(str(proposals.shape[0])[0]) * (10 ** (len(str(proposals.shape[0])) - 1))
        post_nms_top_n = int(pre_nms_top_n * AutoSelection)
    # sort scores and only keep top 6000.
    order = scores.ravel().argsort()[::-1]
    if pre_nms_top_n > 0:
        order = order[:pre_nms_top_n]
    proposals = proposals[order, :]
    scores = scores[order]
    # apply NMS to to 6000, and then keep top 300
    keep = py_cpu_nms(np.hstack((proposals, scores)), 0.7)
    if post_nms_top_n > 0:
        keep = keep[:post_nms_top_n]
    proposals = proposals[keep, :]
    scores = scores[keep, :]
    return proposals, scores


def RPN_forward(rpn_model, feature_map, SelectBest=True, AutoSelection=0.5):
    # 利用RPN从特征图得到预测的锚框偏移系数和对应的分数
    deltas, scores = getDeviations(rpn=rpn_model, feature_map=feature_map)
    # 根据预先制定的规则生成原始锚框
    anchors = getAnchors(width=feature_map.shape[1], height=feature_map.shape[2])
    # 根据锚框和偏移量生成实际框坐标（x1, y1, x2, y2）
    proposals, scores = DA2ROI(deltas=deltas, scores=scores, all_anchors=anchors)
    if SelectBest:
        proposals, scores = select_proposals(scores=scores, proposals=proposals, AutoSelection=AutoSelection)
    return proposals, scores


def getImage(file_path="..\\ProcessedData\\Images\\airplane_002.jpg"):
    image = load_img(file_path)
    img_width = 224
    img_height = 224
    image = image.resize((int(img_width), int(img_height)))
    image = img_to_array(image)
    return image


def drawROIs(image, proposals, img_name=None, output=False):
    file_path = "..\\TestResults\\"
    image_copy = image.copy()
    for i in range(proposals.shape[0]):
        x1 = proposals[i, 0]
        y1 = proposals[i, 1]
        x2 = proposals[i, 2]
        y2 = proposals[i, 3]
        image_copy = cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
    if output:
        return image_copy
    plt.figure()
    plt.imshow(image_copy.astype("uint32"))
    if img_name:
        plt.savefig(file_path + img_name + "_" + str(proposals.shape[0]) + ".jpg")
    plt.show()


def RPN_test(image, img_name=None, preload=None):
    if preload:
        rpn_model = preload[0]
        backbone_network = preload[1]
    else:
        rpn_model = RPN_load()
        backbone_network = VGG16(include_top=True, weights="imagenet")
        backbone_network = Model(inputs=backbone_network.input, outputs=backbone_network.layers[17].output)
    # 利用骨干网络对图像特征进行提取，得到14*14*512的特征图
    feature_map = backbone_network.predict(np.expand_dims(image, axis=0) / 255)
    for AS in [1, 0.75, 0.5, 0.25]:
        proposals, scores = RPN_forward(rpn_model=rpn_model, feature_map=feature_map, AutoSelection=AS)
        drawROIs(image, proposals, img_name)
    print("Done")


def batch_test():
    file_path = "..\\ProcessedData\\Images\\"
    rpn_model = RPN_load()
    backbone_network = VGG16(include_top=True, weights="imagenet")
    backbone_network = Model(inputs=backbone_network.input, outputs=backbone_network.layers[17].output)
    for e, i in enumerate(os.listdir(file_path)):
        if i.startswith("airplane"):
            img = getImage(os.path.join(file_path, i))
            RPN_test(img, img_name=i, preload=[rpn_model, backbone_network])
