import sys
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

BBOX_XFORM_CLIP = np.log(1000. / 16.)


def get_iou(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def Activate_GPU():
    gpu_list = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpu_list:
        print(gpu)
        tf.config.experimental.set_memory_growth(gpu, True)


def RPN_load(file_path='..\\TrainedModels\\RPN_Prototype.h5'):
    rpn_model = load_model(
        file_path,
        custom_objects={
            'loss_cls': loss_cls,
            'smoothL1': smoothL1
        }
    )
    return rpn_model


def getAnchors(anchor_scale=np.asarray([3, 6, 12]), img_width=224, img_height=224, width=14, height=14):
    # region configure parameters
    num_feature_map = width * height
    w_stride = img_width / width
    h_stride = img_height / height
    # endregion
    base_anchors = generate_anchors(w_stride, h_stride, scales=anchor_scale)
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
    file_path = "TestResults\\"
    image_copy = image.copy()
    if len(image_copy.shape) > 3:
        image_copy = np.squeeze(image_copy)
    if image_copy.max() <= 1:
        image_copy *= 255
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


def parse_label_csv(csv_file):
    df = pd.read_csv(csv_file)
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    scale = 224 / 256
    for row in df.iterrows():
        x1 = int(row[1][0].split(" ")[0])
        y1 = int(row[1][0].split(" ")[1])
        x2 = int(row[1][0].split(" ")[2])
        y2 = int(row[1][0].split(" ")[3])
        xmin.append(x1 * scale)
        ymin.append(y1 * scale)
        xmax.append(x2 * scale)
        ymax.append(y2 * scale)
    gt_boxes = [list(box) for box in zip(xmin, ymin, xmax, ymax)]
    return np.asarray(gt_boxes, np.float)


def parse_label(xml_file):
    try:
        tree = ET.parse(xml_file)
    except Exception:
        print('Failed to parse: ' + xml_file, file=sys.stderr)
        return None
    root = tree.getroot()
    w_scale = 1
    h_scale = 1
    for x in root.iter('width'):
        w_scale = 224 / int(float(x.text))
    for x in root.iter('height'):
        h_scale = 224 / int(float(x.text))
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for x in root.iter('xmin'):
        xmin.append(int(float(x.text)) * w_scale)
    for x in root.iter('ymin'):
        ymin.append(int(float(x.text)) * h_scale)
    for x in root.iter('xmax'):
        xmax.append(int(float(x.text)) * w_scale)
    for x in root.iter('ymax'):
        ymax.append(int(float(x.text)) * h_scale)
    gt_boxes = [list(box) for box in zip(xmin, ymin, xmax, ymax)]
    return np.asarray(gt_boxes, np.float)


def loss_cls(y_true, y_pred):
    condition = K.not_equal(y_true, -1)
    indices = tf.where(condition)
    target = tf.gather_nd(y_true, indices)
    output = tf.gather_nd(y_pred, indices)
    loss = K.binary_crossentropy(target, output)
    return K.mean(loss)


def smoothL1(y_true, y_pred):
    nd = tf.where(K.not_equal(y_true, 0))
    y_true = tf.gather_nd(y_true, nd)
    y_pred = tf.gather_nd(y_pred, nd)
    h = tf.keras.losses.Huber()
    x = h(y_true, y_pred)
    #     x   = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return x


def draw_anchors(img_path, anchors, pad_size=50):
    im = Image.open(img_path)
    w, h = im.size
    a4im = Image.new('RGB',
                     (w + 2 * pad_size, h + 2 * pad_size),  # A4 at 72dpi
                     (255, 255, 255))  # White
    a4im.paste(im, (pad_size, pad_size))  # Not centered, top-left corner
    for a in anchors:
        a = (a + pad_size).astype(int).tolist()
        draw = ImageDraw.Draw(a4im)
        draw.rectangle(a, outline=(255, 0, 0), fill=None)
    return a4im


def generate_anchors(base_width=16, base_height=16, ratios=[0.5, 1, 2],
                     scales=np.asarray([3, 6, 12])):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, w_stride-1, h_stride-1) window.
    """
    # if scales is None:

    base_anchor = np.array([1, 1, base_width, base_height]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack(
        [_scale_enum(ratio_anchors[i, :], scales) for i in range(ratio_anchors.shape[0])]
    )
    return anchors


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.stack((targets_dx, targets_dy, targets_dw, targets_dh))

    targets = np.transpose(targets)

    return targets


def bbox_transform_inv(boxes, deltas):
    remove_invalid = None
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]
    dw = np.minimum(dw, BBOX_XFORM_CLIP)
    dh = np.minimum(dh, BBOX_XFORM_CLIP)
    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    larger_x1 = (pred_boxes[:, 0::4] + 20 > pred_boxes[:, 2::4])
    larger_y1 = (pred_boxes[:, 1::4] + 20 > pred_boxes[:, 3::4])
    negative_x1 = (pred_boxes[:, 0::4] < 0)
    oversize_x1 = (pred_boxes[:, 0::4] > 224)
    negative_y1 = (pred_boxes[:, 1::4] < 0)
    oversize_y1 = (pred_boxes[:, 1::4] > 224)
    negative_x2 = (pred_boxes[:, 2::4] < 0)
    oversize_x2 = (pred_boxes[:, 2::4] > 224)
    negative_y2 = (pred_boxes[:, 3::4] < 0)
    oversize_y2 = (pred_boxes[:, 3::4] > 224)

    bad_all = np.concatenate(
        (
            np.where(larger_x1)[0],
            np.where(larger_y1)[0],
            np.where(negative_x1)[0],
            np.where(oversize_x1)[0],
            np.where(negative_y1)[0],
            np.where(oversize_y1)[0],
            np.where(negative_x2)[0],
            np.where(oversize_x2)[0],
            np.where(negative_y2)[0],
            np.where(oversize_y2)[0]
        ),
        axis=0
    )
    bad_all = np.unique(bad_all)
    pred_boxes = np.delete(pred_boxes, bad_all, axis=0)
    remove_invalid = bad_all

    return pred_boxes, remove_invalid


def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    boxes = boxes.astype(int)
    N = boxes.shape[0]
    K = query_boxes.shape[0]

    overlaps = np.zeros((N, K), dtype=np.float)

    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1)
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1)

                if ih > 0:
                    ua = float((boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1) + box_area - iw * ih)
                    overlaps[n, k] = iw * ih / ua

    return overlaps


def unmap(data, count, indices, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[indices] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[indices, :] = data
    return ret
