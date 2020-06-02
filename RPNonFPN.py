import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm import tqdm

from FPN import FPN, FPN_BN_Interface
from RPN import RPN_build, data_loader
from utils import loss_cls, smoothL1, getAnchors, bbox_overlaps, unmap, bbox_transform, \
    parse_label_csv, Activate_GPU, DA2ROI, drawROIs, get_iou, select_proposals


def getImage(file_path):
    img = load_img(file_path)
    img_width = 224
    img_height = 224
    img = img.resize((int(img_width), int(img_height)))
    img = img_to_array(img)
    img /= 255
    img = np.expand_dims(img, axis=0)
    return img


def produce_batch(anchor_scale, gt_boxes, fm_size):
    bg_fg_fraction = 2
    k = 9
    total_amount_of_anchors = k * fm_size * fm_size
    img_width = 224
    img_height = 224
    all_anchors = getAnchors(anchor_scale=anchor_scale, width=fm_size, height=fm_size)
    border = 0
    indices_inside = np.where(
        (all_anchors[:, 0] >= -border) &
        (all_anchors[:, 1] >= -border) &
        (all_anchors[:, 2] < img_width + border) &  # width
        (all_anchors[:, 3] < img_height + border)  # height
    )[0]
    anchors = all_anchors[indices_inside]

    # region Calculate label for each box
    overlaps = bbox_overlaps(anchors, gt_boxes)
    # calculate overlaps each anchors to each gt boxes,
    # a matrix with shape [len(anchors) x len(gt_boxes)]
    argmax_overlaps = overlaps.argmax(axis=1)
    # find the gt box with biggest overlap to each anchors,
    # and the overlap ratio. result (len(anchors),)
    max_overlaps = overlaps[np.arange(len(indices_inside)), argmax_overlaps]
    # find the anchor with biggest overlap to each gt boxes,
    # and the overlap ratio. result (len(gt_boxes),)
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
    # labels, 1=fg/0=bg/-1=ignore
    labels = np.empty((len(indices_inside),), dtype=np.float32)
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
    # endregion

    # region Subsample negative labels if there are too many
    fg_indices = np.where(labels == 1)[0]
    # calculate numbers of background batches need via BG_FG_FRACTION
    num_bg = int(len(fg_indices) * bg_fg_fraction)
    bg_indices = np.where(labels == 0)[0]
    if len(bg_indices) > num_bg:
        disable_indices = np.random.choice(
            bg_indices, size=(len(bg_indices) - num_bg), replace=False)
        labels[disable_indices] = -1
    # endregion

    # region Make batches for labels
    full_labels = unmap(labels, total_amount_of_anchors, indices_inside, fill=-1)
    batch_label_targets = full_labels.reshape(-1, fm_size, fm_size, 1 * k)
    # endregion

    # region Make batches for bounding boxes
    pos_anchors = all_anchors[indices_inside[labels == 1]]
    bbox_targets = bbox_transform(pos_anchors, gt_boxes[argmax_overlaps, :][labels == 1])
    bbox_targets = unmap(bbox_targets, total_amount_of_anchors, indices_inside[labels == 1], fill=0)
    batch_bbox_targets = bbox_targets.reshape(-1, fm_size, fm_size, 4 * k)
    # endregion

    return batch_label_targets, batch_bbox_targets


def input_gen_airplane():
    annotation = "ProcessedData\\Airplanes_Annotations"
    images_path = "ProcessedData\\Images"
    anchor_scale = np.asarray([3, 6, 12])
    batch_size = 32
    batch_images = []
    batch_labels_1 = []
    batch_bounding_boxes_1 = []
    batch_labels_2 = []
    batch_bounding_boxes_2 = []
    batch_labels_3 = []
    batch_bounding_boxes_3 = []
    while 1:
        for e, i in enumerate(os.listdir(annotation)):
            if i.startswith("airplane"):
                try:
                    # print(i)
                    gt_boxes = parse_label_csv(os.path.join(annotation, i))
                    image_filename = i.split(".")[0] + ".jpg"
                    image_filename = os.path.join(images_path, image_filename)
                    img = getImage(image_filename)
                    labels_1, bounding_boxes_1 = produce_batch(
                        anchor_scale=anchor_scale,
                        gt_boxes=gt_boxes,
                        fm_size=28
                    )
                    labels_2, bounding_boxes_2 = produce_batch(
                        anchor_scale=anchor_scale,
                        gt_boxes=gt_boxes,
                        fm_size=14
                    )
                    labels_3, bounding_boxes_3 = produce_batch(
                        anchor_scale=anchor_scale,
                        gt_boxes=gt_boxes,
                        fm_size=7
                    )
                except Exception as e:
                    # print("file: " + os.path.join(annotation, i) + " could not be parsed!")
                    # print(repr(e))
                    continue
                batch_images.append(np.squeeze(img))
                batch_labels_1.append(np.squeeze(labels_1))
                batch_bounding_boxes_1.append(np.squeeze(bounding_boxes_1))
                batch_labels_2.append(np.squeeze(labels_2))
                batch_bounding_boxes_2.append(np.squeeze(bounding_boxes_2))
                batch_labels_3.append(np.squeeze(labels_3))
                batch_bounding_boxes_3.append(np.squeeze(bounding_boxes_3))
                if len(batch_images) == batch_size:
                    a = np.asarray(batch_images)
                    b = np.asarray(batch_labels_1)
                    c = np.asarray(batch_bounding_boxes_1)
                    d = np.asarray(batch_labels_2)
                    e = np.asarray(batch_bounding_boxes_2)
                    f = np.asarray(batch_labels_3)
                    g = np.asarray(batch_bounding_boxes_3)
                    if not a.any() or not b.any() or not c.any() or not d.any() or not e.any() or not f.any() or not g.any():
                        # print("empty array found.")
                        batch_images = []
                        batch_labels_1 = []
                        batch_bounding_boxes_1 = []
                        batch_labels_2 = []
                        batch_bounding_boxes_2 = []
                        batch_labels_3 = []
                        batch_bounding_boxes_3 = []
                        continue
                    yield a, [b, c, d, e, f, g]
                    batch_images = []
                    batch_labels_1 = []
                    batch_bounding_boxes_1 = []
                    batch_labels_2 = []
                    batch_bounding_boxes_2 = []
                    batch_labels_3 = []
                    batch_bounding_boxes_3 = []


def FPN_RPN_build():
    model_cnn = tf.keras.applications.VGG16(weights='imagenet', include_top=True)
    model_cnn.trainable = False
    fpn = FPN()
    fpn_result = FPN_BN_Interface(fpn=fpn, backbone=model_cnn)
    rpn_outs = []
    for i in range(len(fpn_result)):
        rpn_out = RPN_build(i + 1, fpn_result[i])
        rpn_outs += rpn_out[0]
    model_fpn_rpn = Model(inputs=model_cnn.input, outputs=rpn_outs)
    model_fpn_rpn.compile(
        optimizer='adam',
        loss={
            's1': loss_cls,
            'd1': smoothL1,
            's2': loss_cls,
            'd2': smoothL1,
            's3': loss_cls,
            'd3': smoothL1
        }
    )
    model_fpn_rpn.summary()
    model_fpn_rpn.save("TrainedModels\\FPN_RPN.h5py")
    return model_fpn_rpn


def FPN_RPN_load():
    model_fpn_rpn = tf.keras.models.load_model(
        "TrainedModels\\FPN_RPN.h5py",
        custom_objects={
            'loss_cls': loss_cls,
            'smoothL1': smoothL1
        }
    )
    return model_fpn_rpn


def FPN_RPN_train(NewModel=False):
    file_path = "TrainedModels\\FPN_RPN.h5py"
    checkpoint = ModelCheckpoint(filepath=file_path,
                                 monitor='loss',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto',
                                 save_freq='epoch'
                                 )
    if (not os.path.exists(file_path)) or NewModel:
        model = FPN_RPN_build()
        model.save(file_path)
    else:
        model = tf.keras.models.load_model(
            file_path,
            custom_objects={
                'loss_cls': loss_cls,
                'smoothL1': smoothL1
            }
        )

    with tf.device('/gpu:0'):
        model.fit_generator(input_gen_airplane(), steps_per_epoch=100, epochs=800, callbacks=[checkpoint])


def FPN_RPN_forward(input_image, fpn_rpn):
    s1, p1, s2, p2, s3, p3 = fpn_rpn.predict(input_image)
    p1 = p1.reshape((-1, 4))
    s1 = s1.reshape((-1, 1))
    p2 = p2.reshape((-1, 4))
    s2 = s2.reshape((-1, 1))
    p3 = p3.reshape((-1, 4))
    s3 = s3.reshape((-1, 1))
    all_anchors_1 = getAnchors(width=28, height=28)
    all_anchors_2 = getAnchors(width=14, height=14)
    all_anchors_3 = getAnchors(width=7, height=7)
    results_1 = DA2ROI(p1, all_anchors_1, s1)
    results_2 = DA2ROI(p2, all_anchors_2, s2)
    results_3 = DA2ROI(p3, all_anchors_3, s3)
    return results_1, results_2, results_3


def FPN_RPN_test():
    model = FPN_RPN_load()
    image_path = "ProcessedData\\Images"
    annotation = "ProcessedData\\Airplanes_Annotations"
    for e, file_path in enumerate(os.listdir(image_path)):
        if not file_path.startswith('4'):
            print("Not a test data, skip it.")
            continue
        count = 0
        image_out = getImage(os.path.join(image_path, file_path))
        r1, r2, r3 = FPN_RPN_forward(image_out, model)
        image_out = np.squeeze(image_out)
        r1 = select_proposals(r1[1], r1[0], AutoSelection=0.25)
        r2 = select_proposals(r2[1], r2[0], AutoSelection=0.25)
        r3 = select_proposals(r3[1], r3[0], AutoSelection=0.25)
        file_path = file_path.split(".")[0] + ".csv"
        gt_boxes = parse_label_csv(os.path.join(annotation, file_path))
        if not gt_boxes.any():
            print("No target inside.")
            continue
        gt_values = []
        for j in range(gt_boxes.shape[0]):
            x1 = int(gt_boxes[j, 0])
            y1 = int(gt_boxes[j, 1])
            x2 = int(gt_boxes[j, 2])
            y2 = int(gt_boxes[j, 3])
            if x1 >= x2 or y1 >= y2:
                print("zero area error!")
                continue
            gt_values.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
        for proposals in [r1[0], r2[0], r3[0]]:
            for roi in range(proposals.shape[0]):
                x1 = int(proposals[roi, 0])
                y1 = int(proposals[roi, 1])
                x2 = int(proposals[roi, 2])
                y2 = int(proposals[roi, 3])
                if x1 >= x2 or y1 >= y2:
                    print("zero area error!")
                    continue
                iou_list = []
                for gt_val in gt_values:
                    temp = get_iou(gt_val, {"x1": x1, "x2": x2, "y1": y1, "y2": y2})
                    iou_list.append(temp)
                if max(iou_list) > 0.7:
                    count += 1
                    image_out = cv2.rectangle(image_out, (x1, y1), (x2, y2), (0, 1, 0), 1, cv2.LINE_AA)
                # else:
                #     image_out = cv2.rectangle(image_out, (x1, y1), (x2, y2), (1, 0, 0), 1, cv2.LINE_AA)
        plt.figure()
        plt.imshow(image_out)
        plt.savefig(
            "TestResults\\第" +
            str(file_path.split(".csv")[0]) +
            "次测试，命中比例：" +
            str(int(100 * count / (r1[0].shape[0] + r2[0].shape[0] + r3[0].shape[0]))) + "%.jpg")
        plt.close()


def test_model_od(CheckTarget=False, CheckNeg=False):
    fpn_rpn = FPN_RPN_load()
    model_cnn = tf.keras.models.load_model("TrainedModels\\RCNN.h5")
    image_path = "ProcessedData\\Images"
    annotation = "ProcessedData\\Airplanes_Annotations"
    for e, file_path in enumerate(os.listdir(image_path)):
        if not file_path.startswith('4'):
            print("Not a test data, skip it.")
            continue
        print(file_path)
        image_out = getImage(os.path.join(image_path, file_path))
        r1, r2, r3 = FPN_RPN_forward(image_out, fpn_rpn)
        image_out = np.squeeze(image_out)
        image_copy = image_out.copy()
        r1 = select_proposals(r1[1], r1[0], AutoSelection=0.25)
        r2 = select_proposals(r2[1], r2[0], AutoSelection=0.25)
        r3 = select_proposals(r3[1], r3[0], AutoSelection=0.25)
        file_path = file_path.split(".")[0] + ".csv"
        gt_boxes = parse_label_csv(os.path.join(annotation, file_path))
        if not gt_boxes.any():
            print("No target inside.")
            continue
        gt_values = []
        for j in range(gt_boxes.shape[0]):
            x1 = int(gt_boxes[j, 0])
            y1 = int(gt_boxes[j, 1])
            x2 = int(gt_boxes[j, 2])
            y2 = int(gt_boxes[j, 3])
            if x1 >= x2 or y1 >= y2:
                print("zero area error!")
                continue
            gt_values.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
        for proposals in [r1[0], r2[0], r3[0]]:
            for roi in tqdm(range(proposals.shape[0])):
                x1 = int(proposals[roi, 0])
                y1 = int(proposals[roi, 1])
                x2 = int(proposals[roi, 2])
                y2 = int(proposals[roi, 3])
                iou_list = []
                for gt_val in gt_values:
                    temp = get_iou(gt_val, {"x1": x1, "x2": x2, "y1": y1, "y2": y2})
                    iou_list.append(temp)
                if max(iou_list) > 0.5:
                    image_out = cv2.rectangle(image_out, (x1, y1), (x2, y2), (1, 0, 0), 1, cv2.LINE_AA)
                target_image = image_copy[y1:y2, x1:x2]
                target_image = cv2.resize(target_image, (224, 224), interpolation=cv2.INTER_AREA)
                out = model_cnn.predict(np.expand_dims(target_image, axis=0))
                positive = out[0][0] > out[0][1]
                if CheckTarget:
                    if positive or CheckNeg:
                        if positive:
                            plt.title("plane     " + str(out))
                        else:
                            plt.title("not plane " + str(out))
                        plt.imshow(np.squeeze(target_image))
                        plt.show()
                if out[0][0] > out[0][1]:
                    image_out = cv2.rectangle(image_out, (x1, y1), (x2, y2), (0, 1, 0), 1, cv2.LINE_AA)
        plt.figure()
        plt.imshow(image_out)
        plt.savefig("TestResults\\" + file_path.split('.')[0] + "_od_test.jpg")
        # plt.show()
        plt.close()


Activate_GPU()
FPN_RPN_test()
# test_model_od()
# FPN_RPN_train()
