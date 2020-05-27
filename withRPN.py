import os
import pickle
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Conv2D, Input
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from RCNN import path, annotation
from RPN_Loss import RPNLoss

roi_count = 200


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


def rois_pack_up(ss_results, gt_values, shape):
    skip = False
    x_y_w_h = True
    rois_count_per_img = 1000
    rois = []
    length = len(rois)
    for e, result in enumerate(ss_results):

        # region Get coordinates
        x, y, w, h = result
        x1 = x
        y1 = y
        x2 = x1 + w
        y2 = y1 + h
        # endregion

        # region Get largest iou
        iou = 0
        for gt_val in gt_values:
            temp = get_iou(gt_val, {"x1": x, "x2": x + w, "y1": y, "y2": y + h})
            if temp > iou:
                iou = temp
        # endregion

        # region Append ROI to list
        if iou > 0.7:
            if x_y_w_h:
                x1 /= shape[0]
                x2 = w / shape[0]
                y1 /= shape[1]
                y2 = h / shape[1]
                x1 *= 224
                x2 *= 224
                y1 *= 224
                y2 *= 224
            else:
                x1 /= shape[0]
                x2 /= shape[0]
                y1 /= shape[1]
                y2 /= shape[1]
                x1 *= 224
                x2 *= 224
                y1 *= 224
                y2 *= 224
            rois.append(
                [
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ]
            )
        length = len(rois)
        # endregion

    if length == 0:
        skip = True
        return rois, skip

    # region Fill list to specific length
    for i in range(0, rois_count_per_img - length):
        index = random.randint(0, length - 1)
        rois.append(rois[index])
    rois = rois[:rois_count_per_img]
    # endregion

    return rois, skip


def process_image_and_rois(train_images, train_labels, ss_results, gt_values, image_out):
    max_rois_per_batch = roi_count
    resized = cv2.resize(image_out, (224, 224), interpolation=cv2.INTER_AREA)
    resized = resized.reshape((224, 224, 3))
    rois, skip = rois_pack_up(ss_results, gt_values, image_out.shape)
    if not skip:
        for i in range(0, int(len(rois) / max_rois_per_batch)):
            train_labels.append(
                np.array(
                    rois[i * max_rois_per_batch:(i + 1) * max_rois_per_batch]
                ).reshape(max_rois_per_batch, 4)
            )
            train_images.append(
                resized
            )
    return train_images, train_labels


def process_annotation_file(i):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    filename = i.split(".")[0] + ".jpg"
    print(filename)
    image_out = cv2.imread(os.path.join(path, filename))
    df = pd.read_csv(os.path.join(annotation, i))
    gt_values = []
    for row in df.iterrows():
        x1 = int(row[1][0].split(" ")[0])
        y1 = int(row[1][0].split(" ")[1])
        x2 = int(row[1][0].split(" ")[2])
        y2 = int(row[1][0].split(" ")[3])
        gt_values.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
        # 把标签的坐标数据存入gt_values
    ss.setBaseImage(image_out)
    ss.switchToSelectiveSearchFast()
    ss_results = ss.process()
    return ss_results, image_out, gt_values


def data_generator():
    train_images = []
    train_labels = []
    for e, i in enumerate(os.listdir(annotation)):
        # 对每一个标记文件（csv）进行操作
        if i.startswith("airplane"):
            ss_results, image_out, gt_values = process_annotation_file(i)
            train_images, train_labels = process_image_and_rois(
                train_images,
                train_labels,
                ss_results,
                gt_values,
                image_out
            )
    ti_pkl = open('ProcessedData\\train_images_rpn.pkl', 'wb')
    tl_pkl = open('ProcessedData\\train_labels_rpn.pkl', 'wb')
    pickle.dump(train_images, ti_pkl)
    pickle.dump(train_labels, tl_pkl)
    ti_pkl.close()
    tl_pkl.close()
    x_new = np.array(train_images)
    y_new = np.array(train_labels)
    return x_new, y_new


def data_loader():
    ti_pkl = open('ProcessedData\\train_images_rpn.pkl', 'rb')
    tl_pkl = open('ProcessedData\\train_labels_rpn.pkl', 'rb')
    train_images = pickle.load(ti_pkl)
    train_labels = pickle.load(tl_pkl)
    ti_pkl.close()
    tl_pkl.close()
    x_new = np.array(train_images)
    y_new = np.array(train_labels)
    return x_new, y_new


def prototype_model_build():
    vgg_model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)
    after_flatten = vgg_model.layers[19].output
    roi_proposal = []
    print("VGG16 imported.")
    for i in tqdm(range(roi_count)):
        x = Dense(16, activation='relu')(after_flatten)
        x = Dense(8, activation='relu')(x)
        x = Dense(4, activation='relu')(x)
        roi_proposal.append(x)
    x = tf.stack(roi_proposal, axis=1)
    model_final = Model(inputs=vgg_model.input, outputs=x)
    opt = Adam(lr=0.0001)
    model_final.compile(
        loss=RPNLoss,
        optimizer=opt,
        metrics=["accuracy"]
    )
    print("Model compiled.")
    # tf.keras.utils.plot_model(
    #     model_final, to_file='model.png', show_shapes=False, show_layer_names=False,
    #     rankdir='TB', expand_nested=False, dpi=96
    # )
    model_final.save("TrainedModels\\RPN_Prototype.h5")
    print("Model saved.")
    return model_final


def prototype_model_train():
    x_new, y_new = data_loader()
    model_final = tf.keras.models.load_model("TrainedModels\\RPN_Prototype.h5", custom_objects={'RPNLoss': RPNLoss})
    checkpoint = ModelCheckpoint(
        "TrainedModels\\RPN_Prototype.h5",
        monitor='val_loss',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'
    )

    with tf.device('/gpu:0'):
        gpu_list = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in gpu_list:
            tf.config.experimental.set_memory_growth(gpu, True)
        model_final.fit(
            x_new,
            y_new,
            callbacks=[checkpoint],
            epochs=100,
            verbose=1,
            batch_size=16
        )


def prototype_model_test():
    x_y_w_h = True
    x_new, y_new = data_loader()
    model_final = tf.keras.models.load_model("TrainedModels\\RPN_Prototype.h5", custom_objects={'RPNLoss': RPNLoss})
    for i in range(0, x_new.shape[0], 5):
        count = 0
        result = model_final.predict(x_new[i, :, :, :].reshape((1, 224, 224, 3)))
        image_out = x_new[i, :, :, :]
        b, g, r = cv2.split(image_out)
        image_out = cv2.merge([r, g, b])
        gt_values = []
        for roi in tqdm(range(200)):
            x1 = int(y_new[i, roi, 0])
            y1 = int(y_new[i, roi, 1])
            x2 = int(y_new[i, roi, 2])
            y2 = int(y_new[i, roi, 3])
            if x_y_w_h:
                x2 += x1
                y2 += y1
            if x1 == x2 or y1 == y2:
                print("zero area error!")
            gt_values.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
        for roi in tqdm(range(200)):
            x1 = int(result[:, roi, 0])
            y1 = int(result[:, roi, 1])
            x2 = int(result[:, roi, 2])
            y2 = int(result[:, roi, 3])
            if x_y_w_h:
                x2 += x1
                y2 += y1
            if x1 >= x2 or y1 >= y2:
                print("zero area error!")
                continue
            iou_list = []
            for gt_val in gt_values:
                temp = get_iou(gt_val, {"x1": x1, "x2": x2, "y1": y1, "y2": y2})
                iou_list.append(temp)
            if max(iou_list) > 0.7:
                count += 1
                image_out = cv2.rectangle(image_out, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
            else:
                image_out = cv2.rectangle(image_out, (x1, y1), (x2, y2), (255, 0, 0), 1, cv2.LINE_AA)
        plt.figure()
        plt.imshow(image_out)
        plt.savefig("TestResults\\第" + str(i) + "次测试，命中比例：" + str(int(100 * count / 200)) + "%.jpg")
        plt.close()


def copy_model_build():
    k = 9
    feature_map_tile = Input(shape=(None, None, 1536))
    convolution_3x3 = Conv2D(
        filters=512,
        kernel_size=(3, 3),
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
    model_rpn.compile(optimizer='adam', loss={'scores1': 'binary_crossentropy', 'deltas1': 'mse'})
    tf.keras.utils.plot_model(
        model_rpn, to_file='model.png', show_shapes=True, show_layer_names=False,
        rankdir='TB', expand_nested=False, dpi=96
    )
    model_rpn.save("TrainedModels\\RPN_Prototype.h5")

