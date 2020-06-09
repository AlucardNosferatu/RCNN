import os
import pickle
import platform
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input, Dense, Flatten, TimeDistributed, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.applications.vgg16 import preprocess_input

from ROI_Pooling import RoiPoolingConv


def get_slash():
    linux = False
    if platform.system() == "Linux":
        linux = True
    if linux:
        print("Linux")
        sl = "/"
    else:
        print("Windows")
        sl = "\\"
    return sl, linux


slash, linux = get_slash()

EP = 100
BS = 2


class OneHot(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1 - Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)


path = "ProcessedData" + slash + "Images"
annotation = "ProcessedData" + slash + "Airplanes_Annotations"


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


def build_model():
    pooled_square_size = 7
    roi_input = Input(shape=(None, 4), name="input_2")
    model_cnn = tf.keras.applications.VGG16(
        include_top=True,
        weights='imagenet'
    )
    model_cnn.trainable = True
    x = model_cnn.layers[17].output
    x = RoiPoolingConv(pooled_square_size)([x, roi_input])
    x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(
        Dense(
            4096,
            activation='selu',
            kernel_initializer=RandomNormal(stddev=0.01),
            kernel_regularizer=l2(0.0005),
            bias_regularizer=l2(0.0005)
        )
    )(x)
    x = TimeDistributed(
        Dense(
            4096,
            activation='selu',
            kernel_initializer=RandomNormal(stddev=0.01),
            kernel_regularizer=l2(0.0005),
            bias_regularizer=l2(0.0005)
        )
    )(x)
    x = TimeDistributed(Dense(2, activation='softmax', kernel_initializer='zero'))(x)
    model_final = Model(inputs=[model_cnn.input, roi_input], outputs=x)
    opt = Adam(lr=0.0001)
    model_final.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=opt,
        metrics=["accuracy"]
    )
    if not linux:
        tf.keras.utils.plot_model(
            model_final,
            "model.png",
            show_shapes=True,
            show_layer_names=False,
            rankdir='TB'
        )
    model_final.save("TrainedModels" + slash + "FastRCNN.h5")


def prepare_test_data():
    for e, i in enumerate(os.listdir(annotation)):
        # 对每一个标记文件（csv）进行操作
        if i.startswith("airplane_001"):
            # 只有名称带airplane才是有目标的存在的样本
            filename = i.split(".")[0] + ".jpg"
            image = cv2.imread(os.path.join(path, filename))
            image = cv2.resize(image, (224, 224))
            image = image.reshape((-1, 224, 224, 3))
            image = tf.cast(image, tf.float32)
            df = pd.read_csv(os.path.join(annotation, i))
            gt_values = []
            for row in df.iterrows():
                x1 = float(int(row[1][0].split(" ")[0]) / 256)
                y1 = float(int(row[1][0].split(" ")[1]) / 256)
                x2 = float(int(row[1][0].split(" ")[2]) / 256)
                y2 = float(int(row[1][0].split(" ")[3]) / 256)
                gt_values.append([x1, y1, x2, y2])
            roi = np.array(gt_values[0], dtype='float32').reshape((-1, 1, 4))
    return image, roi


def test_model(image, roi, model, file_name):
    # fm_before_roip = Model(inputs=model.input, outputs=model.layers[13].output)
    # fm_after_roip = Model(inputs=model.input, outputs=model.layers[16].output)
    # fm_b = fm_before_roip.predict([image / 255, roi])
    # fm_a = fm_after_roip.predict([image / 255, roi])
    result = model.predict([preprocess_input(image), roi])
    # plt.figure()
    # for i in tqdm(range(fm_b.shape[3])):
    #     plt.subplot(16, 32, i + 1)
    #     plt.axis('off')
    #     plt.imshow(fm_b[0, :, :, i])
    # plt.savefig("TestResults" + slash + file_name + "fm_b.jpg")
    # plt.close()
    # for i in range(fm_a.shape[1]):
    #     plt.figure()
    #     for j in tqdm(range(fm_a.shape[4])):
    #         plt.subplot(16, 32, j + 1)
    #         plt.axis('off')
    #         plt.imshow(fm_a[0, i, :, :, j])
    #     plt.savefig("TestResults" + slash + file_name + "fm_a_" + str(i) + ".jpg")
    #     plt.close()
    image = np.array(image).astype('uint8').reshape((224, 224, 3))
    for i in range(result.shape[1]):
        x1 = int(roi[0][i][0] * 224)
        y1 = int(roi[0][i][1] * 224)
        x2 = int(roi[0][i][2] * 224)
        y2 = int(roi[0][i][3] * 224)
        if result[0, i, 1] > result[0, i, 0]:
            print(str(result[0, i, :]) + " " + "plane")
            image = cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )
        else:
            # print(str(result[0, i, :]) + " " + "not plane")
            # image = cv2.rectangle(
            #     image,
            #     (x1, y1),
            #     (x2, y2),
            #     (255, 0, 0),
            #     1,
            #     cv2.LINE_AA
            #
            pass
    plt.figure()
    plt.imshow(image)
    plt.savefig("TestResults" + slash + file_name + "_result.jpg")
    # plt.show()
    plt.close()


def batch_test():
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    model = tf.keras.models.load_model(
        "TrainedModels" + slash + "FastRCNN.h5",
        custom_objects={'RoiPoolingConv': RoiPoolingConv}
    )
    for e, i in enumerate(os.listdir(path)):
        if i.startswith("4"):
            image = cv2.imread(os.path.join(path, i))
            ss.setBaseImage(image)
            ss.switchToSelectiveSearchFast()
            ss_results = ss.process()
            image = np.expand_dims(cv2.resize(image, (224, 224)), axis=0)
            rois_list = []
            for e_roi, result in enumerate(ss_results):
                if len(rois_list) >= 32:
                    continue
                x, y, w, h = result
                if w <= (256 / 14) or h <= (256 / 14):
                    continue
                else:
                    x1 = x
                    y1 = y
                    x2 = x1 + w
                    y2 = y1 + h
                    x1 = float(x1) / 256
                    y1 = float(y1) / 256
                    x2 = float(x2) / 256
                    y2 = float(y2) / 256
                rois_list.append([x1, y1, x2, y2])
            length = len(rois_list)
            while len(rois_list) < 64:
                index = random.randint(0, length - 1)
                rois_list.append(rois_list[index])
            roi = np.array(rois_list, dtype='float32').reshape((1, -1, 4))
            # try:
            test_model(image, roi, model, i)
            # except Exception as e:
            #     print(repr(e))


def rois_pack_up(ss_results, gt_values, rois_count_per_img):
    skip = False
    smallest_fm_size = 14
    src_img_size = 256
    th = src_img_size / smallest_fm_size
    rois = []
    labels = []
    count = 0
    false_count = 0
    for e, result in enumerate(ss_results):
        x, y, w, h = result
        x1 = x
        y1 = y
        x2 = x1 + w
        y2 = y1 + h
        if abs(x1 - x2) <= th or abs(y1 - y2) <= th:
            continue
        iou = 0
        for gt_val in gt_values:
            temp = get_iou(gt_val, {"x1": x, "x2": x + w, "y1": y, "y2": y + h})
            if temp > iou:
                iou = temp
        if iou > 0.7:
            labels.append([0, 1])
            rois.append(
                [
                    x1 / src_img_size,
                    y1 / src_img_size,
                    x2 / src_img_size,
                    y2 / src_img_size
                ]
            )
            count += 1
        if false_count <= count:
            if iou < 0.3:
                labels.append([1, 0])
                rois.append(
                    [
                        x1 / src_img_size,
                        y1 / src_img_size,
                        x2 / src_img_size,
                        y2 / src_img_size
                    ]
                )
                false_count += 1
    length = len(rois)
    print(length)
    if len(labels) <= 0:
        skip = True
    else:
        for i in range(0, rois_count_per_img - length):
            index = random.randint(0, length - 1)
            rois.append(rois[index])
            labels.append(labels[index])
        while len(rois) > rois_count_per_img and ([1, 0] in labels):
            index = labels.index([1, 0])
            del labels[index]
            del rois[index]
        labels = labels[:rois_count_per_img]
        rois = rois[:rois_count_per_img]
    return rois, labels, skip


def process_image_and_rois(train_images, train_labels, ss_results, gt_values, image_out):
    max_rois_per_batch = 64
    resized = cv2.resize(image_out, (224, 224), interpolation=cv2.INTER_AREA)
    resized = resized.reshape((224, 224, 3))
    rois, labels, skip = rois_pack_up(ss_results, gt_values, rois_count_per_img=64)
    if not skip:
        for i in range(0, int(len(rois) / max_rois_per_batch)):
            train_labels.append(
                np.array(
                    labels[i * max_rois_per_batch:(i + 1) * max_rois_per_batch]
                ).reshape(max_rois_per_batch, 2)
            )
            train_images.append(
                [
                    resized,
                    np.array(
                        rois[i * max_rois_per_batch:(i + 1) * max_rois_per_batch]
                    ).reshape(max_rois_per_batch, 4)
                ]
            )
    else:
        print("Due to too many negative samples, skip this img")
        plt.imshow(resized)
        plt.show()
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
    ti_pkl = open("ProcessedData" + slash + 'train_images_fast.pkl', 'wb')
    tl_pkl = open("ProcessedData" + slash + 'train_labels_fast.pkl', 'wb')
    pickle.dump(train_images, ti_pkl)
    pickle.dump(train_labels, tl_pkl)
    ti_pkl.close()
    tl_pkl.close()
    return train_images, train_labels


def data_loader(trainFast=True, shuffle=True):
    if trainFast:
        ti_pkl = open("ProcessedData" + slash + 'train_images_fast.pkl', 'rb')
        tl_pkl = open("ProcessedData" + slash + 'train_labels_fast.pkl', 'rb')
    else:
        ti_pkl = open("ProcessedData" + slash + 'train_images_cnn.pkl', 'rb')
        tl_pkl = open("ProcessedData" + slash + 'train_labels_cnn.pkl', 'rb')
    train_images = pickle.load(ti_pkl)
    train_labels = pickle.load(tl_pkl)
    ti_pkl.close()
    tl_pkl.close()
    if shuffle:
        state = np.random.get_state()
        np.random.shuffle(train_images)
        np.random.set_state(state)
        np.random.shuffle(train_labels)
    return train_images, train_labels


def train(model_path="TrainedModels" + slash + "FastRCNN.h5", gen_data=False):
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'RoiPoolingConv': RoiPoolingConv}
    )
    opt = Adam(lr=0.0001)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=opt,
        metrics=["accuracy"]
    )
    for layer in model.layers:
        print(layer.name + "  " + str(layer.trainable))
    if not linux:
        tf.keras.utils.plot_model(
            model,
            "model.png",
            show_shapes=True,
            show_layer_names=False,
            rankdir='TB'
        )
    if gen_data:
        x, y = data_generator()
    else:
        x, y = data_loader()

    x_images = []
    x_rois = []
    for each in x:
        x_images.append(each[0])
        x_rois.append(each[1])
    x_images = np.array(x_images)
    x_images = preprocess_input(x_images)
    x_rois = np.array(x_rois)
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir="TensorBoard",
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        save_weights_only=False,
        verbose=1
    )

    with tf.device('/gpu:0'):
        model.fit(
            [x_images, x_rois],
            np.array(y),
            verbose=1,
            epochs=EP,
            batch_size=BS,
            callbacks=[cp_callback, tb_callback],
            # steps_per_epoch=100
        )


def data_divider(data):
    data = list(data)
    db = data[:int(len(data) / 2)]
    for i in range(0, int(len(data) / 2)):
        del data[int(len(data) / 2)]
    return data, db


def data_sampler(X, y):
    X, Xb = data_divider(X)
    y, yb = data_divider(y)
    X, Xc = data_divider(X)
    y, yc = data_divider(y)
    X, Xd = data_divider(X)
    y, yd = data_divider(y)
    X, Xe = data_divider(X)
    y, ye = data_divider(y)
    X, Xf = data_divider(X)
    y, yf = data_divider(y)
    X, Xg = data_divider(X)
    y, yg = data_divider(y)
    X, Xh = data_divider(X)
    y, yh = data_divider(y)
    X, Xi = data_divider(X)
    y, yi = data_divider(y)
    X, Xj = data_divider(X)
    y, yj = data_divider(y)
    del Xb, yb, Xc, yc, Xd, yd, Xe, ye, Xf, yf, Xg, yg, Xh, yh, Xi, yi, Xj, yj
    return X, y


def data_cleaner(X, y):
    th = 1 / 14
    i = 0
    length = len(X)
    while i < length:
        roi = list(X[i][1].reshape(4))
        x1 = roi[0]
        y1 = roi[1]
        x2 = roi[2]
        y2 = roi[3]
        if abs(x1 - x2) <= th or abs(y1 - y2) <= th:
            print(roi)
            del X[i], y[i]
        else:
            i += 1
        length = len(X)
    return X, y


def Activate_GPU():
    gpu_list = tf.config.experimental.list_physical_devices(device_type='GPU')
    print(gpu_list)
    for gpu in gpu_list:
        tf.config.experimental.set_memory_growth(gpu, True)


def CheckBatch(trainFast=True):
    x, y = data_loader(trainFast, shuffle=False)
    if trainFast:
        for i in range(len(x)):
            print(i)
            image = x[i][0].copy()
            rois = x[i][1]
            for j in range(rois.shape[0]):
                roi = rois[j, :]
                # print(roi)
                x1 = int(roi[0] * 224)
                y1 = int(roi[1] * 224)
                x2 = int(roi[2] * 224)
                y2 = int(roi[3] * 224)
                if y[i][j, 1]:
                    image = cv2.rectangle(
                        image,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA
                    )
                else:
                    image = cv2.rectangle(
                        image,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA
                    )
            plt.figure()
            plt.imshow(image)
            plt.savefig("TestResults" + slash + "train_batch" + slash + str(i) + ".jpg")
            # plt.show()
            plt.close()
    else:
        for i in range(len(x)):
            if y[i]:
                plt.figure()
                plt.imshow(x[i])
                plt.show()
    print("Done")


# CheckBatch()
Activate_GPU()
batch_test()
# train()
# build_model()
# data_generator()
