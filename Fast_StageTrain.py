import os
import pickle
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Flatten, TimeDistributed, Dense, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm import tqdm

from FastRCNN import process_annotation_file, rois_pack_up, Activate_GPU, get_slash, OneHot, CheckBatch, get_iou
from ROI_Pooling import RoiPoolingConv

slash, linux = get_slash()
path = "ProcessedData" + slash + "Images"
annotation = "ProcessedData" + slash + "Airplanes_Annotations"


def load_vgg16():
    model = tf.keras.models.load_model("TrainedModels" + slash + "VGG16_FULL.h5")
    return model


def build_vgg16():
    model = tf.keras.applications.VGG16(include_top=True, weights="imagenet")
    x = Dense(2, activation='softmax')(model.output)
    model_final = Model(inputs=model.input, outputs=x)
    opt = Adam(lr=0.0001)
    model_final.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=opt,
        metrics=["accuracy"]
    )
    model_final.save("TrainedModels" + slash + "VGG16_FULL.h5")


def train_vgg16():
    model = load_vgg16()
    x, y, z = data_loader(trainVGG=True)
    x = np.array(x) / 255
    one_hot = OneHot()
    y = one_hot.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
    trdata = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90
    )
    traindata = trdata.flow(
        x=x_train,
        y=y_train
    )
    tsdata = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90
    )
    testdata = tsdata.flow(
        x=x_test,
        y=y_test
    )
    checkpoint = ModelCheckpoint(
        "TrainedModels" + slash + "VGG16_FULL.h5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch"
    )
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir="TensorBoard",
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )
    with tf.device("/gpu:0"):
        hist = model.fit_generator(
            callbacks=[checkpoint, tb_callback],
            validation_data=testdata,
            validation_steps=2,
            generator=traindata,
            steps_per_epoch=50,
            epochs=1000
        )


def load_fm_extractor():
    model = tf.keras.models.load_model("TrainedModels" + slash + "VGG16_FM_28X28.h5")
    return model


def build_fm_extractor():
    model_cnn = load_vgg16()
    x = model_cnn.layers[13].output
    model = Model(inputs=model_cnn.input, outputs=x)
    model.compile()
    model.save("TrainedModels" + slash + "VGG16_FM_28X28.h5")


def load_roi_p_classifier():
    model = tf.keras.models.load_model(
        "TrainedModels" + slash + "ROI_P_CLASSIFIER.h5",
        custom_objects={'RoiPoolingConv': RoiPoolingConv}
    )
    return model


def build_roi_p_classifier():
    num_rois = 32
    pooled_square_size = 56
    roi_input = Input(shape=(num_rois, 4), name="roi_input")
    fm_input = Input(shape=(224, 224, 3), name="fm_input")
    x = RoiPoolingConv(pooled_square_size, roi_input.shape[1])([fm_input, roi_input])
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(x)
    x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(Dense(512, activation='tanh'))(x)
    x = TimeDistributed(Dense(2, activation='softmax'))(x)
    model_final = Model(inputs=[fm_input, roi_input], outputs=x)
    opt = Adam(lr=0.0001)
    model_final.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=opt,
        metrics=["accuracy"]
    )
    model_final.save("TrainedModels" + slash + "ROI_P_CLASSIFIER.h5")


def train_roi_p_classifier():
    model = load_roi_p_classifier()
    opt = Adam(lr=0.0001)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=opt,
        metrics=["accuracy"]
    )
    x, y, z = data_loader(trainVGG=False)
    x_fms = []
    x_rois = []
    for i in tqdm(range(len(x))):
        x_fms.append(z[i]/255)
        x_rois.append(x[i][1])
    x_fms = np.array(x_fms)
    x_rois = np.array(x_rois)
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir="TensorBoard",
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="TrainedModels" + slash + "ROI_P_CLASSIFIER.h5",
        save_weights_only=False,
        verbose=1
    )
    with tf.device('/gpu:0'):
        model.fit(
            [x_fms, x_rois],
            np.array(y),
            verbose=1,
            epochs=100,
            batch_size=16,
            callbacks=[tb_callback, cp_callback],
            # steps_per_epoch=200
        )


def rois_pack_up(ss_results, gt_values, rois_count_per_img):
    skip = False
    smallest_fm_size = 28
    src_img_size = 256
    th = src_img_size / smallest_fm_size
    rois = []
    labels = []
    count = 0
    false_count = 0
    neg_ratio = 0.7
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
        if false_count < (neg_ratio * rois_count_per_img):
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
    print(false_count/length)
    pos_indices = list(np.where(np.array(labels)[:, 0] != 1)[0])
    neg_indices = list(np.where(np.array(labels)[:, 0] == 1)[0])
    if len(pos_indices) <= 0:
        skip = True
    else:
        for i in range(0, rois_count_per_img - length):
            if labels.count([1, 0]) / len(labels) > neg_ratio:
                index = random.choice(pos_indices)
            else:
                index = random.choice(neg_indices)
            rois.append(rois[index])
            labels.append(labels[index])
        labels = labels[:rois_count_per_img]
        rois = rois[:rois_count_per_img]
    return rois, labels, skip


def process_image_and_rois(src_images, train_images, train_labels, ss_results, gt_values, image_out, vgg):
    max_rois_per_batch = 32
    resized = cv2.resize(image_out, (224, 224), interpolation=cv2.INTER_AREA)
    resized = resized.reshape((224, 224, 3))
    fm = vgg.predict(np.expand_dims(resized / 255, axis=0))
    rois, labels, skip = rois_pack_up(ss_results, gt_values, rois_count_per_img=128)
    if not skip:
        for i in range(0, int(len(rois) / max_rois_per_batch)):
            train_labels.append(
                np.array(
                    labels[i * max_rois_per_batch:(i + 1) * max_rois_per_batch]
                ).reshape(max_rois_per_batch, 2)
            )
            train_images.append(
                [
                    np.squeeze(fm),
                    np.array(
                        rois[i * max_rois_per_batch:(i + 1) * max_rois_per_batch]
                    ).reshape(max_rois_per_batch, 4)
                ]
            )
            src_images.append(resized)
    else:
        print("Due to too many negative samples, skip this img")
    return train_images, train_labels, src_images


def fm_generator():
    train_images = []
    train_labels = []
    src_images = []
    vgg = load_fm_extractor()
    for e, i in enumerate(os.listdir(annotation)):
        # 对每一个标记文件（csv）进行操作
        if i.startswith("airplane"):
            ss_results, image_out, gt_values = process_annotation_file(i)
            train_images, train_labels, src_images = process_image_and_rois(
                src_images,
                train_images,
                train_labels,
                ss_results,
                gt_values,
                image_out,
                vgg
            )
    tf_pkl = open("ProcessedData" + slash + 'train_fm&roi_fm_fast.pkl', 'wb')
    tl_pkl = open("ProcessedData" + slash + 'train_labels_fm_fast.pkl', 'wb')
    ti_pkl = open("ProcessedData" + slash + 'train_images_fm_fast.pkl', 'wb')
    pickle.dump(train_images, tf_pkl)
    pickle.dump(train_labels, tl_pkl)
    pickle.dump(src_images, ti_pkl)
    ti_pkl.close()
    tl_pkl.close()
    tf_pkl.close()
    return train_images, train_labels


def data_loader(trainVGG=True):
    if trainVGG:
        ti_pkl = open("ProcessedData" + slash + "train_images_cnn.pkl", "rb")
        tl_pkl = open("ProcessedData" + slash + "train_labels_cnn.pkl", "rb")
        train_images = pickle.load(ti_pkl)
        train_labels = pickle.load(tl_pkl)
        src_images = []
        ti_pkl.close()
        tl_pkl.close()
    else:
        tf_pkl = open("ProcessedData" + slash + 'train_fm&roi_fm_fast.pkl', 'rb')
        tl_pkl = open("ProcessedData" + slash + 'train_labels_fm_fast.pkl', 'rb')
        ti_pkl = open("ProcessedData" + slash + 'train_images_fm_fast.pkl', 'rb')
        train_images = pickle.load(tf_pkl)
        train_labels = pickle.load(tl_pkl)
        src_images = pickle.load(ti_pkl)
        tf_pkl.close()
    ti_pkl.close()
    tl_pkl.close()
    state = np.random.get_state()
    np.random.shuffle(train_images)
    np.random.set_state(state)
    np.random.shuffle(train_labels)
    np.random.set_state(state)
    np.random.shuffle(src_images)
    return train_images, train_labels, src_images


def CheckBatch_FM():
    x, y, z = data_loader()
    plt.rcParams['savefig.dpi'] = 720
    plt.rcParams['figure.dpi'] = 720
    for i in range(len(x)):
        image = x[i][0].copy()
        plt.figure(1)
        plt.imshow(z[i])
        plt.savefig("TestResults" + slash + str(i) + "_src.jpg")
        plt.close(1)
        plt.figure(2)
        for j in tqdm(range(image.shape[2])):
            plt.subplot(16, 32, j + 1)
            plt.axis('off')
            plt.imshow(image[:, :, j])
        plt.savefig("TestResults" + slash + str(i) + "_fm.jpg")
        # plt.show()
        plt.close(2)


def CheckVGG():
    vgg = load_fm_extractor()
    plt.rcParams['savefig.dpi'] = 720
    plt.rcParams['figure.dpi'] = 720
    for e, i in enumerate(os.listdir(path)):
        # 对每一个标记文件（csv）进行操作
        if i.startswith("airplane"):
            image_out = cv2.imread(os.path.join(path, i))
            image_out = cv2.resize(image_out, (224, 224))
            plt.figure(1)
            plt.imshow(image_out)
            plt.savefig("TestResults" + slash + str(i) + "_src.jpg")
            # plt.show()
            plt.close(1)
            fm = vgg.predict(np.expand_dims(image_out, axis=0))
            fm = np.squeeze(fm)
            plt.figure(2)
            for j in tqdm(range(fm.shape[2])):
                plt.subplot(16, 32, j + 1)
                plt.axis('off')
                plt.imshow(fm[:, :, j])
            plt.savefig("TestResults" + slash + str(i) + "_fm.jpg")
            # plt.show()
            plt.close(2)


def TestROIP():
    vgg = load_fm_extractor()
    roi_p = load_roi_p_classifier()
    for e, i in enumerate(os.listdir(annotation)):
        # 对每一个标记文件（csv）进行操作
        if i.startswith("airplane"):
            ss_results, image_out, gt_values = process_annotation_file(i)
            scale = image_out.shape[0]
            image_out = cv2.resize(image_out, (224, 224))
            fm = vgg.predict(np.expand_dims(image_out, axis=0))
            ss_results = ss_results[:32].tolist()
            for j in range(len(ss_results)):
                x = ss_results[j][0]
                y = ss_results[j][1]
                w = ss_results[j][2]
                h = ss_results[j][3]
                ss_results[j] = [x, y, x + w, y + h]
            ss_results = np.array(ss_results).reshape((32, 4)) / scale
            ss_results = np.expand_dims(ss_results, axis=0)
            try:
                result = roi_p.predict([fm, ss_results])
            except Exception as e:
                print(repr(e))
                continue
            ss_results = np.squeeze(ss_results * 224).astype('uint32')
            result = np.squeeze(result)
            for j in range(result.shape[0]):
                print(result[j, :])
                if result[j, 0] < result[j, 1]:
                    image_out = cv2.rectangle(
                        image_out,
                        (ss_results[j, 0], ss_results[j, 1]),
                        (ss_results[j, 2], ss_results[j, 3]),
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA
                    )
            plt.figure()
            plt.imshow(image_out)
            plt.show()
        print("Done")


# Activate_GPU()
# TestROIP()
# build_fm_extractor()
# build_roi_p_classifier()
train_roi_p_classifier()
# build_vgg16()
# train_vgg16()
# fm_generator()
# CheckBatch_FM()
# CheckBatch(trainFast=False)
# CheckVGG()
