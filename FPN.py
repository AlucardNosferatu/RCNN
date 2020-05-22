import random
import cv2
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Add, UpSampling2D, Input
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from ROI_Pooling import RoiPoolingConv


class OneHotGen(LabelBinarizer):
    def transform(self, y):
        y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((y, 1 - y))
        else:
            return y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)


EP = 100
BS = 4
pooled_square_size = 3
path = "Images"
annotation = "Airplanes_Annotations"
num_rois = 4


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


def process_roi_to_data(image_out, train_images, train_labels, counter, c, pos=True):
    resized = cv2.resize(image_out, (224, 224), interpolation=cv2.INTER_AREA)
    train_images.append([resized.reshape((224, 224, 3)), np.array([
        float(c[0]) / 256,
        float(c[1]) / 256,
        float(c[2]) / 256,
        float(c[3]) / 256
    ]).reshape((1, 4))])
    if pos:
        train_labels.append(1)
    else:
        train_labels.append(0)
    counter += 1
    return counter, train_images, train_labels


def process_image_and_single_roi(train_images, train_labels, ss_results, gt_values, image_out):
    counter = 0
    false_counter = 0
    flag = 0
    f_flag = 0
    b_flag = 0
    e_for_test = 0
    for gt_val in gt_values:
        print(gt_val)
        for e, result in enumerate(ss_results):
            e_for_test = e
            if e < 2000 and flag == 0:
                x, y, w, h = result
                x1 = x
                y1 = y
                x2 = x1 + w
                y2 = y1 + h
                c = [x1, y1, x2, y2]
                iou = get_iou(gt_val, {"x1": x, "x2": x + w, "y1": y, "y2": y + h})
                if counter < 30:
                    # 选择交并比大于阈值的头30个候选坐标
                    if iou > 0.70:
                        # 交并比阈值0.7
                        counter, train_images, train_labels = process_roi_to_data(
                            image_out,
                            train_images,
                            train_labels,
                            counter,
                            c,
                            pos=True
                        )
                else:
                    # 正样本多于30个
                    f_flag = 1
                if false_counter < 30:
                    # IoU低于阈值0.3，前30个坐标作为负样本（背景）
                    if iou < 0.3:
                        false_counter, train_images, train_labels = process_roi_to_data(
                            image_out,
                            train_images,
                            train_labels,
                            false_counter,
                            c,
                            pos=False
                        )
                else:
                    # 负样本多于30个
                    b_flag = 1
                if f_flag == 1 and b_flag == 1:
                    # 全部样本数量已达到
                    flag = 1
    return train_images, train_labels


def process_image_and_rois(train_images, train_labels, ss_results, gt_values, image_out):
    max_rois_per_batch = 4
    resized = cv2.resize(image_out, (224, 224), interpolation=cv2.INTER_AREA)
    resized = resized.reshape((224, 224, 3))
    rois, labels = rois_pack_up(ss_results, gt_values)
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
    return train_images, train_labels


def rois_pack_up(ss_results, gt_values):
    smallest_fm_size = 7
    src_img_size = 256
    th = src_img_size / smallest_fm_size
    rois_count_per_img = 400
    rois = []
    labels = []
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
        elif iou < 0.3:
            labels.append([1, 0])
            rois.append(
                [
                    x1 / src_img_size,
                    y1 / src_img_size,
                    x2 / src_img_size,
                    y2 / src_img_size
                ]
            )

    length = len(rois)
    for i in range(0, rois_count_per_img - length):
        index = random.randint(0, length - 1)
        rois.append(rois[index])
        labels.append(labels[index])

    rois = rois[:rois_count_per_img]
    labels = labels[:rois_count_per_img]
    return rois, labels


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


def data_cleaner_singleROI(X, y):
    # The th need to be set to fit the smallest FM output by FPN
    th = 1 / 7
    i = 0
    length = len(X)
    old = length
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
    new = length
    print(new)
    print(old)
    print(new / old)
    return X, y


def data_generator():
    train_images = []
    train_labels = []
    for e, i in enumerate(os.listdir(annotation)):
        # 对每一个标记文件（csv）进行操作
        if i.startswith("airplane"):
            ss_results, image_out, gt_values = process_annotation_file(i)
            if num_rois > 1:
                train_images, train_labels = process_image_and_rois(
                    train_images,
                    train_labels,
                    ss_results,
                    gt_values,
                    image_out
                )
            else:
                train_images, train_labels = process_image_and_single_roi(
                    train_images,
                    train_labels,
                    ss_results,
                    gt_values,
                    image_out
                )
    if num_rois > 1:
        pass
    else:
        train_images, train_labels = data_cleaner_singleROI(train_images, train_labels)
    ti_pkl = open('train_images.pkl', 'wb')
    tl_pkl = open('train_labels.pkl', 'wb')
    pickle.dump(train_images, ti_pkl)
    pickle.dump(train_labels, tl_pkl)
    ti_pkl.close()
    tl_pkl.close()
    return train_images, train_labels


def data_loader():
    ti_pkl = open('train_images.pkl', 'rb')
    tl_pkl = open('train_labels.pkl', 'rb')
    train_images = pickle.load(ti_pkl)
    train_labels = pickle.load(tl_pkl)
    ti_pkl.close()
    tl_pkl.close()
    return train_images, train_labels


class FPN(Model):
    def __init__(self):
        super(FPN, self).__init__()
        self.conv1 = Conv2D(256, (1, 1), name='fpn_c5p5')
        self.us1 = UpSampling2D(size=(2, 2), name="fpn_p5upsampled")
        self.conv2 = Conv2D(256, (1, 1), name='fpn_c4p4')
        self.add1 = Add(name="fpn_p4add")
        self.us2 = UpSampling2D(size=(2, 2), name="fpn_p4upsampled")
        self.conv3 = Conv2D(256, (1, 1), name='fpn_c3p3')
        self.add2 = Add(name="fpn_p3add")
        # Attach 3x3 conv to all P layers to get the final feature maps.
        self.conv6 = Conv2D(256, (3, 3), padding="SAME", name="fpn_p3")
        self.conv7 = Conv2D(256, (3, 3), padding="SAME", name="fpn_p4")
        self.conv8 = Conv2D(256, (3, 3), padding="SAME", name="fpn_p5")
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        self.mp = MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")

    def call(self, x):
        x3 = x[0]
        x4 = x[1]
        x5 = x[2]
        p5 = self.conv1(x5)
        p4 = self.add1(
            [
                self.us1(p5),
                self.conv2(x4)
            ]
        )
        p3 = self.add2(
            [
                self.us2(p4),
                self.conv3(x3)
            ]
        )
        p3 = self.conv6(p3)
        p4 = self.conv7(p4)
        p5 = self.conv8(p5)
        return [p3, p4, p5]


def build_model():
    # TODO: Need to change network structure to adapt num_rois
    vgg_model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)
    roi_input = Input(shape=(num_rois, 4), name="input_2")
    v16_layer_indices = [10, 14, 18]
    fpn_input = []
    for each in v16_layer_indices:
        fpn_input.append(vgg_model.layers[each].output)
    fpn_result = FPN()(fpn_input)
    for layers in vgg_model.layers[:18]:
        layers.trainable = False
    roi_result = []

    for i in range(0, len(fpn_result)):
        x = RoiPoolingConv(pooled_square_size, num_rois)([fpn_result[i], roi_input])
        roi_result.append(x)
    cls_result = []
    fpn_len = len(roi_result)
    for i in range(0, num_rois):
        x_list = []
        for j in range(0, fpn_len):
            roi_list = tf.split(roi_result[j], num_rois, 1)
            x = Flatten(name='flat_afterRP_' + str(i) + "_" + str(j))(roi_list[i])
            x = Dense(2, activation='relu')(x)
            x_list.append(x)
        x = tf.concat(x_list, axis=1)
        x = Dense(2, activation='softmax')(x)
        cls_result.append(x)
    x = tf.stack(cls_result, axis=1)
    model_final = Model(inputs=[vgg_model.input, roi_input], outputs=x)
    opt = Adam(lr=0.0001)
    model_final.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=opt,
        metrics=["accuracy"]
    )
    model_final.save("ieeercnn_vgg16_1.h5py")
    return model_final


def train(NewModel=False, GenData=False, UseFPN=True):
    devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    print(devices)
    for gpu in devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    if NewModel:
        build_model()
    model_final = keras.models.load_model(
        "ieeercnn_vgg16_1.h5py",
        custom_objects={
            'RoiPoolingConv': RoiPoolingConv
        }
    )
    tf.keras.utils.plot_model(
        model_final, to_file='model.png', show_shapes=False, show_layer_names=False,
        rankdir='TB', expand_nested=False, dpi=96
    )
    if GenData:
        x_new, y_new = data_generator()
    else:
        x_new, y_new = data_loader()
    x_images = []
    x_rois = []
    for each in tqdm(x_new):
        x_images.append(each[0])
        x_rois.append(each[1])
    x_images = np.array(x_images)
    x_rois = np.array(x_rois)
    if num_rois > 1:
        y = np.array(y_new)
    else:
        one_hot = OneHotGen()
        y = one_hot.fit_transform(y_new)

    checkpoint = ModelCheckpoint(
        "ieeercnn_vgg16_1.h5py",
        monitor='val_loss',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'
    )
    with tf.device('/gpu:0'):
        hist = model_final.fit(
            [x_images, x_rois],
            y,
            verbose=1,
            epochs=EP,
            batch_size=BS,
            callbacks=[checkpoint]
        )


def test_model_cl():
    model_final = keras.models.load_model("ieeercnn_vgg16_1.h5")
    x_new, y_new = data_loader()
    one_hot = OneHotGen()
    y = one_hot.fit_transform(y_new)
    x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size=0.10)
    for test in x_test:
        im = test
        plt.imshow(im)
        img = np.expand_dims(im, axis=0)
        out = model_final.predict(img)

        if out[0][0] > out[0][1]:
            print(str(out) + " " + "plane")
        else:
            print(str(out) + " " + "not plane")
        plt.show()


def test_model_od():
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    model_loaded = keras.models.load_model("ieeercnn_vgg16_1.h5")
    z = 0
    for e, i in enumerate(os.listdir(path)):
        if i.startswith("4"):
            z += 1
            img = cv2.imread(os.path.join(path, i))
            image_out = img.copy()

            # Selective Search will be replaced by ROI proposal
            ss.setBaseImage(img)
            ss.switchToSelectiveSearchFast()
            ss_results = ss.process()

            for e, result in enumerate(ss_results):
                if e < 2000:
                    x, y, w, h = result
                    test_image = image_out[y:y + h, x:x + w]
                    resized = cv2.resize(test_image, (224, 224), interpolation=cv2.INTER_AREA)
                    img = np.expand_dims(resized, axis=0)
                    out = model_loaded.predict(img)
                    if out[0][0] > 0.65:
                        cv2.rectangle(image_out, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
            plt.figure()
            plt.imshow(image_out)
            plt.show()


train()
