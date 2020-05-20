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


class MyLabelBinarizer(LabelBinarizer):
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


EP = 100
BS = 8
pooled_square_size = 7
path = "Images"
annotation = "Airplanes_Annotations"


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


def process_roi_to_data(imout, train_images, train_labels, counter, c, pos=True):
    resized = cv2.resize(imout, (224, 224), interpolation=cv2.INTER_AREA)
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


def process_image_and_roi(train_images, train_labels, ssresults, gtvalues, imout):
    counter = 0
    falsecounter = 0
    flag = 0
    fflag = 0
    bflag = 0
    e_for_test = 0
    for gtval in gtvalues:
        print(gtval)
        for e, result in enumerate(ssresults):
            e_for_test = e
            if e < 2000 and flag == 0:
                x, y, w, h = result
                x1 = x
                y1 = y
                x2 = x1 + w
                y2 = y1 + h
                c = [x1, y1, x2, y2]
                iou = get_iou(gtval, {"x1": x, "x2": x + w, "y1": y, "y2": y + h})
                if counter < 30:
                    # 选择交并比大于阈值的头30个候选坐标
                    if iou > 0.70:
                        # 交并比阈值0.7
                        counter, train_images, train_labels = process_roi_to_data(
                            imout,
                            train_images,
                            train_labels,
                            counter,
                            c,
                            pos=True
                        )
                else:
                    # 正样本多于30个
                    fflag = 1
                if falsecounter < 30:
                    # IoU低于阈值0.3，前30个坐标作为负样本（背景）
                    if iou < 0.3:
                        falsecounter, train_images, train_labels = process_roi_to_data(
                            imout,
                            train_images,
                            train_labels,
                            falsecounter,
                            c,
                            pos=False
                        )
                else:
                    # 负样本多于30个
                    bflag = 1
                if fflag == 1 and bflag == 1:
                    # 全部样本数量已达到
                    flag = 1
    return train_images, train_labels


def process_annot_file(i):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    filename = i.split(".")[0] + ".jpg"
    print(filename)
    imout = cv2.imread(os.path.join(path, filename))
    df = pd.read_csv(os.path.join(annotation, i))
    gtvalues = []
    for row in df.iterrows():
        x1 = int(row[1][0].split(" ")[0])
        y1 = int(row[1][0].split(" ")[1])
        x2 = int(row[1][0].split(" ")[2])
        y2 = int(row[1][0].split(" ")[3])
        gtvalues.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
        # 把标签的坐标数据存入gtvalues
    ss.setBaseImage(imout)
    ss.switchToSelectiveSearchFast()
    ssresults = ss.process()
    return ssresults, imout, gtvalues


def data_cleaner(X, y):
    # TODO: Need more strict filter to deal with data like (x1=4.5 x2=5.1)
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


def data_generator():
    train_images = []
    train_labels = []
    for e, i in enumerate(os.listdir(annotation)):
        # 对每一个标记文件（csv）进行操作
        if i.startswith("airplane"):
            ssresults, imout, gtvalues = process_annot_file(i)
            train_images, train_labels = process_image_and_roi(train_images, train_labels, ssresults, gtvalues, imout)
    train_images, train_labels = data_cleaner(train_images, train_labels)
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
    num_rois = 1
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
    for i in range(0, len(roi_result)):
        x = Flatten(name='flat_afterRP_' + str(i))(roi_result[i])
        x = Dense(64, activation='relu')(x)
        cls_result.append(x)

    x = tf.concat(cls_result, axis=1)
    x = Dense(2, activation="softmax")(x)

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
        custom_objects={'RoiPoolingConv': RoiPoolingConv}
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

    lenc = MyLabelBinarizer()
    y = lenc.fit_transform(y_new)

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
    one_hot = MyLabelBinarizer()
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
            imout = img.copy()

            # Selective Search will be replaced by ROI proposal
            ss.setBaseImage(img)
            ss.switchToSelectiveSearchFast()
            ssresults = ss.process()

            for e, result in enumerate(ssresults):
                if e < 2000:
                    x, y, w, h = result
                    timage = imout[y:y + h, x:x + w]
                    resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                    img = np.expand_dims(resized, axis=0)
                    out = model_loaded.predict(img)
                    if out[0][0] > 0.65:
                        cv2.rectangle(imout, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
            plt.figure()
            plt.imshow(imout)
            plt.show()


train()
