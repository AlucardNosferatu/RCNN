import os, cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Add, UpSampling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model, optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from ROI_Pooling import RoiPoolingConv
import matplotlib.pyplot as plt
import pickle

tf.compat.v1.disable_eager_execution()


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


path = "Images"
annot = "Airplanes_Annotations"


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


def data_generator():
    train_images = []
    train_labels = []
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    for e, i in enumerate(os.listdir(annot)):
        # 对每一个标记文件（csv）进行操作
        try:
            if i.startswith("airplane"):
                # 只有名称带airplane才是有目标的存在的样本
                filename = i.split(".")[0] + ".jpg"
                print(e, filename)
                image = cv2.imread(os.path.join(path, filename))
                df = pd.read_csv(os.path.join(annot, i))
                gtvalues = []
                for row in df.iterrows():
                    x1 = int(row[1][0].split(" ")[0])
                    y1 = int(row[1][0].split(" ")[1])
                    x2 = int(row[1][0].split(" ")[2])
                    y2 = int(row[1][0].split(" ")[3])
                    gtvalues.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
                    # 把标签的坐标数据存入gtvalues

                ss.setBaseImage(image)
                ss.switchToSelectiveSearchFast()
                ssresults = ss.process()
                # 加载SS ROI提出器

                imout = image.copy()
                counter = 0
                falsecounter = 0
                flag = 0
                fflag = 0
                bflag = 0
                for e, result in enumerate(ssresults):
                    if e < 2000 and flag == 0:
                        # 对SS产生的头2k个结果（坐标）进行处理
                        for gtval in gtvalues:
                            # 对这张图上多个标签坐标进行处理
                            x, y, w, h = result
                            iou = get_iou(gtval, {"x1": x, "x2": x + w, "y1": y, "y2": y + h})
                            # 计算候选坐标和这一标签坐标的交并比
                            if counter < 30:
                                # 选择交并比大于阈值的头30个候选坐标
                                if iou > 0.70:
                                    # 交并比阈值0.7
                                    timage = imout[y:y + h, x:x + w]
                                    resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                                    train_images.append(resized)
                                    train_labels.append(1)
                                    counter += 1
                            else:
                                fflag = 1
                            if falsecounter < 30:
                                # IoU低于阈值0.3，前30个坐标作为负样本（背景）
                                if iou < 0.3:
                                    timage = imout[y:y + h, x:x + w]
                                    resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                                    train_images.append(resized)
                                    train_labels.append(0)
                                    falsecounter += 1
                            else:
                                bflag = 1
                        if fflag == 1 and bflag == 1:
                            # print("inside")
                            flag = 1
        except Exception as e:
            print(e)
            print("error in " + filename)
            continue
    TI_PKL = open('train_images.pkl', 'wb')
    TL_PKL = open('train_labels.pkl', 'wb')
    pickle.dump(train_images, TI_PKL)
    pickle.dump(train_labels, TL_PKL)
    TI_PKL.close()
    TL_PKL.close()
    X_new = np.array(train_images)
    y_new = np.array(train_labels)
    return X_new, y_new


def data_loader():
    TI_PKL = open('train_images.pkl', 'rb')
    TL_PKL = open('train_labels.pkl', 'rb')
    train_images = pickle.load(TI_PKL)
    train_labels = pickle.load(TL_PKL)
    TI_PKL.close()
    TL_PKL.close()
    X_new = np.array(train_images)
    y_new = np.array(train_labels)
    return X_new, y_new


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


def train(NewModel=False, GenData=False, UseFPN=True):
    if NewModel:
        vgg_model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)
        result = None
        if UseFPN:
            fpn = FPN()
            v16_layer_indices = [10, 14, 18]
            fpn_input = []
            for each in v16_layer_indices:
                fpn_input.append(vgg_model.layers[each].output)
            result = fpn(fpn_input)

        for layers in vgg_model.layers[:18]:
            layers.trainable = False

        if result:
            roi_result = []
            for i in range(0, len(result)):
                x = Flatten(name='flat_afterFM_' + str(i))(result[i])
                x = Dense(64, activation='relu')(x)
                roi_result.append(x)
            x = Add()(roi_result)
            predictions = Dense(2, activation="softmax")(x)
        else:
            x = vgg_model.layers[-2].output
            predictions = Dense(2, activation="softmax")(x)

        model_final = Model(inputs=vgg_model.input, outputs=predictions)
        opt = Adam(lr=0.0001)
        model_final.compile(
            loss=keras.losses.CategoricalCrossentropy(),
            optimizer=opt,
            metrics=["accuracy"]
        )
    else:
        model_final = keras.models.load_model("ieeercnn_vgg16_1.h5")

    if GenData:
        x_new, y_new = data_generator()
    else:
        x_new, y_new = data_loader()

    lenc = MyLabelBinarizer()
    Y = lenc.fit_transform(y_new)
    x_train, x_test, y_train, y_test = train_test_split(x_new, Y, test_size=0.10)
    train_gen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90
    )
    train_data = train_gen.flow(
        x=x_train,
        y=y_train
    )
    test_gen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90
    )
    test_data = test_gen.flow(
        x=x_test,
        y=y_test
    )
    checkpoint = ModelCheckpoint(
        "ieeercnn_vgg16_1.h5",
        monitor='val_loss',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'
    )
    early = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=100,
        verbose=1,
        mode='auto'
    )
    with tf.device('/gpu:0'):
        devices = tf.config.experimental.list_physical_devices(device_type='GPU')
        print(devices)
        for gpu in devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        hist = model_final.fit_generator(
            callbacks=[checkpoint],
            validation_data=test_data,
            validation_steps=2,
            generator=train_data,
            steps_per_epoch=10,
            epochs=1000
        )
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title("model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Loss", "Validation Loss"])
    plt.show()
    plt.savefig('chart loss.png')


def test_model_cl():
    model_final = keras.models.load_model("ieeercnn_vgg16_1.h5")
    X_new, y_new = data_loader()
    lenc = MyLabelBinarizer()
    Y = lenc.fit_transform(y_new)
    X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.10)
    for test in X_test:
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


test_model_od()
