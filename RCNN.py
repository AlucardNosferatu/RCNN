import os
import cv2
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

from utils import loss_cls, smoothL1, RPN_forward, RPN_load, Activate_GPU, get_iou

path = "ProcessedData\\Images"
annotation = "ProcessedData\\Airplanes_Annotations"


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


def getROIs_fromRPN(image, model_rpn, backbone=None):
    if image.shape != (224, 224, 3):
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    if backbone:
        feature_map = backbone.predict(np.expand_dims(image, axis=0) / 255)
        result, scores = RPN_forward(rpn_model=model_rpn, feature_map=feature_map, AutoSelection=1)
    else:
        result = model_rpn.predict(np.expand_dims(image, axis=0) / 255)
        result = result.reshape((200, 4))
    result_list = []
    for i in range(result.shape[0]):
        result_list.append(list(result[i, :].reshape(4).astype('uint32')))
    return result_list


def data_generator(UseRPN=True, balance=None):
    train_images = []
    train_labels = []
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    if UseRPN:
        backbone_network = VGG16(include_top=True, weights="imagenet")
        backbone_network = Model(inputs=backbone_network.input, outputs=backbone_network.layers[13].output)
        model_final = tf.keras.models.load_model(
            "TrainedModels\\RPN_Prototype_28X28.h5",
            custom_objects={
                'loss_cls': loss_cls,
                'smoothL1': smoothL1
            }
        )
    for e, i in enumerate(os.listdir(annotation)):
        # 对每一个标记文件（csv）进行操作
        if i.startswith("airplane"):
            # 只有名称带airplane才是有目标的存在的样本
            filename = i.split(".")[0] + ".jpg"
            print(e, filename)
            image = cv2.imread(os.path.join(path, filename))
            df = pd.read_csv(os.path.join(annotation, i))
            gt_values = []
            for row in df.iterrows():
                x1 = int(row[1][0].split(" ")[0])
                y1 = int(row[1][0].split(" ")[1])
                x2 = int(row[1][0].split(" ")[2])
                y2 = int(row[1][0].split(" ")[3])
                gt_values.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
                # 把标签的坐标数据存入gt_values
            if UseRPN:
                results = getROIs_fromRPN(image, model_final, backbone_network)
            else:
                ss.setBaseImage(image)
                ss.switchToSelectiveSearchFast()
                results = ss.process()
            # 加载SS ROI提出器

            image_out = image.copy()
            counter = 0
            false_counter = 0
            flag = 0
            f_flag = 0
            b_flag = 0
            for e_roi, result in enumerate(results):
                try:
                    if e_roi < 200 and flag == 0:
                        # 对SS产生的头2k个结果（坐标）进行处理
                        for gt_val in gt_values:
                            # 对这张图上多个标签坐标进行处理
                            if UseRPN:
                                x1, y1, x2, y2 = result
                            else:
                                x, y, w, h = result
                                assert w > 0
                                assert h > 0
                                x1 = x
                                y1 = y
                                x2 = x + w
                                y2 = y + h
                            iou = get_iou(gt_val, {"x1": x1, "x2": x2, "y1": y1, "y2": y2})
                            # 计算候选坐标和这一标签坐标的交并比
                            if counter < 5:
                                # 选择交并比大于阈值的头30个候选坐标
                                if iou > 0.70:
                                    # 交并比阈值0.7
                                    target_image = image_out[y1:y2, x1:x2]
                                    resized = cv2.resize(target_image, (224, 224), interpolation=cv2.INTER_AREA)
                                    train_images.append(resized)
                                    train_labels.append(1)
                                    counter += 1
                            else:
                                f_flag = 1
                            if false_counter < 5:
                                # IoU低于阈值0.3，前30个坐标作为负样本（背景）
                                if iou < 0.3:
                                    target_image = image_out[y1:y2, x1:x2]
                                    resized = cv2.resize(target_image, (224, 224), interpolation=cv2.INTER_AREA)
                                    train_images.append(resized)
                                    train_labels.append(0)
                                    false_counter += 1
                            else:
                                b_flag = 1
                        if f_flag == 1 and b_flag == 1:
                            # print("inside")
                            flag = 1
                except Exception as e:
                    print(repr(e))
                    print("error in " + filename + "_" + str(e_roi))
                    continue
    if balance:
        while train_labels.count(0) > balance * len(train_labels):
            print("Negative ratio: " + str(int(100 * train_labels.count(0) / len(train_labels))) + "%")
            index = train_labels.index(0)
            del train_labels[index]
            del train_images[index]
    ti_pkl = open('ProcessedData\\train_images_cnn.pkl', 'wb')
    tl_pkl = open('ProcessedData\\train_labels_cnn.pkl', 'wb')
    pickle.dump(train_images, ti_pkl)
    pickle.dump(train_labels, tl_pkl)
    ti_pkl.close()
    tl_pkl.close()
    x_new = np.array(train_images)
    y_new = np.array(train_labels)
    return x_new, y_new


def data_loader():
    ti_pkl = open('ProcessedData\\train_images_cnn.pkl', 'rb')
    tl_pkl = open('ProcessedData\\train_labels_cnn.pkl', 'rb')
    train_images = pickle.load(ti_pkl)
    train_labels = pickle.load(tl_pkl)
    ti_pkl.close()
    tl_pkl.close()
    state = np.random.get_state()
    np.random.shuffle(train_images)
    np.random.set_state(state)
    np.random.shuffle(train_labels)
    x_new = np.array(train_images)
    y_new = np.array(train_labels)
    return x_new, y_new


def RCNN_train(NewModel=False, GenData=False):
    if NewModel:
        vgg_model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)
        for layers in vgg_model.layers[:15]:
            layers.trainable = False
        x = vgg_model.layers[-2].output
        predictions = Dense(2, activation="softmax")(x)
        model_final = Model(inputs=vgg_model.input, outputs=predictions)
        opt = Adam(lr=0.0001)
        model_final.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=opt,
            metrics=["accuracy"]
        )
    else:
        model_final = tf.keras.models.load_model("TrainedModels\\RCNN.h5")
    if GenData:
        x_new, y_new = data_generator(UseRPN=True, balance=True)
    else:
        x_new, y_new = data_loader()
    one_hot = OneHot()
    y = one_hot.fit_transform(y_new)
    x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size=0.10)
    train_gen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90
    )
    train_data = train_gen.flow(
        batch_size=64,
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
        "TrainedModels\\RCNN.h5",
        monitor='loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'
    )
    with tf.device('/gpu:0'):
        gpu_list = tf.config.experimental.list_physical_devices(device_type='GPU')
        print(gpu_list)
        for gpu in gpu_list:
            tf.config.experimental.set_memory_growth(gpu, True)
        hist = model_final.fit_generator(
            callbacks=[checkpoint],
            validation_data=test_data,
            validation_steps=2,
            generator=train_data,
            steps_per_epoch=100,
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
    model_final = tf.keras.models.load_model("ieeercnn_vgg16_1.h5")
    X_new, y_new = data_loader()
    lenc = OneHot()
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


def test_model_od(UseRPN=True, x_y_w_h=False, CheckTarget=False):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    model_cnn = tf.keras.models.load_model("TrainedModels\\RCNN.h5")
    print("Classifier has been loaded.")
    # custom_rpn = tf.keras.models.load_model("TrainedModels\\RPN_Prototype.h5", custom_objects={'RPNLoss': RPNLoss})
    model_rpn = RPN_load("TrainedModels\\RPN_Prototype_28X28.h5")
    print("RPN has been loaded.")
    backbone = Model(inputs=model_cnn.input, outputs=model_cnn.layers[13].output)
    print("Backbone network has been loaded.")
    z = 0
    for e, i in enumerate(os.listdir(path)):
        if i.startswith("4"):
            z += 1
            img = cv2.imread(os.path.join(path, i))
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            image_out = img.copy()
            if UseRPN:
                ss_results = getROIs_fromRPN(img, model_rpn, backbone)
            else:
                ss.setBaseImage(img)
                ss.switchToSelectiveSearchFast()
                ss_results = ss.process()
            for e_roi, result in enumerate(ss_results):
                if e_roi < 200:
                    try:
                        if x_y_w_h:
                            x, y, w, h = result
                            assert w > 0
                            assert h > 0
                            x1 = x
                            y1 = y
                            x2 = x + w
                            y2 = y + h
                        else:
                            x1, y1, x2, y2 = result
                        target_image = img[y1:y2, x1:x2]
                        resized = cv2.resize(target_image, (224, 224), interpolation=cv2.INTER_AREA)
                        resized = np.expand_dims(resized, axis=0)
                        out = model_cnn.predict(resized)
                        if CheckTarget:
                            print(out)
                            plt.imshow(resized.reshape((224, 224, 3)))
                            plt.show()
                        if out[0][0] > out[0][1]:
                            image_out = cv2.rectangle(image_out, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
                        # else:
                        #     image_out = cv2.rectangle(image_out, (x1, y1), (x2, y2), (255, 0, 0), 1, cv2.LINE_AA)
                    except Exception as e:
                        print(repr(e))
                        print("error in " + i + "_" + str(e_roi))
                        continue
            plt.figure()
            plt.imshow(image_out)
            plt.savefig("TestResults\\" + i + "_od_test.jpg")
            # plt.show()
            plt.close()


def CheckBatch(ShowNeg=True):
    x_new, y_new = data_loader()
    print("Negative Ratio: " + str(list(y_new).count(0) / y_new.shape[0]))
    for i in range(x_new.shape[0]):
        print(str(y_new[i]) + "   " + str(int(100 * i / x_new.shape[0])) + "%")
        if ShowNeg or y_new[i] == 1:
            plt.imshow(x_new[i].reshape((224, 224, 3)))
            plt.show()


# Activate_GPU()
# data_generator(UseRPN=True, balance=0.7)
# test_model_od()
# RCNN_train()
