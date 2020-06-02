import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Add, UpSampling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from RCNN import OneHot, data_generator, data_loader, getROIs_fromRPN, CheckBatch
from utils import RPN_load, Activate_GPU


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


def EachRouteAfterFPN(FPN_FM, index):
    x = Conv2D(64, (3, 3), padding='same')(FPN_FM)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Flatten(name='flat_afterFM_' + str(index))(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    return x


def ClassifierBlockAfterFPN(fpn_result):
    roi_result = []
    for i in range(0, len(fpn_result)):
        x = EachRouteAfterFPN(fpn_result[i], i)
        roi_result.append(x)
    x = tf.concat(roi_result, axis=1)
    x = BatchNormalization()(x)
    x = Dense(2, activation="softmax")(x)
    return x


def FPN_BN_Interface(fpn, backbone):
    fm_layer_indices = [10, 14, 18]
    fpn_input = []
    for each in fm_layer_indices:
        fpn_input.append(backbone.layers[each].output)
    fpn_result = fpn(fpn_input)
    return fpn_result


def FPN_build():
    vgg_model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)
    fpn = FPN()
    fpn_result = FPN_BN_Interface(fpn, vgg_model)
    for layers in vgg_model.layers[:18]:
        layers.trainable = False
    x = ClassifierBlockAfterFPN(fpn_result)
    model_final = Model(inputs=vgg_model.input, outputs=x)
    opt = Adam(lr=0.0001)
    model_final.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=opt,
        metrics=["accuracy"]
    )
    return model_final


def FPN_train(NewModel=False, GenData=False):
    batch_size = 8
    if NewModel:
        model_final = FPN_build()
    else:
        model_final = tf.keras.models.load_model("TrainedModels\\FPN_Prototype.h5py")
    tf.keras.utils.plot_model(
        model_final, to_file='model.png', show_shapes=True, show_layer_names=True,
        rankdir='TB', expand_nested=True, dpi=96
    )
    if GenData:
        x_new, y_new = data_generator(balance=0.8)
    else:
        x_new, y_new = data_loader()
    one_hot = OneHot()
    y = one_hot.fit_transform(y_new)
    x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size=0.25)
    del x_new, y, y_new
    train_gen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90
    )
    train_data = train_gen.flow(
        batch_size=batch_size,
        x=x_train,
        y=y_train
    )
    test_gen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90
    )
    test_data = test_gen.flow(
        batch_size=batch_size,
        x=x_test,
        y=y_test
    )
    checkpoint = ModelCheckpoint(
        "TrainedModels\\FPN_Prototype.h5py",
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'
    )
    with tf.device('/gpu:0'):
        hist = model_final.fit_generator(
            callbacks=[checkpoint],
            validation_data=test_data,
            validation_steps=10,
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
    model_final = tf.keras.models.load_model("TrainedModels\\FPN_Prototype.h5py")
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


def test_model_od(UseRPN=False, InspectEach=False, InspectNeg=False, SizeFilter=None):
    threshold = 0.001
    path = "ProcessedData\\Images"
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    model_fpn = tf.keras.models.load_model("TrainedModels\\FPN_Prototype.h5py")
    if UseRPN:
        model_rpn = RPN_load("TrainedModels\\RPN_Prototype_28X28.h5")
        print("RPN has been loaded.")
        backbone = Model(inputs=model_fpn.input, outputs=model_fpn.layers[13].output)
        print("Backbone network has been loaded.")
    z = 0
    for e, i in enumerate(os.listdir(path)):
        if i.startswith("airplane"):
            print(i)
            z += 1
            img = cv2.imread(os.path.join(path, i))
            image_out = img.copy()
            if UseRPN:
                ss_results = getROIs_fromRPN(img, model_rpn, backbone)
            else:
                ss.setBaseImage(img)
                ss.switchToSelectiveSearchFast()
                ss_results = ss.process()

            for roi, result in enumerate(ss_results):
                if roi < len(ss_results):
                    if UseRPN:
                        x1, y1, x2, y2 = result
                    else:
                        x, y, w, h = result
                        x1 = x
                        y1 = y
                        x2 = x + w
                        y2 = y + h
                        if SizeFilter and (w > SizeFilter or h > SizeFilter):
                            print("Size exceeds limit, skip this ROI.")
                            continue
                    target_image = image_out[y1:y2, x1:x2]
                    resized = cv2.resize(target_image, (224, 224), interpolation=cv2.INTER_AREA)
                    resized = np.expand_dims(resized, axis=0)
                    out = model_fpn.predict(resized)
                    # positive = (out[0][0] - out[0][1]) > threshold
                    positive = out[0][0] > out[0][1]
                    if positive:
                        str_result = "plane     " + str(out)
                    else:
                        str_result = "not plane " + str(out)
                    if InspectEach:
                        if positive or InspectNeg:
                            plt.figure()
                            plt.title(str_result)
                            plt.imshow(resized.reshape((224, 224, 3)))
                            plt.show()
                    if positive:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
            plt.figure()
            plt.imshow(img)
            plt.savefig("TestResults\\" + i + "_od_test.jpg")
            # plt.show()
            plt.close()


# Activate_GPU()
# train(NewModel=True)
# data_generator(UseRPN=False, balance=0.95)
# CheckBatch(ShowNeg=False)
# test_model_od()
