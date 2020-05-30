import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from RCNN import get_iou
from RPN_Sample.RPN_Sample_Caller import RPN_load, RPN_forward, select_proposals
from RPN_Sample.utils import Activate_GPU, loss_cls, smoothL1
from Obsolete.CustomRPN import data_loader


def standard_model_test():
    rpn_model = RPN_load(file_path="TrainedModels\\RPN_Prototype.h5")
    backbone_network = VGG16(include_top=True, weights="imagenet")
    backbone_network = Model(inputs=backbone_network.input, outputs=backbone_network.layers[17].output)
    x_new, y_new = data_loader()
    for i in range(0, x_new.shape[0], 5):
        count = 0
        image_out = x_new[i, :, :, :]
        feature_map = backbone_network.predict(np.expand_dims(image_out, axis=0) / 255)
        proposals, scores = RPN_forward(rpn_model=rpn_model, feature_map=feature_map)
        proposals, scores = select_proposals(scores=scores, proposals=proposals, AutoSelection=0.25)
        gt_values = []
        for j in range(y_new.shape[1]):
            x1 = int(y_new[i, j, 0])
            y1 = int(y_new[i, j, 1])
            x2 = int(y_new[i, j, 2])
            y2 = int(y_new[i, j, 3])
            if x1 == x2 or y1 == y2:
                print("zero area error!")
                continue
            gt_values.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
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
            if max(iou_list) > 0.25:
                count += 1
                image_out = cv2.rectangle(image_out, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
            else:
                image_out = cv2.rectangle(image_out, (x1, y1), (x2, y2), (255, 0, 0), 1, cv2.LINE_AA)
        plt.figure()
        plt.imshow(image_out)
        plt.savefig("TestResults\\第" + str(i) + "次测试，命中比例：" + str(int(100 * count / proposals.shape[0])) + "%.jpg")
        plt.close()


# LFM means Larger Feature Map
def LFM_model_build():
    k = 9
    # region RPN Model
    feature_map_tile = Input(shape=(None, None, 512))
    convolution_3x3 = Conv2D(
        filters=512,
        kernel_size=(3, 3),
        padding='same',
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
    model_rpn.compile(optimizer='adam', loss={'scores1': loss_cls, 'deltas1': smoothL1})
    return model_rpn
    # endregion


model_cnn = tf.keras.models.load_model("TrainedModels\\RCNN.h5")
print("Classifier has been loaded.")
backbone = Model(inputs=model_cnn.input, outputs=model_cnn.layers[17].output)
print("Backbone network has been loaded.")
