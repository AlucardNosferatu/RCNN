import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tqdm import tqdm
from Config import batch_size, img_height, img_width, n_channels
from ROI_Pooling import RoiPoolingConv


# Create feature map input
def prototype_test():
    feature_maps_shape = (batch_size, img_height, img_width, n_channels)
    feature_maps_np = np.ones(feature_maps_shape, dtype='float32')
    feature_maps_np[0, img_height - 1, img_width - 3, 0] = 50
    roiss_np = np.asarray(
        [
            [[0.0, 0.0, 0.1, 0.1], [0.0, 0.0, 0.2, 0.2]],
            [[0.0, 0.0, 0.3, 0.3], [0.0, 0.0, 0.4, 0.4]],
            [[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.6, 0.6]]
        ],
        dtype='float32'
    )
    out_roi_pool = RoiPoolingConv(7)([feature_maps_np, roiss_np])
    print(out_roi_pool)


def test_without_dense(model, I, Y, e):
    with tf.device('/cpu:0'):
        r = model.predict([I, Y])
    r = r.reshape((7, 7, 512))
    plt.figure()
    for i in tqdm(range(0, r.shape[2])):
        plt.subplot(16, 32, i + 1)
        plt.imshow(r[:, :, i])
        plt.axis('off')
    plt.savefig("RP_TestResult\\" + str(e) + "_RP.png")
    plt.close('all')
    P1 = (
        int(Y[0][0][0] * 224),
        int(Y[0][0][1] * 224)
    )
    P2 = (
        int(Y[0][0][2] * 224),
        int(Y[0][0][3] * 224)
    )
    I = np.array(I, dtype="uint8").reshape((224, 224, 3))
    cv2.rectangle(
        I,
        P1,
        P2,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )
    plt.figure()
    plt.imshow(I)
    plt.savefig("RP_TestResult\\" + str(e) + "_CP.png")
    plt.close('all')


def test_per_layer(model, I, Y, e):
    for i in tqdm(range(19, len(model.layers))):
        model_final = Model(inputs=model.input, outputs=model.layers[i].output)
        with tf.device('/cpu:0'):
            r = model_final.predict([I, Y])
        plt.figure()
        plt.scatter(range(0, r.shape[1]), r.reshape((r.shape[1])))
        plt.savefig("RP_TestResult\\" + str(e) + "_" + str(i) + "_FoD.png")
        plt.close('all')
    P1 = (
        int(Y[0][0][0] * 224),
        int(Y[0][0][1] * 224)
    )
    P2 = (
        int(Y[0][0][2] * 224),
        int(Y[0][0][3] * 224)
    )
    I = np.array(I, dtype="uint8").reshape((224, 224, 3))
    cv2.rectangle(
        I,
        P1,
        P2,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )
    plt.figure()
    plt.imshow(I)
    plt.savefig("RP_TestResult\\" + str(e) + "_CP.png")
    plt.close('all')


prototype_test()
