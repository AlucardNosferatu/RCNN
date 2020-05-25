import tensorflow.keras.backend as K
import tensorflow as tf
from Config import CheckLoss
from tensorflow.keras.losses import Loss

HUBER_DELTA = 0.5


def ROILoss(y_true, y_pred):
    if CheckLoss:
        tf.print()
        tf.print(y_true)
        tf.print(y_pred)
    x = K.abs(y_true - y_pred)
    x = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    result = K.sum(x)
    if CheckLoss:
        tf.print(result)
    return result
