import tensorflow.keras.backend as K
import tensorflow as tf

HUBER_DELTA = 0.5


def RPNLoss(y_true, y_pred):
    total_loss = 0
    for i in range(y_pred.shape[1]):
        temp = ROILoss(y_true[:, i, :], y_pred[:, i, :])
        total_loss += temp
    tf.print(total_loss)
    total_loss /= y_pred.shape[1]
    tf.print(total_loss)
    tf.print()
    return total_loss


def ROILoss(single_y_true, single_y_pred):
    x = K.abs(single_y_true - single_y_pred)
    x = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    result = K.sum(x)
    return result
