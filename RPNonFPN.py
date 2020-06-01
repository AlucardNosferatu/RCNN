import tensorflow as tf
from tensorflow.keras.models import Model
from FPN import FPN, FPN_BN_Interface
from RPN import LFM_model_build
from utils import loss_cls, smoothL1


def build_fpn_rpn():
    model_cnn = tf.keras.applications.VGG16(weights='imagenet', include_top=True)
    model_cnn.trainable = False
    fpn = FPN()
    fpn_result = FPN_BN_Interface(fpn=fpn, backbone=model_cnn)
    rpn_outs = []
    for i in range(len(fpn_result)):
        rpn_out = LFM_model_build(i + 1, fpn_result[i])
        rpn_outs += rpn_out
    model_fpn_rpn = Model(inputs=model_cnn.input, outputs=rpn_outs)
    model_fpn_rpn.compile(
        optimizer='adam',
        loss={
            'scores1': loss_cls,
            'deltas1': smoothL1,
            'scores2': loss_cls,
            'deltas2': smoothL1,
            'scores3': loss_cls,
            'deltas3': smoothL1
        }
    )
    model_fpn_rpn.summary()
    model_fpn_rpn.save("TrainedModels\\FPN_RPN.h5py")
    return model_fpn_rpn


def load_model():
    model_fpn_rpn = tf.keras.models.load_model(
        "TrainedModels\\FPN_RPN.h5py",
        custom_objects={
            'loss_cls': loss_cls,
            'smoothL1': smoothL1
        }
    )
    return model_fpn_rpn


build_fpn_rpn()
