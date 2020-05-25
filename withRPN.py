from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import tensorflow as tf


def prototype_model_build():
    vgg_model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)
    
