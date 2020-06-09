import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class RoiPoolingConv(Layer):

    def __init__(self, pool_size, **kwargs):
        self.pool_size = pool_size
        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]
        super(RoiPoolingConv, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return None, None, self.pool_size, self.pool_size, self.nb_channels

    def crop_and_resize(self, image, boxes):
        """Crop the image given boxes and resize with bilinear interplotation.
        # Parameters
        image: Input image of shape (1, image_height, image_width, depth)
        boxes: Regions of interest of shape (num_boxes, 4),
        each row [y1, x1, y2, x2]
        size: Fixed size [h, w], e.g. [7, 7], for the output slices.
        # Returns
        4D Tensor (number of regions, slice_height, slice_width, channels)
        """
        box_ind = tf.range(tf.shape(boxes)[0])
        box_ind = tf.reshape(box_ind, (-1, 1))
        box_ind = tf.tile(box_ind, [1, tf.shape(boxes)[1]])
        boxes = tf.keras.backend.cast(
            tf.reshape(boxes, (-1, 4)), "float32"
        )
        box_ind = tf.reshape(box_ind, (1, -1))[0]
        result = tf.image.crop_and_resize(image, boxes, box_ind, [self.pool_size, self.pool_size])
        result = tf.reshape(result, (tf.shape(image)[0], -1, self.pool_size, self.pool_size, self.nb_channels))
        return result

    def call(self, x, mask=None):
        assert (len(x) == 2)
        # x[0] is image with shape (rows, cols, channels)
        img = x[0]
        # x[1] is roi with shape (num_rois,4) with ordering (x1,y1,x2,y2)
        rois = x[1]

        input_shape = img.shape

        outputs = []

        x1 = rois[:, :, 0]
        y1 = rois[:, :, 1]
        x2 = rois[:, :, 2]
        y2 = rois[:, :, 3]

        boxes = tf.stack([y1, x1, y2, x2], axis=-1)

        # tf.keras.backend.print_tensor(x2 - x1)
        # tf.keras.backend.print_tensor(y2 - y1)
        # tf.keras.backend.print_tensor(self.pool_size)
        # tf.keras.backend.print_tensor("-----------")
        # Resized roi of the image to pooling size (7x7)
        rs = self.crop_and_resize(img, boxes)
        # final_output = tf.keras.backend.concatenate(rs, axis=0)
        #
        # # Reshape to (1, num_rois, pool_size, pool_size, nb_channels)
        # # Might be (1, 4, 7, 7, 3)
        # final_output = tf.keras.backend.reshape(final_output,
        #                                         (-1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
        #
        # # permute_dimensions is similar to transpose
        # final_output = tf.keras.backend.permute_dimensions(final_output, (0, 1, 2, 3, 4))
        return rs

    def get_config(self):
        config = {'pool_size': self.pool_size}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
