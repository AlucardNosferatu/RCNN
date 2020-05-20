import tensorflow as tf
from tensorflow.keras.layers import Layer


class RoiPoolingConv(Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, rows, cols, channels)`
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    '''

    def __init__(self, pool_size, num_rois, **kwargs):
        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        assert (len(x) == 2)

        # x[0] is image with shape (rows, cols, channels)
        img = x[0]

        # x[1] is roi with shape (num_rois,4) with ordering (x,y,w,h)
        rois = x[1]

        input_shape = img.shape

        outputs = []

        for roi_idx in range(self.num_rois):
            x1 = rois[0, roi_idx, 0]
            y1 = rois[0, roi_idx, 1]
            x2 = rois[0, roi_idx, 2]
            y2 = rois[0, roi_idx, 3]

            x1 *= input_shape[1]
            y1 *= input_shape[2]
            x2 *= input_shape[1]
            y2 *= input_shape[2]

            x1 = tf.cast(x1, 'int32')
            y1 = tf.cast(y1, 'int32')
            x2 = tf.cast(x2, 'int32')
            y2 = tf.cast(y2, 'int32')

            # Resized roi of the image to pooling size (7x7)
            rs = tf.compat.v1.image.resize_images(img[:, x1:x2, y1:y2, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = tf.keras.backend.concatenate(outputs, axis=0)

        # Reshape to (1, num_rois, pool_size, pool_size, nb_channels)
        # Might be (1, 4, 7, 7, 3)
        final_output = tf.keras.backend.reshape(final_output,
                                                (-1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        # permute_dimensions is similar to transpose
        final_output = tf.keras.backend.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RoiPoolingConvDep(Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, rows, cols, channels)`
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    '''

    def __init__(self, pool_size, num_rois, **kwargs):
        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        assert (len(x) == 2)

        # x[0] is image with shape (rows, cols, channels)
        img = x[0]

        # x[1] is roi with shape (num_rois,4) with ordering (x,y,w,h)
        rois = x[1]

        input_shape = img.shape

        outputs = []

        for roi_idx in range(self.num_rois):
            x1 = rois[0, roi_idx, 0]
            y1 = rois[0, roi_idx, 1]
            x2 = rois[0, roi_idx, 2]
            y2 = rois[0, roi_idx, 3]

            x1 *= input_shape[1]
            y1 *= input_shape[2]
            x2 *= input_shape[1]
            y2 *= input_shape[2]
            # tf.print(x2-x1)
            # tf.print(y2-y1)
            # tf.cast cut off fraction part of float number: 4.7 → 4
            x_d = tf.cast(tf.round(x2-x1), 'int32')
            y_d = tf.cast(tf.round(y2-y1), 'int32')
            # tf.round is used for distance but not coordinates:
            # x1=4.5≈x1'=5
            # x2=5.1≈x2'=5
            # x2'-x1'=0
            # x2-x1=0.6≈1
            x1 = tf.cast(x1, 'int32')
            y1 = tf.cast(y1, 'int32')
            x2 = x1 + x_d
            y2 = y1 + y_d
            # tf.print(x2-x1)
            # tf.print(y2-y1)
            # tf.print(img.shape)
            # tf.print()
            # Resized roi of the image to pooling size (7x7)
            rs = tf.compat.v1.image.resize_images(img[:, x1:x2, y1:y2, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = tf.keras.backend.concatenate(outputs, axis=0)

        # Reshape to (1, num_rois, pool_size, pool_size, nb_channels)
        # Might be (1, 4, 7, 7, 3)
        final_output = tf.keras.backend.reshape(final_output,
                                                (-1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        # permute_dimensions is similar to transpose
        final_output = tf.keras.backend.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
