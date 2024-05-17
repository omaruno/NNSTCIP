import tensorflow as tf
import numpy as np
import tensorflow as tf

class SeparableConv1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='same', depth_multiplier=1, activation=None):
        super(SeparableConv1D, self).__init__()
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D((kernel_size, 1), strides=(strides, 1), padding=padding, depth_multiplier=depth_multiplier, activation=activation)
        self.pointwise_conv = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', activation=activation)

    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=-1)  # expand dimensions for Conv2D
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return tf.squeeze(x, axis=-1)  # squeeze to return to 1D
import numpy as np
import tensorflow as tf

class XceptionModule(tf.keras.layers.Layer):
    def __init__(self, ni, nf, bottleneck=False, **kwargs):
        super(XceptionModule, self).__init__(**kwargs)
        ks = [3, 5, 7]
        self.convs = [tf.keras.layers.Conv1D(nf if bottleneck else ni, nf, k, padding='same') for k in ks]
        self.concat = tf.keras.layers.Concatenate()

    def call(self, inputs):
        x = self.concat([l(inputs) for l in self.convs] + [inputs])
        return x

class XceptionBlock(tf.keras.layers.Layer):
    def __init__(self, c_in, nf, **kwargs):
        super(XceptionBlock, self).__init__(**kwargs)
        self.xception = [XceptionModule(ni if i == 0 else nf * 2**(i-1), nf * 2**i) for i in range(4)]

    def call(self, inputs):
        x = inputs
        for layer in self.xception:
            x = layer(x)
        return x

class XceptionTime(tf.keras.Model):
    def __init__(self, c_in, c_out, **kwargs):
        super(XceptionTime, self).__init__(**kwargs)
        self.block = XceptionBlock(c_in, 16)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(c_out, activation='softmax')

    def call(self, inputs):
        x = self.block(inputs)
        x = self.flatten(x)
        x = self.dense(x)
        return x

# Example usage:
X = np.random.randn(7, 5, 5)  # Example input data
y = np.random.randint(0, 4, size=7)  # Example output data
model = XceptionTime(c_in=5, c_out=4)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10)
