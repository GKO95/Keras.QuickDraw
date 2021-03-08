import tensorflow as tf
import tensorflow.keras as keras

""" 2D CONVOLUTION LAYER 
> Reference: https://en.wikipedia.org/wiki/Convolutional_neural_network#Convolutional_layer

The Conv2D layer has a depth of filters (aka. a number of kernels; 32 or 64) that 
detects certain pixel pattern using convolution operation. These kernels stride/shift 
along the input data pixels-by-pixels, resulting a single scalar value:

    Conv2D Operation -> Summation of element-wise multiplication

Hence, the portion of the input data that has less similar traits to the pattern
results relatively low value. And higher value means higher similarity to the kernel.


The output size is determined by the following formula:
    * Input: Input size
    * Kernel: Kernel size
    * Stride: Pixels to stride/shift (generally stride < 3)
    * Padding: 0 when padding is set to 'valid'

    Output: 
      => size: [(Input - Kernel + (2 * Padding)) / Stride] + 1
      => shape: (size, size, depth)
      ...but if the output size is not an integer, the strides are incorrect!

    Example:
      >> Input = (28, 28), Depth = 32, Kernel = (5, 5), Stride = (1, 1)...
      => size: [(28 - 5 + (2 * 0)) / 1] + 1 = 24
      => shape: (24, 24, 32)


Meanwhile, when padding is set as the 'same', the input data is padded with zeros
to make the output data have the same size as the input data (right & bottom has 
a higher priority than left & top).

Thus, the output size calculation formula is as follows:
    Output:
      => size: Input / Stride
      => shape: (size, size, depth)


After the operation, there will be total 32 matrixes sized 24 x 24, containing
scalar values from the convolution operaiton (as mentioned on the output shape).
The Conv2D may return these matrixes AS THEY ARE (= feature maps), or return after
processing them with activation function (= activation maps).

    Types: Sigmoid, Hyperbolic tangent (tanh), ReLU, etc...

ReLU activation function is currently best used on CNN model or probably in 
general cases. This is because of the nature where ReLU calculation is simple 
and inexpensive as it only has derivative of 0 and 1.


In summary, when this Conv2D layer accepts a single 28 x 28 of grayscale data, 
it returns 32 of 24 x 24 activation maps. Each activation map represents the
pattern trait similarity measurement to its corresponding kernel; higher the
value indicates that's where the kernel pattern could be found!
"""
layerConv2D_1 = tf.keras.layers.Conv2D(
    filters = 32,               # The number of filter in the Conv2D layer: 32
    kernel_size = (5, 5),       # The size of the filter in the Conv2D layer.
    strides = (1, 1),           # The amount of pixels to shift the kernel.
    padding = 'valid',          # Leave the input data AS IS wihtout any padding ('valid').
    activation = 'relu',        # Activation function (default: None).
    input_shape = (28, 28, 1),  # First Conv2D must specify the shape of input: 28 x 28 Grayscale channel 
)

""" 2D MAX POOLING LAYER
The MaxPool2D layer works similar to Conv2D; when kernel operates convolution
that gives a single scalar, MaxPooling gives a MAX value within the window.
This compresses the activation maps that only leave the significants.

MaxPooling layer does not increase the depth of the activation maps. It only
shrinks the size of currently existing activation maps.


The calculation formual on deriving output size is the same as Conv2D's since
the fact that the pooling operation returns a single scalar is the same:
    * Input: Input size
    * Window: Pooling Window size (generally 2 x 2)
    * Stride: Pixels to stride/shift (default: size of the Pooling Window)
    * Padding: 0 when padding is set to 'valid'

    Output: 
      => size: [(Input - Kernel + (2 * Padding)) / Stride] + 1
      => shape: (size, size, depth)
      ...but if the output size is not an integer, the strides are incorrect!

    Example:
      >> Input = (24, 24), Window = (2, 2), Stride = (2, 2)...
      => size: [(24 - 2 + (2 * 0)) / 2] + 1 = 12
      => shape: (12, 12, 32)
"""
layerMaxPooling2D_1 = tf.keras.layers.MaxPool2D(
    pool_size = (2, 2),         # The size of a pooling window in the Conv2D layer.
    strides = (2, 2),           # The amount of pixels to shift a pooling window.
    padding = 'valid',          # Leave the input data AS IS wihtout any padding ('valid').
)

""" OUTPUT
      >> Input = (12, 12), Depth = 64, Kernel = (3, 3), Stride = (1, 1)...
      => size: [(12 - 3 + (2 * 0)) / 1] + 1 = 10
      => shape: (10, 10, 64)

DO NOT MISTAKEN: this does not mean the overall activation maps have increased
from 32 to 64 matrixes. The 64 output depth is "per" input data. Because there
were previously 32 activation maps from 'layerConv2D_1', each activation maps 
having its own 64 activation maps from 'layerConv2D_2' makes 32 * 64 = 2048 
activation maps in total.

However, what's important is not the number of total activation maps.
It is the output shape, eg. (10, 10, 64) that matters the most!
"""
layerConv2D_2 = tf.keras.layers.Conv2D(
    filters = 64,
    kernel_size = (3, 3),
)

""" OUTPUT
      >> Input = (10, 12), Window = (2, 2), Stride = (2, 2)...
      => size: [(10 - 2 + (2 * 0)) / 2] + 1 = 5
      => shape: (5, 5, 64)
"""
layerMaxPooling2D_2 = tf.keras.layers.MaxPool2D(
    pool_size = (2, 2),
)

""" OUTPUT
      >> Input = (5, 5), Depth = 64, Kernel = (3, 3), Stride = (1, 1)...
      => size: [(5 - 3 + (2 * 0)) / 1] + 1 = 3
      => shape: (3, 3, 64)

In this case, there is no MaxPool2D layer followed by this Conv2D layer.
It does not matter since MaxPool2D is just to compress the size of activation maps.

TOTAL ACTIVATION MAPS: 32 -> 2048 (= 32 * 64) -> 131072 (= 32 * 64 * 64)
Remember, what is more important is not the total number of activation maps
but the output shape from the 'layerConv2D_3' Conv2D layer.
"""
layerConv2D_3 = tf.keras.layers.Conv2D(
    filters = 64,
    kernel_size= (3, 3),
)

""" FLATTEN LAYER
The Flatten layer compresses not just the size of the input data (that is,
output data from 'layerConv2D_3' layer) but "reshapes" to a single dimensional
tensor for a Dense layer.

Considering the output shape from 'layerConv2D_3' layer was (None, 3, 3, 64),
the layer flattens the tensor to (None, 576) which is 3 * 3 * 64 = 576.

Here, the None in the very front of the tensor shape indicates the support for
any batch size (dynamic)!
"""
layerFlatten = tf.keras.layers.Flatten()

""" DENSE LAYER
The Dense (aka. fully-connected) layer is where the training parameters and
forward propagation for hypothesis/prediction occurs. Previous layers
(eg. Conv2D, MaxPool2D, Flatten) are just to extract the characteristics & traits
for image classification that is stored in activation maps.
"""
layerDense_1 = tf.keras.layers.Dense(
    units = 192,
    activation = 'relu',
)

layerDense_2 = tf.keras.layers.Dense(
    units = 64,
    activation = 'relu',
)

layerSoftmax = tf.keras.layers.Dense(
    units = 30,
    activation = 'softmax',
)

""" SEQUENTIAL MODEL

"""
model = tf.keras.models.Sequential(
    layers = [
        # INPUT LAYER
        layerConv2D_1,
        layerMaxPooling2D_1,
        layerConv2D_2,
        layerMaxPooling2D_2,
        layerConv2D_3,
        layerFlatten,
        # HIDDEN LAYER
        layerDense_1,
        layerDense_2,
        # OUTPUT LAYER
        layerSoftmax,
    ],
    name = "QModel",
)

""" TRAINING MODEL """
if __name__ == "__main__":
    model.summary()
