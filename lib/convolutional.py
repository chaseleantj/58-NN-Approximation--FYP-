import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

class Layer_Flatten:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.input_shape = np.shape(self.inputs)
        self.output = np.reshape(inputs, (np.shape(inputs)[0], -1))
        self.output_shape = np.shape(self.output)
        return self.output
    def backward(self, dinputs):
        self.dinputs = np.reshape(dinputs, self.input_shape)
        return self.dinputs

# https://towardsdatascience.com/forward-and-backward-propagation-of-pooling-layers-in-convolutional-neural-networks-11e36d169bec

class Layer_Pooling:
    def __init__(self, pool_width=2, pool_height=2, stride=2):
        self.pool_width = pool_width
        self.pool_height = pool_height
        self.stride = stride
    
    def forward(self, inputs, training):
        '''
        N is the number of inputs
        C is the number of channels
        H is the height is the image
        W is the width of the image
        '''
        self.inputs = inputs
        N, C, H, W = np.shape(inputs)

        H_out = int(1 + (H - self.pool_height) / self.stride)
        W_out = int(1 + (W - self.pool_width) / self.stride)

        self.output = np.zeros((N, C, H_out, W_out))

        for n in range(N):
            for c in range(C):
                for hi in range(H_out):
                    for wi in range(W_out):
                        self.output[n, c, hi, wi] = np.max(inputs[n, c, hi * self.stride : hi * self.stride + self.pool_height, wi * self.stride : wi * self.stride + self.pool_width ])

        return self.output
    
    def backward(self, dvalues):

        N, C, H_out, W_out = np.shape(self.output)
        self.dinputs = np.zeros_like(self.inputs)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        # get the index in the region i,j where the value is the maximum
                        i_t, j_t = np.where(np.max(self.inputs[n, c, i * self.stride : i * self.stride + self.pool_height, j * self.stride : j * self.stride + self.pool_width]) == self.inputs[n, c, i * self.stride : i * self.stride + self.pool_height, j * self.stride : j * self.stride + self.pool_width])
                        i_t, j_t = i_t[0], j_t[0]
                        # only the position of the maximum element in the region i,j gets the incoming gradient, the other gradients are zero
                        self.dinputs[n, c, i * self.stride : i * self.stride + self.pool_height, j * self.stride : j * self.stride + self.pool_width][i_t, j_t] = dvalues[n, c, i, j]
        return self.dinputs

class Layer_Convolutional:
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.weights = np.random.normal(0, np.sqrt(2 / (input_depth * input_width * input_height)), self.kernels_shape) # weights is the kernel
        # self.weights = 0.1 * np.random.randn(*self.kernels_shape) # weights is the kernel
        self.biases = np.zeros(self.output_shape)

        self.weight_regularizer_l1 = 0
        self.weight_regularizer_l2 = 0
        self.bias_regularizer_l1 = 0
        self.bias_regularizer_l2 = 0

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.concatenate([[self.biases]] * len(self.inputs), axis=0)
        for i in range(len(self.inputs)):
            for j in range(self.depth):
                for k in range(self.input_depth):
                    self.output[i][j] += signal.correlate2d(self.inputs[i][k], self.weights[j, k], "valid")
        return self.output

    def backward(self, dvalues):
        self.dweights = np.zeros(self.kernels_shape)
        self.dbiases = np.sum(dvalues, axis=0)
        self.dinputs = np.zeros_like(self.inputs)

        for i in range(len(self.inputs)):
            for j in range(self.depth):
                for k in range(self.input_depth):
                    self.dweights[j, k] = signal.correlate2d(self.inputs[i][k], dvalues[i][j], "valid")
                    self.dinputs[i][k] = self.dinputs[i][k] + signal.convolve2d(dvalues[i][j], self.weights[j, k], "full")

        return self.dinputs

# https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f381
def correlate2d(image, kernel, padding=0, strides=1):

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        imagePadded = image
    plt.imshow(imagePadded)
    plt.show()
    # Iterate through image
    for y in range(imagePadded.shape[1]):
        # Exit Convolution
        if y > imagePadded.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(imagePadded.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > imagePadded.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

# https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f381
# Not used; the scipy correlate2d is about 100x faster using the FFT
def correlate2d(image, kernel, padding=0, strides=1):

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        imagePadded = image

    # Iterate through image
    for y in range(imagePadded.shape[1]):
        # Exit Convolution
        if y > imagePadded.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(imagePadded.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > imagePadded.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

def convolve2d(image, kernel, padding=0, strides=1):
    kernel = np.flipud(np.fliplr(kernel))
    output = correlate2d(image, kernel, padding, strides)
    return output