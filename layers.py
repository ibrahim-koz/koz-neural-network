# Base class
import numpy as np

from activation_functions import tanh_prime_v2, tanh_prime


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagate(self, input_data):
        raise NotImplementedError

    def backward_propagate(self, d_error, learning_rate):
        raise NotImplementedError


class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagate(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagate(self, d_error, learning_rate):
        d_error_input = np.dot(d_error, self.weights.T)
        d_error_weight = np.dot(self.input.T, d_error)
        d_error_bias = d_error * 1  # 1 is put to make it explicit.

        # during the backward propagation, it is performed that descending.
        self.weights -= learning_rate * d_error_weight
        self.bias -= learning_rate * d_error_bias
        return d_error_input


class ActivationLayer(Layer):
    def __init__(self, activation_function, d_activation_function):
        super().__init__()
        self.activation_function = activation_function
        self.d_activation_function = d_activation_function

    def forward_propagate(self, input_data):
        self.input = input_data
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagate(self, d_error, learning_rate):
        product = np.dot(d_error, self.d_activation_function(self.input))
        return product
        # v1 = tanh_prime(self.input)
        # v2 = tanh_prime_v2(self.input)
        # res1 = np.dot(d_error, v1)
        # res2 = v2 * d_error
        # #var2 = self.d_activation_function(self.input) * d_error
        # return np.dot(d_error, self.d_activation_function(self.input))
        # #return self.d_activation_function(self.input) * d_error


class ConvLayer:
    # A Convolution layer using 3x3 filters.

    def __init__(self, num_filters):
        self.num_filters = num_filters

        # filters is a 3d array with dimensions (num_filters, 3, 3)
        # We divide by 9 to reduce the variance of our initial values
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        '''
        Generates all possible 3x3 image regions using valid padding.
        - image is a 2d numpy array
        '''
        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    def forward(self, input):
        '''
        Performs a forward pass of the conv layer using the given input.
        Returns a 3d numpy array with dimensions (h, w, num_filters).
        - input is a 2d numpy array
        '''
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        return output
