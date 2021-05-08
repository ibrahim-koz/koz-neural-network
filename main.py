import pickle

import numpy as np
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from activation_functions import argmax, swish_prime, swish, log_softmax_prime, log_softmax, tanh, tanh_prime, softmax, \
    softmax_prime, relu, relu_prime
from layers import ActivationLayer
from layers import FCLayer
from loss_functions import cross_entropy, cross_entropy_prime, mse_prime, mse
from neural_networks import NeuralNetwork

x_train_in = open("feature_extracted_data/x_train.pickle", "rb")
x_train = pickle.load(x_train_in)
del x_train_in

y_train_in = open("feature_extracted_data/y_train.pickle", "rb")
y_train = pickle.load(y_train_in)
del y_train_in

x_validation_in = open("feature_extracted_data/x_validation.pickle", "rb")
x_validation = pickle.load(x_validation_in)
del x_validation_in

y_validation_in = open("feature_extracted_data/y_validation.pickle", "rb")
y_validation = pickle.load(y_validation_in)
del y_validation_in

one_hot_encoded_y_train = np_utils.to_categorical(y_train)

batch_size = 128
x_train_batches = np.array_split(x_train, len(x_train) // batch_size)
y_train_batches = np.array_split(one_hot_encoded_y_train, len(one_hot_encoded_y_train) // batch_size)

# # training data
# x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
# one_hot_encoded_y_train = np.array([[[1, 0]], [[0, 1]], [[0, 1]], [[0, 1]]])
# y_train =[0, 1, 1, 1]

# this line is used to catch the errors arising from numpy.
np.seterr(all='raise')

input_number = x_train.shape[2]
output_number = 6
size_of_hidden_layer = 10

neural_network = NeuralNetwork(cross_entropy, cross_entropy_prime)
neural_network.add_layer(FCLayer(input_number, size_of_hidden_layer, diminishing_factor=10))
neural_network.add_layer(ActivationLayer(swish, swish_prime))
neural_network.add_layer(FCLayer(size_of_hidden_layer, output_number))
neural_network.add_layer(ActivationLayer(softmax, softmax_prime))

neural_network.fit(x_train, one_hot_encoded_y_train, epoch_number=10, initial_learning_rate=0.5, decay=0.01)
out = neural_network.predict(x_train)

predictions = argmax(out)
print("confusion matrix:", confusion_matrix(y_train, predictions), sep="\n")
print("accuracy: ", accuracy_score(y_train, predictions))
print("end")
