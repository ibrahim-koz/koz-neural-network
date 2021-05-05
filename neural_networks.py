import numpy as np
class NeuralNetwork:
    def __init__(self, loss_function, d_loss_function):
        self.layers = []
        self.loss_function = loss_function
        self.d_loss_function = d_loss_function

    def add_layer(self, layer):
        self.layers.append(layer)

    def predict(self, input_matrix):
        results = []
        for input_row in input_matrix:
            output = input_row
            for layer in self.layers:
                output = layer.forward_propagate(output)
            results.append(output)
        return results

    def fit(self, x_train, y_train, epoch_number, initial_learning_rate, decay):
        for i in range(epoch_number):
            cost = 0
            learning_rate = initial_learning_rate * (1 / (1 + decay * i))
            for input_row, output_row in zip(x_train, y_train):
                # forward propagation
                propagating_input = input_row
                for layer in self.layers:
                    propagating_input = layer.forward_propagate(propagating_input)
                predicted_value = propagating_input

                # compute loss (for display purpose only)
                cost += self.loss_function(output_row, predicted_value)

                # backward propagation
                d_error = self.d_loss_function(output_row, predicted_value)
                # we need the following line to compute our varying derivative.
                # this value
                for layer in reversed(self.layers):
                    d_error = layer.backward_propagate(d_error, learning_rate)

            cost /= len(x_train)
            print('epoch %d/%d   error=%f' % (i + 1, epoch_number, cost))