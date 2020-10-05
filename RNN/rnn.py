import numpy as np  # for efficient matrix and vector operations
import math
import random


def relu(x):
    return np.piecewise(x, [x < 0, x >= 0], [lambda a: 0, lambda a: a])


def leaky_relu(x):
    return np.piecewise(x, [x < 0, x >= 0], [lambda a: -0.01*a, lambda a: a])


def sigmoid(x):
    return 1 / (1 + 2.71**(-x))


class RNN:

    def __init__(self, input_size, state_sizes, output_size, activation_function=relu):
        self.all_sizes = [input_size] + state_sizes + [output_size]
        self.layer_sizes = state_sizes + [output_size]

        # first in list will be inputs, last will be outputs, between is hidden state
        self.values = [np.zeros((i, 1)) for i in self.all_sizes]
        # (for training) store what the input to each layer was before applying activ. function
        self.pre_activations = [np.zeros((i, 1)) for i in self.layer_sizes]

        # bias vectors
        self.biases = [np.random.normal(
            0, 0.01, (i, 1)) for i in self.layer_sizes]

        # weight matrices
        self.forward_weights = [np.random.normal(
            0, 0.01, (self.all_sizes[i+1], self.all_sizes[i])) for i in range(0, self.all_sizes.__len__() - 1)]

        self.recurrent_weights = [np.random.normal(
            0, 0.01, (i, i)) for i in state_sizes]

        self.activation_function = activation_function

    def reset_state(self):
        for layer in self.values:
            layer = layer * 0
        for layer in self.pre_activations:
            layer = layer * 0

    def perform_timestep(self, input_vector):
        # makes it a (x, 1) shape matrix
        self.values[0] = np.transpose(np.array([input_vector]))
        for i in range(1, self.all_sizes.__len__()):
            # calculate weighted sum in from previous layer or input
            new_vals = self.biases[i-1] + \
                np.dot(self.forward_weights[i-1], self.values[i-1])
            # if this is a hidden layer, add the recurrent signal from its previous state
            if i < len(self.all_sizes) - 1:
                new_vals += np.dot(self.recurrent_weights[i-1], self.values[i])
            # apply activation function
            self.pre_activations[i-1] = new_vals
            self.values[i] = self.activation_function(new_vals)

    def predict(self):
        return self.values[len(self.values)-1]

    def sample_text(self, charset, temp, length=100):
        sample_str = ""
        self.perform_timestep([0] * self.all_sizes[0])
        for i in range(0, length):
            raw = self.predict()
            summed_exp = np.sum(math.e ** (raw * 1.0 / temp))
            softmax = math.e ** (raw * 1.0 / temp) / summed_exp
            pick_float = random.uniform(0, 1)
            pick_index = -1
            counter = 0.0
            new_input = []
            for j in range(0, len(raw)):
                if pick_float > counter and pick_float <= counter+softmax[j]:
                    pick_index = j
                    new_input.append(1)
                else:
                    new_input.append(0)
                counter += softmax[j]
            sample_str += charset[pick_index]
            self.perform_timestep(new_input)

        return sample_str


if __name__ == '__main__':
    # myRnn = RNN(2, [3, 2], 3)
    # myRnn.perform_timestep([0.5, 0.4])
    # print(myRnn.values)
    matrix1 = np.ones((2, 1))
    matrix2 = np.ones((1, 1))
    print(np.dot(matrix1, [[0]]).shape)
