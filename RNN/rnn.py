import numpy as np  # for efficient matrix and vector operations


def relu(x):
    return np.piecewise(x, [x < 0, x >= 0], [lambda a: 0, lambda a: a])


def sigmoid(x):
    return 2 / (1 + 2**(-x)) - 1


class RNN:

    def __init__(self, input_size, state_sizes, output_size, activation_function=relu):
        self.all_sizes = [input_size] + state_sizes + [output_size]
        self.layer_sizes = state_sizes + [output_size]

        # first in list will be inputs, last will be outputs, between is hidden state
        self.values = [np.zeros((i, 1)) for i in self.all_sizes]
        # (for training) store what the input to each layer was before applying activ. function
        self.pre_activations = [np.zeros((i, 1)) for i in self.all_sizes]

        # bias vectors
        self.biases = [np.random.normal(
            0, 0.02, (i, 1)) for i in self.layer_sizes]

        # weight matrices
        self.forward_weights = [np.random.normal(
            0, 0.02, (self.all_sizes[i+1], self.all_sizes[i])) for i in range(0, self.all_sizes.__len__() - 1)]

        self.recurrent_weights = [np.random.normal(
            0, 0.02, (i, i)) for i in state_sizes]

        self.activation_function = activation_function

    def reset_state(self):
        for layer in self.values:
            layer = layer * 0

    def perform_timestep(self, input_vector):
        self.values[0] = [input_vector]  # makes it a (x, 1) shape matrix
        for i in range(1, self.all_sizes.__len__()):
            # calculate weighted sum in from previous layer or input
            new_vals = self.biases[i-1] + \
                np.dot(self.forward_weights[i-1], self.values[i-1])
            # if this is a hidden layer, add the recurrent signal from its previous state
            if i < len(self.layer_sizes)-1:
                new_vals += np.dot(self.recurrent_weights[i-1], self.values[i])
            # apply activation function
            self.pre_activations[i] = new_vals
            self.values[i] = self.activation_function(new_vals)

    def predict(self):
        return self.values[len(self.values)-1]


if __name__ == '__main__':
    # myRnn = RNN(2, [3, 2], 3)
    # myRnn.perform_timestep([0.5, 0.4])
    # print(myRnn.values)
    matrix1 = np.ones((2, 1))
    matrix2 = np.ones((1, 1))
    print(np.dot(matrix1, [[0]]).shape)
