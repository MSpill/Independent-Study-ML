import numpy as np  # for efficient matrix and vector operations


class RNN:

    def __init__(self, input_size, state_sizes, output_size):
        self.all_sizes = [input_size] + state_sizes + [output_size]
        self.layer_sizes = state_sizes + [output_size]

        # first in list will be inputs, last will be outputs, between is hidden state
        self.values = [np.zeros((i, 1)) for i in self.all_sizes]

        # bias vectors
        self.biases = [np.random.normal(
            0, 0.01, (i, 1)) for i in self.layer_sizes]

        # weight matrices
        self.forward_weights = [np.random.normal(
            0, 0.01, (self.all_sizes[i+1], self.all_sizes[i])) for i in range(0, self.all_sizes.__len__() - 1)]

        self.recurrent_weights = [np.random.normal(
            0, 0.01, (i, i)) for i in state_sizes]

    def set_input(self, input_vector):
        self.values[0] = input_vector

    def perform_timestep(self):
        for i in range(1, self.all_sizes.__len__()):
            # calculate weighted sum in from previous layer or input
            new_vals = self.biases[i-1] + \
                np.dot(self.forward_weights[i-1], self.values[i-1])
            # if this is a hidden layer, add the recurrent signal from its previous state
            if i < len(self.layer_sizes)-1:
                new_vals += np.dot(self.recurrent_weights[i-1], self.values[i])
            # apply activation function
            self.values[i] = sigmoid(new_vals)


def sigmoid(x):
    return 1 / (1 + 2**(-x))


if __name__ == '__main__':
    myRnn = RNN(10, [5, 4], 7)
    myRnn.perform_timestep()
    print(myRnn.values)
