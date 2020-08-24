import numpy as np  # for efficient matrix and vector operations

# needs three vectors: input, state, and output
# needs 2 bias vectors: state and output
# needs three weight matrices: input to state (U), state to state (W), and state to output (V)


class RNN:

    def __init__(self, input_size, state_size, output_size):
        # vectors
        self.inputs = np.zeros((input_size, 1))
        self.state = np.zeros((state_size, 1))
        self.output = np.zeros((output_size, 1))

        # bias vectors
        self.state_biases = np.random.normal(0, 0.01, (state_size, 1))
        self.output_biases = np.random.normal(0, 0.01, (output_size, 1))

        # weight matrices
        self.input_to_state = np.random.normal(
            0, 0.01, (state_size, input_size))
        self.state_to_state = np.random.normal(
            0, 0.01, (state_size, state_size))
        self.state_to_output = np.random.normal(
            0, 0.01, (output_size, state_size))

    def set_input(self, input_vector):
        self.inputs = input_vector

    def perform_timestep(self):
        # calculate weighted sum
        self.state = self.state_biases + \
            np.dot(self.input_to_state, self.inputs) + \
            np.dot(self.state_to_state, self.state)

        # apply activation function
        self.state = sigmoid(self.state)

    def calculate_output(self):
        # calculate weighted sum
        self.output = self.output_biases + \
            np.dot(self.state_to_output, self.state)

        # apply activation function
        self.output = sigmoid(self.output)


def sigmoid(x):
    return 1 / (1 + 2**(-x))


myRnn = RNN(10, 5, 7)
myRnn.perform_timestep()
myRnn.calculate_output()
print(myRnn.state_to_state)
print(myRnn.state)
print(myRnn.output)
