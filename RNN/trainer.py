import numpy as np
import random
import rnn


def derivative(activation_function):
    if activation_function is rnn.relu:
        return lambda x: np.piecewise(x, [x < 0, x >= 0], [lambda a: 0, lambda a: 1])
    elif activation_function is rnn.sigmoid:
        return lambda x: 2 * (2.71 ** (-x)) / ((1 + 2.71 ** (-x)) ** 2)


class RNNTrainer:

    def __init__(self, rnn, inputs, outputs, learning_rate=0.01, batch_size=200):
        self.rnn = rnn
        self.inputs = inputs
        self.outputs = outputs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def train(self, num_epochs):
        # an epoch is when you've gone through as many example inputs as there are in the training set
        curr_epoch = 0.0
        while curr_epoch < num_epochs:
            self.rnn.reset_state()
            curr_epoch += self.batch_size / len(self.inputs)
            start_index = random.randint(0, len(self.inputs) - self.batch_size)
            predictions = []
            states = []
            pre_activations = []
            # feed in batch_size inputs and record all the predictions and internal states
            step = 0
            while step < self.batch_size:
                step += 1
                self.rnn.perform_timestep(self.inputs[start_index+step])
                predictions.append(self.rnn.predict())
                states.append(self.rnn.values[1:-1])
                pre_activations.append(self.rnn.pre_activations[1:])

            # perform backprop through time

            squared_error = 0.0
            # place to store derivatives, I'm calling them derivs for short
            feedforward_derivs = [w * 0 for w in self.rnn.forward_weights]
            recurrent_derivs = [w * 0 for w in self.rnn.recurrent_weights]
            bias_derivs = [b * 0 for b in self.rnn.biases]
            for t in range(0, len(predictions)):

                # this stores the derivatives of error w/ respect to neuron outputs
                # it will be the info used to calculate derivatives for weights and biases
                # we only need to store the most recently calculated timestep's deltas
                state_derivs = [self.rnn.values[i] *
                                0 for i in range(1, len(self.rnn.values)-1)]

                output_error = predictions[t] - self.outputs[start_index+t]
                squared_error += np.sum(output_error ** 2) / \
                    len(self.outputs[start_index+t]) / len(predictions)

                # calculate derivs for state-to-output weights and output biases
                # these only need to be calculated once, they aren't used earlier
                output_deriv = output_error * \
                    derivative(self.rnn.activation_function)(
                        pre_activations[t][-1])
                bias_derivs[-1] += output_deriv
                feedforward_derivs[-1] += np.dot(predictions[t],
                                                 np.transpose(states[t][-1]))

                # now calculate derivs for other feedforward/recurrent weights and biases
                # they were used more than once to produce this timestep's predict, so we need
                # to backpropagate through time to sum up all the derivatives

            # finally, use the partial derivatives to update the weights and biases
            mult_factor = self.learning_rate/len(self.inputs)
            for i in range(0, len(self.rnn.forward_weights)):
                self.rnn.forward_weights[i] -= feedforward_derivs[i] * mult_factor
            for i in range(0, len(self.rnn.recurrent_weights)):
                self.rnn.recurrent_weights[i] -= recurrent_derivs[i] * mult_factor
            for i in range(0, len(self.rnn.biases)):
                self.rnn.biases[i] -= bias_derivs[i] * mult_factor

            # Print info to track training progress
            print("Avg squared error: {}".format(squared_error))


if __name__ == '__main__':
    inputs = [[0], [1], [0], [1]] * 20000
    outputs = [[1], [0], [1], [0]] * 20000
    my_rnn = rnn.RNN(1, [2, 3], 1, activation_function=rnn.relu)
    # for i in range(10):
    #    my_rnn.perform_timestep(1)
    #    print(my_rnn.predict())
    rnn_trainer = RNNTrainer(my_rnn, inputs, outputs,
                             batch_size=100, learning_rate=0.5)
    rnn_trainer.train(num_epochs=1)
