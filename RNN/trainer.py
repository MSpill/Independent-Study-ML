import numpy as np
import random
import matplotlib.pyplot as plt
import rnn


def derivative(activation_function):
    if activation_function is rnn.relu:
        return lambda x: np.piecewise(x, [x < 0, x >= 0], [lambda a: 0, lambda a: 1])
    elif activation_function is rnn.leaky_relu:
        return lambda x: np.piecewise(x, [x < 0, x >= 0], [lambda a: -0.01, lambda a: 1])
    elif activation_function is rnn.sigmoid:
        return lambda x: (2.71 ** (-x)) / ((1 + 2.71 ** (-x)) ** 2)


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
        error_list = []
        while curr_epoch < num_epochs:
            self.rnn.reset_state()
            curr_epoch += (self.batch_size+0.0) / len(self.inputs)
            start_index = random.randint(0, len(self.inputs) - self.batch_size)
            predictions = []
            states = []
            pre_activations = []
            # feed in batch_size inputs and record all the predictions and internal states
            step = 0
            while step < self.batch_size:
                self.rnn.perform_timestep(self.inputs[start_index+step])
                predictions.append(self.rnn.predict())
                # including input in states array for ease in backprop
                states.append(self.rnn.values[0:-1])
                pre_activations.append(
                    [i * 1.0 for i in self.rnn.pre_activations])
                step += 1

            # perform backprop through time

            squared_error = 0.0
            # place to store derivatives, I'm calling them derivs for short
            feedforward_derivs = [w * 0 for w in self.rnn.forward_weights]
            recurrent_derivs = [w * 0 for w in self.rnn.recurrent_weights]
            bias_derivs = [b * 0 for b in self.rnn.biases]
            for t in range(0, len(predictions)):

                # this stores the derivatives of error w/ respect to neuron outputs
                # it will be the info used to calculate derivatives for weights and biases
                # we only need to store the most recently calculated timestep's derivs
                state_derivs = [self.rnn.values[i] *
                                0 for i in range(1, len(self.rnn.values)-1)]

                output_error = predictions[t] - \
                    np.transpose([self.outputs[start_index+t]])
                squared_error += np.sum(output_error ** 2) / \
                    len(self.outputs[start_index+t]) / len(predictions)

                # calculate derivs for state-to-output weights and output biases
                # these only need to be calculated once, they aren't used earlier
                output_deriv = output_error * \
                    derivative(self.rnn.activation_function)(
                        pre_activations[t][-1])
                bias_derivs[-1] += output_deriv
                feedforward_derivs[-1] += np.dot(output_deriv,
                                                 np.transpose(states[t][-1]))

                # now calculate derivs for other feedforward/recurrent weights and biases
                # they were used more than once to produce this timestep's predict, so we need
                # to backpropagate through time to sum up all the derivatives

                # most recent timestep, feedforward only
                for i in range(len(state_derivs)-1, -1, -1):
                    # top layer will depend on just output or just next top layer
                    if i == len(state_derivs)-1:
                        # case where it just depends on output
                        state_derivs[i] = derivative(
                            self.rnn.activation_function)(pre_activations[t][i]) * np.dot(np.transpose(self.rnn.forward_weights[i+1]), output_deriv)
                    else:
                        state_derivs[i] = derivative(self.rnn.activation_function)(
                            pre_activations[t][i]) * np.dot(np.transpose(self.rnn.forward_weights[i+1]), state_derivs[i+1])
                    bias_derivs[i] += state_derivs[i]
                    feedforward_derivs[i] += np.dot(state_derivs[i],
                                                    np.transpose(states[t][i]))
            # finally, use the partial derivatives to update the weights and biases
            mult_factor = self.learning_rate/self.batch_size
            for i in range(0, len(self.rnn.forward_weights)):
                self.rnn.forward_weights[i] -= feedforward_derivs[i] * mult_factor
            for i in range(0, len(self.rnn.recurrent_weights)):
                self.rnn.recurrent_weights[i] -= recurrent_derivs[i] * mult_factor
            for i in range(0, len(self.rnn.biases)):
                self.rnn.biases[i] -= bias_derivs[i] * mult_factor

            # Print info to track training progress
            error_list.append(squared_error)
            print("Avg squared error: {}".format(squared_error))
            if squared_error < 0.0001:
                # this is mostly to keep plots nice
                break
        plt.plot(error_list)
        plt.xlabel("training step")
        plt.ylabel("squared error")
        plt.show()


if __name__ == '__main__':
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]] * 2000
    outputs = [[0, 1], [1, 0], [1, 1], [0, 1]] * 2000
    my_rnn = rnn.RNN(2, [5, 5, 5], 2, activation_function=rnn.sigmoid)

    rnn_trainer = RNNTrainer(my_rnn, inputs, outputs,
                             batch_size=4, learning_rate=15)
    rnn_trainer.train(num_epochs=5)

    for i in range(100):
        my_rnn.perform_timestep(inputs[i])
        print(my_rnn.predict())
