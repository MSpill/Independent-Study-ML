import numpy as np
import random
import matplotlib.pyplot as plt
import rnn
import data.onehottext as one_hotter
import pickle


def derivative(activation_function):
    if activation_function is rnn.relu:
        return lambda x: np.piecewise(x, [x < 0, x >= 0], [lambda a: 0, lambda a: 1])
    elif activation_function is rnn.leaky_relu:
        return lambda x: np.piecewise(x, [x < 0, x >= 0], [lambda a: -0.01, lambda a: 1])
    elif activation_function is rnn.sigmoid:
        return lambda x: (2.71 ** (-x)) / ((1 + 2.71 ** (-x)) ** 2)


class RNNTrainer:

    def __init__(self, rnn, inputs, outputs, batch_size=200, learning_rate=0.01, momentum=0.9):
        self.rnn = rnn
        self.inputs = inputs
        self.outputs = outputs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum

    def train(self, num_epochs):
        # an epoch is when you've gone through as many example inputs as there are in the training set
        curr_epoch = 0.0
        error_list = []
        gradient_mags = [[] for i in range(len(self.rnn.forward_weights))]

        prev_feedforward_derivs = [w * 0 for w in self.rnn.forward_weights]
        prev_recurrent_derivs = [w * 0 for w in self.rnn.recurrent_weights]
        prev_bias_derivs = [b * 0 for b in self.rnn.biases]
        while curr_epoch < num_epochs:
            self.rnn.reset_state()
            curr_epoch += (self.batch_size+0.0) / len(self.inputs)
            start_index = random.randint(0, len(self.inputs) - self.batch_size)
            predictions = []
            states = [self.rnn.values[0:-1]]
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

                target = np.transpose([self.outputs[start_index+t]])

                output_error = predictions[t] - target
                squared_error += np.sum(output_error ** 2) / \
                    len(self.outputs[start_index+t]) / len(predictions)

                # output_error = -(target / predictions[t]) - \
                #    (target - 1.0) / (1.0 - predictions[t])

                # calculate derivs for state-to-output weights and output biases
                # these only need to be calculated once, they aren't used earlier

                # I have checked this part and I'm confident it's correct
                output_deriv = output_error * \
                    derivative(self.rnn.activation_function)(
                        pre_activations[t][-1])
                bias_derivs[-1] += output_deriv
                feedforward_derivs[-1] += np.dot(output_deriv,
                                                 np.transpose(states[t+1][-1]))

                # now calculate derivs for other feedforward/recurrent weights and biases
                # they were used more than once to produce this timestep's predict, so we need
                # to backpropagate through time to sum up all the derivatives

                # most recent timestep, feedforward only
                for t2 in range(t, -1, -1):
                    for i in range(len(state_derivs)-1, -1, -1):
                        if i == len(state_derivs)-1:
                            # top hidden units will depend on just output or just next top layer
                            if t2 == t:
                                # case where it just depends on output
                                state_derivs[i] = derivative(
                                    self.rnn.activation_function)(pre_activations[t2][i]) * np.dot(np.transpose(self.rnn.forward_weights[i+1]), output_deriv)
                            else:
                                # case where it just depends on next timestep's top layer
                                state_derivs[i] = derivative(
                                    self.rnn.activation_function)(pre_activations[t2][i]) * np.dot(np.transpose(self.rnn.recurrent_weights[i]), state_derivs[i])
                        else:
                            # middle hidden units will depend on the higher hidden units and maybe the next timestep's units on same layer
                            base_deriv = derivative(self.rnn.activation_function)(
                                pre_activations[t2][i]) * np.dot(np.transpose(self.rnn.forward_weights[i+1]), state_derivs[i+1])
                            if t2 != t:
                                base_deriv += derivative(self.rnn.activation_function)(
                                    pre_activations[t2][i]) * np.dot(np.transpose(self.rnn.recurrent_weights[i]), state_derivs[i])
                            state_derivs[i] = base_deriv
                        bias_derivs[i] += state_derivs[i]
                        feedforward_derivs[i] += np.dot(state_derivs[i],
                                                        np.transpose(states[t2+1][i]))
                        recurrent_derivs[i] += np.dot(state_derivs[i],
                                                      np.transpose(states[t2][i+1]))
            # finally, use the partial derivatives to update the weights and biases
            mult_factor = self.learning_rate/self.batch_size
            for i in range(0, len(self.rnn.forward_weights)):
                delta = feedforward_derivs[i] * \
                    mult_factor + self.momentum * prev_feedforward_derivs[i]
                self.rnn.forward_weights[i] -= delta
                prev_feedforward_derivs[i] = delta
                gradient_mags[i].append(np.mean(np.abs(feedforward_derivs[i])))
            for i in range(0, len(self.rnn.recurrent_weights)):
                delta = recurrent_derivs[i] * \
                    mult_factor + self.momentum * prev_recurrent_derivs[i]
                self.rnn.recurrent_weights[i] -= delta
                prev_recurrent_derivs[i] = delta
            for i in range(0, len(self.rnn.biases)):
                delta = bias_derivs[i] * \
                    mult_factor + self.momentum * prev_bias_derivs[i]
                self.rnn.biases[i] -= delta
                prev_bias_derivs[i] = delta

            # Print info to track training progress
            error_list.append(squared_error)
            print("Avg squared error: {}".format(squared_error))
            if squared_error < 0.005:
                # this is mostly to keep plots nice
                break
                pass
        #plt.plot([0.2] * len(error_list))
        plt.subplot(2, 1, 1)
        for i in range(len(gradient_mags)):
            plt.plot(gradient_mags[i])
        plt.legend([0, 1, 2, 3])
        plt.ylabel("avg. magnitude of weight gradient")

        plt.subplot(2, 1, 2)
        plt.plot(error_list)
        plt.xlabel("training step")
        plt.ylabel("squared error")
        plt.show()
        # plt.plot(error_list)
        # plt.show()


if __name__ == '__main__':

    thicc_data = one_hotter.one_hot_text_data("data/data.c", size=1000000)
    print(len(thicc_data))

    thicc_input = thicc_data
    thicc_output = [thicc_data[n+1] for n in range(0, len(thicc_input)-1)]
    # make them the same size, one noisy target is fine
    thicc_output.append(thicc_output[0])

    input_size = len(thicc_input[0])

    # inputs = [[0], [0], [1], [0], [1], [1], [1], [0], [0], [1], [0], [1], [0], [1], [1],
    #          [0], [1], [1], [0], [1], [0], [0], [0], [1], [0], [0], [1], [1], [0]] * 200
    #outputs = [[inputs[i][0]*inputs[i-1][0]] for i in range(len(inputs))]

    my_rnn = rnn.RNN(input_size, [400, 200], input_size,
                     activation_function=rnn.sigmoid)
    #my_rnn = pickle.load(open('rnn1.rnn', 'rb'))

    rnn_trainer = RNNTrainer(my_rnn, thicc_input, thicc_output,
                             batch_size=25, learning_rate=0.003, momentum=0.96)
    rnn_trainer.train(num_epochs=25)

    rnn_file = open('rnn2.rnn', 'wb')
    pickle.dump(my_rnn, rnn_file)
    rnn_file.close()

    for i in range(100):
        my_rnn.perform_timestep(thicc_input[i])
        print(my_rnn.predict())
