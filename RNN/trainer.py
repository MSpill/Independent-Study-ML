import numpy as np
import random
import matplotlib.pyplot as plt
import rnn
import data.onehotgenome as one_hotter
import data.onehottext as text_hotter
import pickle

# takes a function as an argument and returns that function's derivative function
# I hardcoded each derivative since there were only a few functions that would be arguments


def derivative(activation_function):
    if activation_function is rnn.relu:
        return lambda x: np.piecewise(x, [x < 0, x >= 0], [lambda a: 0, lambda a: 1])
    elif activation_function is rnn.leaky_relu:
        return lambda x: np.piecewise(x, [x < 0, x >= 0], [lambda a: -0.01, lambda a: 1])
    elif activation_function is rnn.sigmoid:
        return lambda x: (2.71 ** (-x)) / ((1 + 2.71 ** (-x)) ** 2)

# takes an RNN and some data and trains the RNN on the data


class RNNTrainer:

    def __init__(self, rnn, inputs, outputs):
        self.rnn = rnn
        self.inputs = inputs
        self.outputs = outputs

    def train(self, num_epochs, batch_size=10, time_depth=20, learning_rate=0.01, momentum=0.9, charset=[]):
        # an epoch is when you've gone through as many example inputs as there are in the training set
        curr_epoch = 0.0

        # keep track of training metrics for later analysis
        error_list = []
        gradient_mags = [[] for i in range(len(self.rnn.forward_weights))]

        # keep track of previous gradients for the momentum term of gradient descent
        prev_feedforward_derivs = [w * 0 for w in self.rnn.forward_weights]
        prev_recurrent_derivs = [w * 0 for w in self.rnn.recurrent_weights]
        prev_bias_derivs = [b * 0 for b in self.rnn.biases]

        start_index = 0
        states = []
        pre_activations = []
        total_steps = 0

        def reset_run():
            nonlocal start_index
            nonlocal states
            nonlocal pre_activations
            nonlocal total_steps
            total_steps = 0
            start_index = random.randint(0, len(self.inputs) - batch_size)
            self.rnn.reset_state()
            states = [self.rnn.values[0:-1]] * (time_depth+1)
            pre_activations = [[
                i * 0.0 for i in self.rnn.pre_activations]] * time_depth

        reset_run()
        while curr_epoch < num_epochs:
            curr_epoch += (batch_size+0.0) / len(self.inputs)
            predictions = []
            # feed in batch_size inputs and record all the predictions and internal states
            for i in range(start_index, start_index+batch_size):
                self.rnn.perform_timestep(self.inputs[i])
                predictions.append(self.rnn.predict())
                # including input in states array for ease in backprop
                states.append(self.rnn.values[0:-1])
                pre_activations.append(
                    [i * 1.0 for i in self.rnn.pre_activations])
                if len(states) > time_depth+batch_size+1:
                    states.pop(0)
                    pre_activations.pop(0)

            squared_error = 0.0

            # place to store derivatives, I'm calling them derivs for short
            feedforward_derivs = [w * 0 for w in self.rnn.forward_weights]
            recurrent_derivs = [w * 0 for w in self.rnn.recurrent_weights]
            bias_derivs = [b * 0 for b in self.rnn.biases]

            # perform backpropagation through time
            for t in range(0, len(predictions)):

                # this stores the derivatives of error w/ respect to neuron outputs
                # it will be the info used to calculate derivatives for weights and biases
                # we only need to store the most recently calculated timestep's derivs
                state_derivs = [self.rnn.values[i] *
                                0 for i in range(1, len(self.rnn.values)-1)]

                target = np.transpose([self.outputs[start_index+t]])

                output_error = predictions[t] - target
                squared_error += np.sum(output_error ** 2) / \
                    len(self.outputs[0]) / len(predictions)

                # calculate derivs for state-to-output weights and output biases
                # these only need to be calculated once, they aren't used earlier
                output_deriv = output_error * \
                    derivative(self.rnn.activation_function)(
                        pre_activations[t+time_depth][-1])
                bias_derivs[-1] += output_deriv
                feedforward_derivs[-1] += np.dot(output_deriv,
                                                 np.transpose(states[t+time_depth+1][-1]))

                # now calculate derivs for other feedforward/recurrent weights and biases
                # they were used more than once to produce this timestep's prediction, so we need
                # to backpropagate through time to sum up all the derivatives

                # backprop through time with respect to this timestep's output
                for t2 in range(0, min(time_depth, t+total_steps+1)):
                    for i in range(len(state_derivs)-1, -1, -1):
                        if i == len(state_derivs)-1:
                            # top hidden units will affect just output or just next top layer
                            if t2 == 0:
                                # case where it just affects output
                                state_derivs[i] = derivative(
                                    self.rnn.activation_function)(pre_activations[t-t2+time_depth][i]) * np.dot(np.transpose(self.rnn.forward_weights[i+1]), output_deriv)
                            else:
                                # case where it just affects next timestep's top layer
                                state_derivs[i] = derivative(
                                    self.rnn.activation_function)(pre_activations[t-t2+time_depth][i]) * np.dot(np.transpose(self.rnn.recurrent_weights[i]), state_derivs[i])
                        else:
                            # middle hidden units will affect the higher hidden units and maybe the next timestep's units on same layer
                            base_deriv = derivative(self.rnn.activation_function)(
                                pre_activations[t-t2+time_depth][i]) * np.dot(np.transpose(self.rnn.forward_weights[i+1]), state_derivs[i+1])
                            if t2 != 0:  # they affect next timestep's output
                                base_deriv += derivative(self.rnn.activation_function)(
                                    pre_activations[t-t2+time_depth][i]) * np.dot(np.transpose(self.rnn.recurrent_weights[i]), state_derivs[i])
                            state_derivs[i] = base_deriv
                        bias_derivs[i] += state_derivs[i]
                        feedforward_derivs[i] += np.dot(state_derivs[i],
                                                        np.transpose(states[t-t2+time_depth+1][i]))
                        recurrent_derivs[i] += np.dot(state_derivs[i],
                                                      np.transpose(states[t-t2+time_depth][i+1]))
            # finally, use the partial derivatives to update the weights and biases
            mult_factor = learning_rate/batch_size
            for i in range(0, len(self.rnn.forward_weights)):
                delta = feedforward_derivs[i] * \
                    mult_factor + momentum * prev_feedforward_derivs[i]
                self.rnn.forward_weights[i] -= delta
                prev_feedforward_derivs[i] = delta
                gradient_mags[i].append(np.mean(np.abs(feedforward_derivs[i])))
            for i in range(0, len(self.rnn.recurrent_weights)):
                delta = recurrent_derivs[i] * \
                    mult_factor + momentum * prev_recurrent_derivs[i]
                self.rnn.recurrent_weights[i] -= delta
                prev_recurrent_derivs[i] = delta
            for i in range(0, len(self.rnn.biases)):
                delta = bias_derivs[i] * \
                    mult_factor + momentum * prev_bias_derivs[i]
                self.rnn.biases[i] -= delta
                prev_bias_derivs[i] = delta

            # restart from a new point in the data every so often
            start_index += batch_size
            total_steps += batch_size
            if start_index >= len(self.outputs) - batch_size or total_steps > 100:
                reset_run()
                print(self.rnn.sample_text(charset, 0.1, 100))

            # Print info to track training progress
            error_list.append(squared_error)
            print("Avg squared error: {}".format(squared_error))
            if squared_error < 0.005:
                # this is mostly to keep plots nice
                break
                pass

        # once training is complete, plot gradient magnitudes and error over time
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

        # save error over time to a file
        error_file = open('error_list3', 'wb')
        pickle.dump(error_list, error_file)
        error_file.close()


if __name__ == '__main__':

    # load data
    onehot_data = one_hotter.one_hot_genome(
        "/Users/matthewspillman/Documents/_12th/Indep Study/Independent-Study-ML/RNN/data/genome.fna")
    charset = onehot_data[0]
    dataset = onehot_data[1]
    print(len(dataset))

    # split data into input and output
    input_data = dataset
    output_data = [dataset[n+1] for n in range(0, len(input_data)-1)]
    # make them the same size, one noisy target is fine
    output_data.append(output_data[0])
    input_size = len(input_data[0])

    # if creating an RNN from scratch:
    # my_rnn = rnn.RNN(input_size, [500], input_size,
    #                 activation_function=rnn.sigmoid)

    # if loading a saved RNN from a file:
    my_rnn = pickle.load(open('rnn9.rnn', 'rb'))

    # train the RNN
    rnn_trainer = RNNTrainer(my_rnn, input_data, output_data)
    rnn_trainer.train(num_epochs=0.5, batch_size=100,
                      time_depth=20, learning_rate=0.007, momentum=0.9, charset=charset)

    # save the RNN to a file
    rnn_file = open('rnn10.rnn', 'wb')
    pickle.dump(my_rnn, rnn_file)
    rnn_file.close()
