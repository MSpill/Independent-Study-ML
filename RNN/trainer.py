import numpy as np
import random
import rnn


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
            # feed in batch_size inputs and record all the predictions and internal states
            step = 0
            while step < self.batch_size:
                step += 1
                self.rnn.perform_timestep(self.inputs[start_index+step])
                predictions.append(self.rnn.predict())
                states.append(self.rnn.values[1:-1])

            # perform backprop through time
            squared_error = 0.0
            for t in range(0, len(predictions)):
                # print("predict: {0} target: {1}".format(
                #    predictions[t], self.outputs[start_index+t]))
                output_delta = predictions[t] - self.outputs[start_index+t]
                squared_error += np.sum(output_delta ** 2) / \
                    len(self.outputs[start_index+t]) / len(predictions)

            print("Squared error: {}".format(squared_error))


if __name__ == '__main__':
    inputs = [[0], [1], [0], [1]] * 20000
    outputs = [[1], [0], [1], [0]] * 20000
    my_rnn = rnn.RNN(1, [2, 3], 1, activation_function=rnn.relu)
    # for i in range(10):
    #    my_rnn.perform_timestep(1)
    #    print(my_rnn.predict())
    rnn_trainer = RNNTrainer(my_rnn, inputs, outputs, batch_size=200)
    rnn_trainer.train(1)
