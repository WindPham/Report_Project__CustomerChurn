import numpy as np

import Neuron as unit

class NeuronLayer(object):
    """description of class"""
    def __init__(self, number_of_neurons, bias):
        self.neurons = np.array([]);
        for i in range(number_of_neurons):
            self.neurons = np.append(self.neurons, unit.Neuron(bias))
        #self.neurons = np.full(shape=int(number_of_neurons), fill_value = unit.Neuron(self.bias), dtype = type(neuron_temp) )
    #this the feedward in a layer
    def feed_forward(self, inputs):
        # first wT.x = s, second: tanh(s), third: feed_forward(--S--)
        outputs = np.array([]);
        for neuron in self.neurons:
            outputs = np.append(outputs, neuron.tanh(neuron.wT_dot_x(inputs)))
        return outputs;

    def get_outputs(self):
        outputs = np.append([]);
        for neuron in self.neurons:
            outputs = np.append(outputs, neuron.output);
        return outputs;

    