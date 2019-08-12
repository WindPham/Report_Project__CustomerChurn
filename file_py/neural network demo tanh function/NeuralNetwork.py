import NeuronLayer as layer
import Neuron as unit
import numpy as np
class NeuralNetwork(object):
    """description of class"""
    LEARING_RATE = 0.5;
    def __init__(self, 
                 number_of_inputs,
                 number_of_neurons_in_hidden_layer,
                 number_of_neurons_in_output_layer,
                 hidden_layer_weights = None,
                 hidden_layer_bias = None,
                 output_layer_weights = None, 
                 output_layer_bias = None):

        self.number_of_inputs = number_of_inputs;
        self.hidden_layer = layer.NeuronLayer(number_of_neurons_in_hidden_layer, hidden_layer_bias);
        self.output_layer = layer.NeuronLayer(number_of_neurons_in_output_layer, output_layer_bias);

        
        self.init_weights_InputLayerHiddenLayer(hidden_layer_weights);
        self.init_weights_HiddenLayerOutputLayer(output_layer_weights);

        return;
    
    def init_weights_InputLayerHiddenLayer(self, weights):
        number_of_neurons = len(self.hidden_layer.neurons);
        weight_num = 0;
        for h in range(number_of_neurons):
            for i in range(self.number_of_inputs):
                if not weights:
                    self.hidden_layer.neurons[h].weights = np.append(self.hidden_layer.neurons[h].weights, np.array([np.random.random()]));
                else:
                    self.hidden_layer.neurons[h].weights = np.append(self.hidden_layer.neurons[h].weights, np.array([weights[weight_num]]));
                weight_num += 1;
        return;

    def init_weights_HiddenLayerOutputLayer(self, weights):
        number_of_neurons = len(self.output_layer.neurons);
        weight_num = 0;
        for o in range(number_of_neurons):
            for i in range(self.number_of_inputs):
                if not weights:
                    self.output_layer.neurons[o].weights = np.append(self.output_layer.neurons[o].weights, np.array([np.random.random()]));
                else:                                                     
                    self.output_layer.neurons[o].weights = np.append(self.output_layer.neurons[o].weights, np.array([weights[weight_num]]));
                weight_num += 1;
        return;

    def feed_forward(self, inputs):
        temp = self.hidden_layer.feed_forward(inputs);
        return self.output_layer.feed_forward(temp);

    def total_error(self, training_set):
        total_err =0.0;
        for i in range(len(training_set)):
            training_inputs, training_outputs = training_set[i];
            self.feed_forward(training_inputs);
            for o in range(len(training_inputs)):
                total_err += self.output_layer.neurons[o].mean_square_error(training_outputs[o]);

        return total_err;

    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs);
        number_of_neurons_output_layer = len(self.output_layer.neurons);
        number_of_neurons_hidden_layer = len(self.hidden_layer.neurons);
        # 1. output neuron delta:
        pd_J__pd_zj = np.zeros((number_of_neurons_output_layer));
        for o in range(number_of_neurons_output_layer):
            pd_J__pd_zj[o] = self.output_layer.neurons[o].pd_J__pd_zj(training_outputs[o]);

        # 2. Hidden layer deltas:
        pd_J__pd_yj = np.zeros((number_of_neurons_hidden_layer));
        for h in range(number_of_neurons_hidden_layer):
            d_J__d_yj = 0;
            for o in range(number_of_neurons_output_layer):
                d_J__d_yj += pd_J__pd_zj[o] * self.output_layer.neurons[o].weights[h]
            pd_J__pd_yj[h] = d_J__d_yj * self.hidden_layer.neurons[h].d_yj__d_zj();

        # 3. Update output neuron weights
        for o in range(number_of_neurons_output_layer):
            for wei_ho in range(len(self.output_layer.neurons[o].weights)):
                pd_J__pd_wij = pd_J__pd_zj[o] * self.output_layer.neurons[o].pd_zj__pd_wi(wei_ho)
                self.output_layer.neurons[o].weights[wei_ho] -= self.LEARING_RATE * pd_J__pd_wij;

        # 4. Update hidden neuron weights
        for h in range(number_of_neurons_hidden_layer):
            for wei_ih in range(len(self.hidden_layer.neurons[h].weights)):
                pd_J__pd_wij = pd_J__pd_zj[h] * self.hidden_layer.neurons[h].pd_zj__pd_wi(wei_ih);
                self.hidden_layer.neurons[h].weights[wei_ih] -= self.LEARING_RATE * pd_J__pd_wij;
                
        return;