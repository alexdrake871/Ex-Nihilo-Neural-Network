import numpy as np
from time import time
import random
import math
from dataset import sin_data_set


class XN_Neural_Network():
    '''The neural network which will be interacted with directly.'''
    def __init__(self, input_layer_size: int, num_hidden_layers: int, nodes_per_hidden_layer: int):
        # Store number of inputs
        self.input_layer_size = input_layer_size

        # Store other parameters
        self.num_hidden_layers = num_hidden_layers
        self.nodes_per_hidden_layer = nodes_per_hidden_layer

        # Create hidden layers
        self.hidden_layers = []
        for i in range(num_hidden_layers):
            # The first hidden layer will have the input layer as inputs, all subsequent layers will have other hidden layers as inputs.
            if i == 0:
                new_hidden_layer = Layer(nodes_per_hidden_layer, input_layer_size)
            else:
                new_hidden_layer = Layer(nodes_per_hidden_layer, nodes_per_hidden_layer)
            self.hidden_layers.append(new_hidden_layer)
    
    def train(self, X, y):
        # Determine how many outputs are needed
        self.output_labels = []
        for output in y:
            if output not in self.output_labels:
                self.output_labels.append(output)
        self.output_layer_size = len(self.output_labels)

        # Create output layer based on y-values
        if self.num_hidden_layers > 0:
            self.output_layer = Layer(self.output_layer_size, self.nodes_per_hidden_layer, True)
        else:
            self.output_layer = Layer(self.output_layer_size, self.input_layer_size, True)
    
    # FIXME
    def display(self):
        '''Display a visual representation of the trained neural network.'''
        pass

    def classify(self, input_vector: list[float]) -> list[float]:
        '''Returns the output from the neural network for the given input.'''
        inputs = input_vector

        for hidden_layer in self.hidden_layers:
            outputs = hidden_layer.forward(inputs)
            inputs = outputs
        
        output_vector = self.output_layer.forward(inputs)
        return output_vector


class Layer():
    '''Hidden layer of neural network.'''
    def __init__(self, num_nodes: int, num_inputs: int, output_layer = False):
        self.output_layer = output_layer    # Flag showing whether its output layer or hidden layer
        self.nodes = []
        for i in range(num_nodes):
            new_node = Node(num_inputs, self.output_layer)
            self.nodes.append(new_node)
    
    def ReLU(self, values: list[float]):
        return [max(0.0, x) for x in values]
    
    def softmax(self, values: list[float]):    # 
        E = math.e
        zeroed_vals = [x - max(values) for x in values]
        e_vals = [E**x for x in zeroed_vals]
        return [x / sum(e_vals) for x in e_vals]

    def forward(self, inputs: list[float]) -> list[float]:
        outputs = []
        for node in self.nodes:
            node_output = node.get_output(inputs)
            outputs.append(node_output)
        
        if not self.output_layer:   # Return the hidden layer's outputs
            return self.ReLU(outputs)
        else:                       # Return the proportional representation of output layer's outputs
            return self.softmax(outputs)


class Node():
    '''A single node/neuron within the layers of the neural network.'''
    def __init__(self, num_inputs: int, output_node = False):
        # FIXME OR DON'T: Can use the numpy zeroes function here
        self.weights = []
        self.bias = 0.0
        
        self.output_node = output_node  # Flag showing whether node is part of output layer or hidden layer
        for i in range(num_inputs):
            self.weights.append(random.uniform(-0.1, 0.1))
    
    def get_output(self, inputs: list[float]) -> float:
        # # FIXME OR DON'T: With numpy can simplify this to:
        # return np.dot(self.weights, inputs) + bias
        output = 0.0

        # Dot product of weights and inputs
        for idx, val in enumerate(inputs):
            output += self.weights[idx] * val
        
        output += self.bias
        return output



# For testing only
random.seed(1234)

the_X, the_y = sin_data_set(1000)
features = len(the_X[0])

the_network = XN_Neural_Network(features, 3, 8)
the_network.train(the_X, the_y)
print(the_network.classify(the_X[4]))
print(the_network.output_layer_size)
