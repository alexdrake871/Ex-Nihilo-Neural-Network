import numpy as np
from time import time
import random
import math
from dataset import sin_data_set


class XN_Neural_Network():
    '''The neural network which will be interacted with directly.'''
    def __init__(self, input_layer_size: int, output_layer_size: int, num_hidden_layers: int, nodes_per_hidden_layer: int):
        # Store number of inputs
        self.input_layer_size = input_layer_size

        # Create output layer
        if num_hidden_layers > 0:
            self.output_layer = Layer(output_layer_size, nodes_per_hidden_layer)
        else:
            self.output_layer = Layer(output_layer_size, input_layer_size)

        # Create hidden layers
        self.hidden_layers = []
        for i in range(num_hidden_layers):
            # The first hidden layer will have the input layer as inputs, all subsequent layers will have other hidden layers as inputs.
            if i == 0:
                new_hidden_layer = Layer(nodes_per_hidden_layer, input_layer_size)
            else:
                new_hidden_layer = Layer(nodes_per_hidden_layer, nodes_per_hidden_layer)
            self.hidden_layers.append(new_hidden_layer)
    
    def train(self, X, Y, ):
        pass

    def classify(self, input_vector: list[float]) -> list[float]:
        '''Returns the output from the neural network for the given input.'''
        inputs = input_vector

        for hidden_layer in self.hidden_layers:
            outputs = hidden_layer.get_node_outputs(inputs)
            inputs = outputs
        
        output_vector = self.output_layer.get_node_outputs(inputs)
        return output_vector


class Layer():
    '''Hidden layer of neural network.'''
    def __init__(self, num_nodes: int, num_inputs: int):
        self.nodes = []
        for i in range(num_nodes):
            new_node = Node(num_inputs)
            self.nodes.append(new_node)
    
    def get_node_outputs(self, inputs: list[float]) -> list[float]:
        outputs = []
        for node in self.nodes:
            node_output = node.get_output(inputs)
            outputs.append(node_output)
        return outputs


class Node():
    '''A single node/neuron within the layers of the neural network.'''
    def __init__(self, num_inputs: int):
        # FIXME OR DON'T: Can use the numpy zeroes function here
        self.weights = []
        self.bias = 0.0
        for i in range(num_inputs):
            self.weights.append(random.uniform(-0.1, 0.1))
    
    def relu(self, x: float):
        return max(0.0, x)
    
    def get_output(self, inputs: list[float]) -> float:
        # # FIXME OR DON'T: With numpy can simplify this to:
        # return np.dot(self.weights, inputs) + bias
        output = 0.0

        # Dot product of weights and inputs
        for idx, val in enumerate(inputs):
            output += self.weights[idx] * val
        
        output += self.bias
        return self.relu(output)



# For testing only

the_X, the_y = sin_data_set(1000)
features = len(the_X[0])

the_network = XN_Neural_Network()
