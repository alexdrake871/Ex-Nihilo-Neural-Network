import numpy as np


class Layer():
    '''Hidden layer of neural network.'''
    def __init__(self, num_nodes: int, num_inputs: int):
        self.nodes = []
        for i in range(num_nodes):
            new_node = Node(num_inputs)
            self.nodes.append()

class Node():
    '''A single node/neuron within the hidden layers of the neural network.'''
    def __init__(self, num_inputs: int):
        self.weights = []
        self.bias = 0.0
        for i in range(num_inputs):
            self.weights.append(0.0)
    
    def calc_output(self, inputs: list[float]):
        output = 0.0

        for idx, val in enumerate(inputs):
            output += self.weights[idx] * val
        
        output += self.bias
        return output

# For testing only
the_node = Node(3)