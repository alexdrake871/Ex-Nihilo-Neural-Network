import numpy as np

# Properties of a neuron
inputs = [4.9, 1.6, 2.3]
weights = [7.7, 6.9, 3.1]
bias = 2

# Output of a neuron == inputs * its weight for all inputs, then add the bias of the neuron
output = 0.0
for i in range(len(inputs)):
    output += inputs[i] * weights[i]


# Can do the same for each neuron in a layer and store their outputs in order within an array














