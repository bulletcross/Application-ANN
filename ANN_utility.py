import numpy as np

"""Variables required to represent a neural network"""
#Numbers of nodes at each layer
layer_node_nr = []
#weight matrix
network_weight = []

"""Initialize weight of network with small random values"""
def weight_initialize():
    nr_weights = len(layer_node_nr)-1
    for i in range(0,nr_weights):
        #construct weight matrix W
        W = np.random.normal(scale=0.01, size = (layer_node_nr[i],layer_node_nr[i+1]))
        #append in network weight
        network_weight.append(W)

"""Activaton function"""
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    y = sigmoid(x)
    return y/(1-y)

def forward(input_layer):
    nr_weights = len(layer_node_nr)-1
    temp_layer = input_layer
    for i in range(0,nr_weights):
        temp_layer = np.dot(temp_layer,network_weight[i])
    return temp_layer


##layer_node_nr = [2,2,1]
##weight_initialize()
##print(network_weight)
##inp = np.array([1,2])
##print(forward(inp))
