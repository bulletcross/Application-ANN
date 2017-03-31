import numpy as np
import random as rd

"""Variables required to represent a neural network"""
#Numbers of nodes at each layer
layer_node_nr = []
#weight matrix
network_weight = []
#input-output layer values
input_layer_values = []
output_layer_values = []
#delta value for gradient computation
delta = []

w_delta = []

"""Initialize weight of network with small random values"""
def weight_initialize():
    nr_weights = len(layer_node_nr)-1
    for i in range(0,nr_weights):
        #construct weight matrix W
        W = np.random.normal(scale=0.1, size = (layer_node_nr[i],layer_node_nr[i+1]))
        #append in network weight
        network_weight.append(W)

"""Activaton function"""
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return np.exp(-x)/((1+np.exp(-x))**2)

"""forward function"""
def forward(input_layer):
    nr_weights = len(layer_node_nr)-1
    temp_layer = input_layer
    input_layer_values.append(np.array(temp_layer))
    output_layer_values.append(np.array(temp_layer))
    for i in range(0,nr_weights):
        temp_layer = np.dot(temp_layer,network_weight[i])
        input_layer_values.append(temp_layer)
        temp_layer = sigmoid(temp_layer)
        output_layer_values.append(temp_layer)

"""output through forward"""
def net_response(input_layer):
    nr_weights = len(layer_node_nr)-1
    temp_layer = input_layer
    for i in range(0,nr_weights):
        temp_layer = np.dot(temp_layer,network_weight[i])
        temp_layer = sigmoid(temp_layer)
    return temp_layer

"""Back propagation procedures"""
def calculate_delta(output):
    nr_weights = len(layer_node_nr)-1
    delta.append((output_layer_values[nr_weights]-output)*sigmoid_der(input_layer_values[nr_weights]))
    for i in range(nr_weights-1,0,-1):
        temp = (np.dot(delta[-1],network_weight[i].transpose()))
        delta.append(temp*sigmoid_der(input_layer_values[i]))
    delta.reverse()
    
def calculate_weight_delta():
    nr_weights = len(layer_node_nr)-1
    for i in range(0,nr_weights):
        weight_delta.append(np.dot(output_layer_values[i].transpose(),delta[i]))

def adjust_weights(scale):
    nr_weights = len(layer_node_nr)-1
    for i in range(0,nr_weights):
        network_weight[i] = network_weight[i] - (scale*weight_delta[i])

"""Forward backward combned"""
def learn_network(input_value, output_value, scale, error_threshold):
    forward(input_value)
    error = np.sum((output_value-output_layer_values[-1])**2)
    if error > error_threshold:
        calculate_delta(output_value)
        calculate_weight_delta()
        adjust_weights(scale)
    return error
