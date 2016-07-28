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
weight_delta = []

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
    input_layer_values.append(np.array(temp_layer))
    output_layer_values.append(np.array(temp_layer))
    for i in range(0,nr_weights):
        temp_layer = np.dot(temp_layer,network_weight[i])
        input_layer_values.append(temp_layer)
        temp_layer = sigmoid(temp_layer)
        output_layer_values.append(temp_layer)

def net_response(input_layer):
    nr_weights = len(layer_node_nr)-1
    temp_layer = input_layer
    for i in range(0,nr_weights):
        temp_layer = np.dot(temp_layer,network_weight[i])
        temp_layer = sigmoid(temp_layer)
    return temp_layer

def calculate_delta(output):
    nr_weights = len(layer_node_nr)-1
##    print(nr_weights)
    #print(output)
    #print(output_layer_values[nr_weights])
    delta.append((output_layer_values[nr_weights]-output)*sigmoid_der(input_layer_values[nr_weights]))
    #print(delta[-1].shape)
    for i in range(nr_weights-1,0,-1):
##        print(network_weight[i].shape)
##        print(network_weight[i].transpose().shape)
##        print(delta[-1].shape)
##        print(network_weight[i].transpose())
##        print(delta[-1])
##        print(delta)
        temp = (np.dot(network_weight[i],delta[-1]))
        #print(temp.shape)
        #print(input_layer_values[i].shape)
        #print(sigmoid_der(input_layer_values[i]))
        delta.append(temp*sigmoid_der(input_layer_values[i]).transpose())
        #print(delta[-1].shape)
    delta.reverse()
    
def calculate_weight_delta():
    nr_weights = len(layer_node_nr)-1
    for i in range(0,nr_weights):
        #print(output_layer_values[i])
        #print(delta[i])
        #print(output_layer_values[i].shape)
        #print(delta[i].shape)
        weight_delta.append(np.dot(output_layer_values[i].transpose(),delta[i].transpose()))

def adjust_weights(scale):
    nr_weights = len(layer_node_nr)-1
    for i in range(0,nr_weights):
        network_weight[i] = network_weight[i] - scale*weight_delta[i]

def learn_network(input_value, output_value, scale):
    #print(input_value)
    #print(output_value)
    
    forward(input_value)
    calculate_delta(output_value)
    calculate_weight_delta()
    adjust_weights(scale)
    error = np.sum((output_value-output_layer_values[-1])**2)
    return error


layer_node_nr = [2,3,4,1]
weight_initialize()
##print(network_weight)
##forward([[1,2]])
##print(input_layer_values)
##print(output_layer_values)

##calculate_delta(np.array([[7]]))
##print(delta)
##calculate_weight_delta()
##adjust_weights(0.5)
inp = [[0,0],[0,1],[1,0],[1,1]]
out = [1,0,0,1]
print(network_weight) 
for i in range(1,10):
    for j in range(0,4):
        input_layer_values = []
        output_layer_values = []
        delta = []
        weight_delta = []
        #print(learn_network(np.array([inp[j]]),np.array([[out[j]]]),0.5))
        learn_network(np.array([inp[j]]),np.array([[out[j]]]),0.5)
print(network_weight)       
##print(network_weight)
##print(weight_delta)
##g = np.array(out)
##print(sigmoid(g))

def func(x,y):
    return (((x*x + y)-3)/32)

##x = rd.uniform(1,5)
##y = rd.uniform(2,10)
##out = func(x,y)
##
##for i in range(1,1000):
##    input_layer_values = []
##    output_layer_values = []
##    delta = []
##    weight_delta = []
##    x = rd.uniform(1,5)
##    y = rd.uniform(2,10)
##    out = func(x,y)
##    learn_network(np.array([[x,y]]),np.array([[out]]),0.1)
##
##x = rd.uniform(1,5)
##y = rd.uniform(2,10)
##out = func(x,y)
##
##for i in range(1,10):
##    input_layer_values = []
##    output_layer_values = []
##    delta = []
##    weight_delta = []
##    x = rd.uniform(1,5)
##    y = rd.uniform(2,10)
##    out = func(x,y)
##    print(out)
##    print(net_response(np.array([[x,y]])))
