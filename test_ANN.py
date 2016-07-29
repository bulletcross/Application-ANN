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
    delta.append((output_layer_values[nr_weights]-output)*sigmoid_der(input_layer_values[nr_weights]))
    for i in range(nr_weights-1,0,-1):
        temp = (np.dot(delta[-1],network_weight[i].transpose()))
        delta.append(temp*sigmoid_der(input_layer_values[i]))
    delta.reverse()
    
def calculate_weight_delta():
    nr_weights = len(layer_node_nr)-1
    for i in range(0,nr_weights):
        weight_delta.append(np.dot(output_layer_values[i].transpose(),delta[i]))

def new(output):
    wdelta3 = (output_layer_values[2]-output)*sigmoid_der(input_layer_values[2])
    w_delta.append(np.dot(output_layer_values[1].transpose(),wdelta3))
    wdelta2 = np.dot(wdelta3,network_weight[1].transpose())*sigmoid_der(input_layer_values[1])
    w_delta.append(np.dot(output_layer_values[0].transpose(),wdelta2))
    w_delta.reverse()

def adjust_weights(scale):
    nr_weights = len(layer_node_nr)-1
    for i in range(0,nr_weights):
        network_weight[i] = network_weight[i] - (scale*weight_delta[i])

def learn_network(input_value, output_value, scale):
    forward(input_value)
    calculate_delta(output_value)
    calculate_weight_delta()
    #new(output_value)
    adjust_weights(scale)
    error = np.sum((output_value-output_layer_values[-1])**2)
    return error


##layer_node_nr = [2,3,1]
##W = np.array([[0.3,0.1,0.8],[0.6,0.5,0.4]])
##network_weight.append(W)
##W = np.array([[0.2],[0.3],[0.9]])
##network_weight.append(W)
##inp = np.array([[1,2]])
##out = np.array([[0.8]])
##
##forward(inp)
##calculate_delta(out)
##calculate_weight_delta()
##print(weight_delta)
##w_delta = []
##wdelta3 = (output_layer_values[2]-out)*sigmoid_der(input_layer_values[2])
##w_delta.append(np.dot(output_layer_values[1].transpose(),wdelta3))
##wdelta2 = np.dot(wdelta3,network_weight[1].transpose())*sigmoid_der(input_layer_values[1])
##w_delta.append(np.dot(output_layer_values[0].transpose(),wdelta2))
##w_delta.reverse()
##print(w_delta)



def func(x,y):
    return (((2*x + y)-3)/17)

x = rd.uniform(1,5)
y = rd.uniform(2,10)
out = func(x,y)
layer_node_nr = [2,3,2,1]
weight_initialize()
print(network_weight)
for i in range(1,100000):
    input_layer_values = []
    output_layer_values = []
    delta = []
    weight_delta = []
    x = rd.uniform(1,5)
    y = rd.uniform(2,10)
    out = func(x,y)
    learn_network(np.array([[x,y]]),np.array([[out]]),0.1)

for i in range(1,10):
    x = rd.uniform(1,5)
    y = rd.uniform(2,10)
    out = func(x,y)
    print(net_response(np.array([[x,y]])))
    print(out)
print(network_weight)
##layer_node_nr = [2,3,1]
##W = np.array([[0.3,0.1,0.8],[0.6,0.5,0.4]])
##network_weight.append(W)
##W = np.array([[0.2],[0.3],[0.9]])
##network_weight.append(W)
##inp = np.array([[1,2]])
##out = np.array([[0.8]])
##forward(inp)
##calculate_delta(out)
##calculate_weight_delta()
##print(weight_delta)

##layer_node_nr = [2,2,1]
##weight_initialize()
##inp = [[0,0],[0,1],[1,0],[1,1]]
##out = [0.05,0.95,0.95,0.05]
##for i in range(1,100000):
##    for j in range(0,4):
##        input_layer_values = []
##        output_layer_values = []
##        delta = []
##        weight_delta = []
##        w_delta = []
##        learn_network(np.array([inp[j]]),np.array([[out[j]]]),10)
##        #learn_network(np.array([inp[0]]),np.array([[out[0]]]),0.5)
####
##
##for j in range(0,4):
##    print(net_response(np.array([inp[j]])))
##        #learn_network(np.array([inp[0]]),np.array([[out[0]]]),0.5)
