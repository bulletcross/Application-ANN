from QL_function import *
import numpy as np
import random as rd

#pybrain ANN tools
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SigmoidLayer, LinearLayer

iteration = 1000
gamma = 0.8
#network training iteration
net_iteration = 1000

#pybrian ANN initialization
net = buildNetwork(4,100,100,1, bias=True, hiddenclass = SigmoidLayer, outclass = LinearLayer)
ds = SupervisedDataSet(4,1)


"""Utilities for value iteration algorithm"""
#This function has to be chnaged to obtain Q value from network
def get_state_value(state):
    return net.activate([state[0],state[1],state[2],state[3]])


##def get_state_value(state,state_value):
##    if state[0] == -1:
##        return state_value[0][state[1]][state[2]][state[3]]
##    else:
##        return state_value[state[0]][state[1]][state[2]][state[3]]

##def set_state_value(state,state_value,new_value):
##    if state[0] == -1:
##        state_value[0][state[1]][state[2]][state[3]] = new_value
##    else:
##        state_value[state[0]][state[1]][state[2]][state[3]] = new_value
##
##def set_action_value(state,action_value,action):
##    if state[0] == -1:
##        action_value[0][state[1]][state[2]][state[3]] = action
##    else:
##        action_value[state[0]][state[1]][state[2]][state[3]] = action

"""Generates all states in list, easy to iterate in value iteration algorithm"""
def generate_states():
    output = []
    for i in range(-1,2,2):
        for j in range(0,4):
            for k in range(0,4):
                for l in range(0,4):
                    output.append((i,j,k,l))
    return output

"""Value iteration algorithm is an iterative version of solving
    Markove Decision Process where state information are observable.
    gamma is the learing factor and higher iteration will give
    state value at higher precision but higher iteraton are not
    useful as only action value are required which will be same no
    matter how many iteration the algorithm runs"""
def learn_network():
    #initialize the value and action as a function of state
    #state_value = np.zeros((2,4,4,4))
    #action_value = np.ones((2,4,4,4))
    states = generate_states()
    actions = [0,-1,1]

    #Control the iteration from variable
    for i in range(0,iteration):
        for state in states:
            best_q_value = 0
            for action in actions:
                
                possible_states = state_transition(state,action)
                summation = 0
                
                for new_state in possible_states:
                    #get_state_value has to be chnaged internally
                    summation +=  transition_probability(state,action,new_state)*get_state_value(new_state)
                    #summation +=  transition_probability(state,action,new_state)*get_state_value(new_state,state_value)
                q_value = reward(state) + gamma*summation
                #ntake best Q value among actions and update
                if best_q_value < q_value:
                    best_q_value = q_value   
                #if get_state_value(state,state_value) < q_value:
                    #set_state_value(state,state_value,q_value)
                    #set_action_value(state,action_value,action)
                    #update network weight here

            #update network with best Q value obtained for state state
            for j in range(0,net_iteration):
                ds.addSample((state[0],state[1],state[2],state[3]),(best_q_value))
                trainer = BackpropTrainer(net,ds)
                trainer.train()
                ds.clear()
    
    #print state_value
    #return action_value

##print len(generate_states())
##print len(learn())
