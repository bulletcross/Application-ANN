import numpy as np
import random as rd
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SigmoidLayer, LinearLayer


"""Training test on non linear functon"""
def func(x,y):
    return ((x*x + y))

net = buildNetwork(2,100,1, bias=True, hiddenclass = SigmoidLayer, outclass = LinearLayer)
ds = SupervisedDataSet(2,1)
for i in range(0,10000):
    x = rd.uniform(1,5)
    y = rd.uniform(2,10)
    out = func(x,y)
    ds.addSample((x,y),(out))
    trainer = BackpropTrainer(net,ds)
    #trainer = BackpropTrainer(net,ds,learningrate = 0.01, momentum = 0.1, verbose = True)
    trainer.train()
    #trainer.trainUntilConvergence(maxEpochs = 10)
    ds.clear()

for i in range(1,10):
    x = rd.uniform(1,5)
    y = rd.uniform(2,10)
    out = func(x,y)
    print(net.activate([x,y]))
    print(out)
