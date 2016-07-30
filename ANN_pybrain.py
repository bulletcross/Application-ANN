import numpy as np
import random as rd
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

def func(x,y):
    return (((2*x + y)-3)/17)

net = buildNetwork(2,3,1)
ds = SupervisedDataSet(2,1)
for i in range(0,10000):
    x = rd.uniform(1,5)
    y = rd.uniform(2,10)
    out = func(x,y)
    ds.addSample((x,y),(out))
    trainer = BackpropTrainer(net,ds)
    trainer.train()
    ds.clear()

for i in range(1,10):
    x = rd.uniform(1,5)
    y = rd.uniform(2,10)
    out = func(x,y)
    print(net.activate([x,y]))
    print(out)
