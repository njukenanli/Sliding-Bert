train=True
test=False
import os
from data import data
from bert import *
if train:
    d=data()
    d.load()
    n=NNmanager()
    if os.path.exists(r'BertClassifier2.pth'):
        n.load()
        print('load previous model')
    else:
        print('new model')
    n.train(30,d.model_input[:1282],[1282:])
if test:
    d=data()
    d.load()
    n=NNmanager()
    n.load()
    n.predlist(d.model_input[1282:])
    
    


