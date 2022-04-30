import math
import random
import numpy as np
import sys
from tqdm import tqdm
import pickle
from easydict import EasyDict
class LastFmStarDataset(object):
    def __init__(self):
        with open('./tmp/last_fm_star/kg.pkl','rb') as f:
            kg=pickle.load(f)
        entity_id=list(kg.G['user'].keys())
        m=EasyDict(id=entity_id, value_len=max(entity_id)+1)
        setattr(self,'user',m)
        
        entity_id=list(kg.G['item'].keys())
        m=EasyDict(id=entity_id, value_len=max(entity_id)+1)
        setattr(self,'item',m)
        
        entity_id=list(kg.G['feature'].keys())
        m=EasyDict(id=entity_id, value_len=max(entity_id)+1)
        setattr(self,'feature',m)