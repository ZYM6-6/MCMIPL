import math
import random
import numpy as np
import sys
from tqdm import tqdm
import pickle
from easydict import EasyDict
from Graph_generate.lastfm_data_process import LastFmDataset
from Graph_generate.lastfm_star_data_process import LastFmStarDataset
from Graph_generate.lastfm_graph import LastFmGraph
from Graph_generate.yelp_data_process import YelpDataset
from Graph_generate.yelp_graph import YelpGraph
# from Graph_generate.yelp_data_process import YelpDataset
from Graph_generate.book_graph import BookGraph
class BookDataset(object):
    def __init__(self):
        with open('./tmp/book/kg.pkl','rb') as f:
            kg=pickle.load(f)
        entity_id=list(kg.G['user'].keys())
        m=EasyDict(id=entity_id, value_len=max(entity_id)+1)
        setattr(self,'user',m)
        
        entity_id=list(kg.G['item'].keys())
        m=EasyDict(id=entity_id, value_len=max(entity_id)+1)
        setattr(self,'item',m)
        
        entity_id=list(kg.G['feature'].keys())
        m=EasyDict(id=entity_id, value_len=max(max(entity_id)+1,988))
        setattr(self,'feature',m)

