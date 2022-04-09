import math
import random
import numpy as np
import sys
from tqdm import tqdm
import pickle
import json
with open('./data/book/fea_item/item_feature.pkl','rb') as f:
    item_feature=pickle.load(f)
with open('./data/book/fea_item/small_to_large.pkl','rb') as f:
    small_to_large=pickle.load(f)
feature_index={}
i=0
for key in small_to_large.keys():
    if key in feature_index:
        continue
    else:
        feature_index[key]=i
        i+=1
class BookGraph(object):
    def __init__(self):
        self.G = dict()
        self.__get_user__()
        self.__get_item__()
        self.__get_feature__()
    def __get_user__(self):
        with open('./data/book/UI_Interaction_data/review_dict_valid.json', 'r', encoding='utf-8') as f:
            ui_train=json.load(f)
            self.G['user']={}
            for user in tqdm(ui_train):
                self.G['user'][int(user)]={}
                self.G['user'][int(user)]['interact']=tuple(ui_train[user])
                self.G['user'][int(user)]['friends']=tuple(())
                self.G['user'][int(user)]['like']=tuple(())
    def __get_item__(self):
        self.G['item']={}
        feature_index={}
        i=0
        for key in small_to_large.keys():
            if key in feature_index:
                continue
            else:
                feature_index[key]=i
                i+=1
        for item in item_feature:
            self.G['item'][item]={}
            fea=[]
            for feature in item_feature[item]:
                fea.append(feature_index[feature])
            self.G['item'][item]['belong_to']=tuple(set(fea))
            self.G['item'][item]['interact']=tuple(())
            self.G['item'][item]['belong_to_large']=tuple(())
        for user in self.G['user']:
            for item in self.G['user'][user]['interact']:
                self.G['item'][item]['interact']+=tuple([user])
    def __get_feature__(self):
        self.G['feature']={}
        feature_index={}
        i=0
        for key in small_to_large.keys():
            if key in feature_index:
                continue
            else:
                feature_index[key]=i
                i+=1
        for key in small_to_large:
            idx=feature_index[key]
            self.G['feature'][idx]={}
            self.G['feature'][idx]['link_to_feature']=tuple(small_to_large[key])
            self.G['feature'][idx]['like']=tuple(())
            self.G['feature'][idx]['belong_to']=tuple(())
        for item in self.G['item']:
            for feature in self.G['item'][item]['belong_to']:
                self.G['feature'][feature]['belong_to']+=tuple([item])
            

            