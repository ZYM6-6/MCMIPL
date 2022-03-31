import math
import random
import numpy as np
import os
import sys
from tqdm import tqdm
# sys.path.append('..')

from collections import namedtuple
import argparse
from itertools import count, chain
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import *
from sum_tree import SumTree


class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=100):
        super(DQN, self).__init__()
        # V(s)
        self.fc2_value = nn.Linear(hidden_size, hidden_size)
        self.out_value = nn.Linear(hidden_size, 1)
        # Q(s,a)
        self.fc2_advantage = nn.Linear(hidden_size + action_size, hidden_size)   
        self.out_advantage = nn.Linear(hidden_size, 1)

    def forward(self, X,  z,choose_action=True,Op=False):
        m=[]
        for x in X:
            value = self.out_value(F.relu(self.fc2_value(x))).squeeze(dim=2) #[N*1*1]
            if choose_action:
                x = x.repeat(1, z.size(1), 1)
            state_cat_action = torch.cat((x,z),dim=2)
            advantage = self.out_advantage(F.relu(self.fc2_advantage(state_cat_action))).squeeze(dim=2) #[N*K]

            if choose_action:
                qsa = advantage + value - advantage.mean(dim=1, keepdim=True)
            else:
                qsa = advantage + value
            m.append(qsa)
        
        # enablePrint()
        # ipdb.set_trace()
        qsa,_=torch.max(torch.stack(m,dim=0).squeeze(0),dim=0)

        return qsa
