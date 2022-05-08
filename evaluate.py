
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

#TODO select env
from RL.env_multi_choice_question import MultiChoiceRecommendEnv
from RL.RL_evaluate import dqn_evaluate
from RL_model import Agent, ReplayMemoryPER
from multi_interest import GraphEncoder
import time
import warnings
from construct_graph import get_graph
warnings.filterwarnings("ignore")
EnvDict = {
    LAST_FM_STAR: MultiChoiceRecommendEnv,
    YELP_STAR: MultiChoiceRecommendEnv,
    BOOK: MultiChoiceRecommendEnv,
    MOVIE:MultiChoiceRecommendEnv

    }

FeatureDict = {
    LAST_FM_STAR: 'feature',
    YELP_STAR: 'feature',
    MOVIE: 'feature',
    BOOK:'feature'
}


def evaluate(args, kg, dataset, filename):
    test_env = EnvDict[args.data_name](kg, dataset, args.data_name, args.embed, seed=args.seed, max_turn=args.max_turn,
                                       cand_num=args.cand_num, cand_item_num=args.cand_item_num, attr_num=args.attr_num, mode='test', ask_num=args.ask_num, entropy_way=args.entropy_method,
                                       fm_epoch=args.fm_epoch)
    set_random_seed(args.seed)
    memory = ReplayMemoryPER(args.memory_size)
    embed = torch.FloatTensor(np.concatenate((test_env.ui_embeds, test_env.feature_emb, np.zeros((1,test_env.ui_embeds.shape[1]))), axis=0))
    G=get_graph(test_env.user_length,test_env.item_length,test_env.feature_length,args.data_name)
    gcn_net = GraphEncoder(graph=G,device=args.device, entity=embed.size(0), emb_size=embed.size(1), kg=kg, embeddings=embed, \
        fix_emb=args.fix_emb, seq=args.seq, gcn=args.gcn, hidden_size=args.hidden,u=test_env.user_length,v=test_env.item_length,f=test_env.feature_length).to(args.device)
    agent = Agent(device=args.device, memory=memory, state_size=args.hidden, action_size=embed.size(1), \
        hidden_size=args.hidden, gcn_net=gcn_net, learning_rate=args.learning_rate, l2_norm=args.l2_norm, PADDING_ID=embed.size(0)-1)
    print('Staring loading rl model in epoch {}'.format(args.load_rl_epoch))
    agent.load_model(data_name=args.data_name, filename=filename, epoch_user=args.load_rl_epoch)
    SR5_mean, SR10_mean, SR15_mean, AvgT_mean, Rank_mean = dqn_evaluate(args, kg, dataset, agent, filename, 0)
    if SR15_mean>SR15_best:
        SR5_best, SR10_best, SR15_best, AvgT_best, Rank_best=SR5_mean, SR10_mean, SR15_mean, AvgT_mean, Rank_mean
    print("best!!!!!!!!!SR5:{}, SR10:{}, SR15:{}, AvgT:{}, Rank:{}!!!!".format(SR5_best, SR10_best, SR15_best, AvgT_best, Rank_best))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-seed', type=int, default=1, help='random seed.')
    parser.add_argument('--gpu', type=str, default='1', help='gpu device.')
    parser.add_argument('--epochs', '-me', type=int, default=50000, help='the number of RL train epoch')
    parser.add_argument('--fm_epoch', type=int, default=0, help='the epoch of FM embedding')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate.')
    parser.add_argument('--l2_norm', type=float, default=1e-6, help='l2 regularization.')
    parser.add_argument('--hidden', type=int, default=100, help='number of samples')
    parser.add_argument('--memory_size', type=int, default=5000, help='size of memory ')

    parser.add_argument('--data_name', type=str, default=YELP_STAR, choices=[ MOVIE,LAST_FM_STAR, YELP_STAR,BOOK],
                        help='One of { LAST_FM_STAR, YELP_STAR, BOOK}.')
    parser.add_argument('--entropy_method', type=str, default='match', help='entropy_method is one of {entropy, weight entropy,match}')
    # Although the performance of 'weighted entropy' is better, 'entropy' is an alternative method considering the time cost.
    parser.add_argument('--max_turn', type=int, default=15, help='max conversation turn')
    parser.add_argument('--attr_num', type=int, help='the number of attributes')
    parser.add_argument('--mode', type=str, default='train', help='the mode in [train, test]')
    parser.add_argument('--ask_num', type=int, default=1, help='the number of features asked in a turn')
    parser.add_argument('--load_rl_epoch', type=int, default=0, help='the epoch of loading RL model')

    parser.add_argument('--sample_times', type=int, default=100, help='the epoch of sampling')
    
    parser.add_argument('--observe_num', type=int, default=100, help='the observe_num')
    parser.add_argument('--max_steps', type=int, default=100, help='max training steps')
    parser.add_argument('--eval_num', type=int, default=0, help='the number of steps to evaluate RL model and metric')
    parser.add_argument('--save_num', type=int, default=10, help='the number of steps to save RL model and metric')
    parser.add_argument('--cand_num', type=int, default=10, help='candidate sampling number')
    parser.add_argument('--cand_item_num', type=int, default=10, help='candidate item sampling number')
    parser.add_argument('--fix_emb', action='store_false', help='fix embedding or not')
    parser.add_argument('--embed', type=str, default='transe', help='pretrained embeddings')
    parser.add_argument('--seq', type=str, default='transformer', choices=['rnn', 'transformer', 'mean'], help='sequential learning method')
    parser.add_argument('--gcn', action='store_false', help='use GCN or not')


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print(args.device)
    print('data_set:{}'.format(args.data_name))
    kg = load_kg(args.data_name)
    #reset attr_num
    feature_name = FeatureDict[args.data_name]
    feature_length = len(kg.G[feature_name].keys())
    print('dataset:{}'.format(args.data_name))
    args.attr_num = feature_length  # set attr_num  = feature_length

    dataset = load_dataset(args.data_name)
    filename = 'train-data-{}-RL-cand_num-{}-cand_item_num-{}-embed-{}-seq-{}-gcn-{}'.format(
        args.data_name, args.cand_num, args.cand_item_num, args.embed, args.seq, args.gcn)
    evaluate(args, kg, dataset, filename)

if __name__ == '__main__':
    main()