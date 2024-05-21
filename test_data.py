import argparse

import torch
# from utils import *
from parsers.config import get_config
from utils.loader import load_data
from parsers.parser import Parser
from data.data_generators import load_dataset
import pickle
from torch_geometric.data import Data
from utils.data_loader import dataloader
from torch_geometric.utils import negative_sampling
from utils.graph_utils import init_features,graphs_to_tensor,pad_adjs
import networkx as nx
import numpy as np

def main(args):
    config=get_config(args.config,args.seed)
    graph_list = load_dataset(data_dir=config.data.dir, file_name=config.data.data)

    adjs_tensor = graphs_to_tensor(graph_list, config.data.max_node_num)
    x_tensor = init_features(config.data.init, adjs_tensor, config.data.max_feat_num)

    all_edge_index = []

    for graph in  graph_list:
        num_nodes = config.data.max_node_num
        edge_index = torch.tensor(list(graph.edges)).t().contiguous()
        data = Data(edge_index=edge_index, num_nodes=num_nodes)

        all_edge_index.append(data.edge_index)


    # features=x_tensor.to(torch.float32).to(device)

    # train_loader,test_loader=load_data(config)
    c=1




if __name__=="__main__":
    data=np.load("./embedding/results/community_small.npy")

    # for graph in data:
    #     a=graph

    reshape_data=data.reshape(-1,20,40)

    # a=reshape_data[0]
    b=reshape_data[:,:,-10:]
    c=torch.tensor(b)

    d=1


    # a=torch.randn(3,3,3)
    # c=a[-1:,:,:]
    #
    #
    # type_args=argparse.ArgumentParser()
    # type_args.add_argument("--type",default="train")
    # type_args.add_argument("--config",default="community_small")
    # type_args.add_argument("--seed",default=42)
    # args = type_args.parse_args()
    # main(args)



# # 指定文件路径
# class Args:
#     def __init__(self, work_type,config,seed):
#         self.type = work_type
#         self.config=config
#         self.seed=seed
#
#
# def main(work_type_args):
#     ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
#     # args = Parser().parse()
#     args=work_type_args
#     config = get_config(args.config, args.seed)
#
#     # -------- Train --------
#     if work_type_args.type == 'train':
#         trainer = Trainer(config)
#         ckpt = trainer.train(ts)
#         if 'sample' in config.keys():
#             config.ckpt = ckpt
#             sampler = Sampler(config)
#             sampler.sample()
#
#     # -------- Generation --------
#     elif work_type_args.type == 'sample':
#         if config.data.data in ['QM9', 'ZINC250k']:
#             sampler = Sampler_mol(config)
#         else:
#             sampler = Sampler(config)
#         sampler.sample()
#
#     else:
#         raise ValueError(f'Wrong type : {work_type_args.type}')
#
#
# if __name__ == '__main__':
#     work_type_args = Args('sample',"community_small",42)  # Change this to 'sample' if needed
#     main(work_type_args)