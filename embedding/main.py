import sys
sys.path.append('/root/autodl-tmp/pgm/')


import torch
import argparse
import time
from Model import Model, RiemannianFeature
from logger import create_logger
from geoopt.optim import RiemannianAdam
import numpy as np
import pickle
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

from data.data_generators import load_dataset
import pickle
from torch_geometric.data import Data,DataLoader
from utils.data_loader import dataloader
from torch_geometric.utils import negative_sampling
from utils.graph_utils import init_features,graphs_to_tensor,pad_adjs

dataset_set={"max_node_num":18,"max_feat_num":17}

def main(configs):
    device = torch.device('cuda:0')

    file_path = "../data/ego_small.pkl"
    with open(file_path, "rb") as f:
        graph_list = pickle.load(f)

    adjs_tensor = graphs_to_tensor(graph_list, dataset_set["max_node_num"])
    x_tensor = init_features("zeros", adjs_tensor, dataset_set["max_feat_num"])

    all_edge_index = []
    i=0
    data_x = []
    for graph in graph_list:
        num_nodes = dataset_set["max_node_num"]
        edge_index = torch.tensor(list(graph.edges)).t().contiguous().to(device)
        data = Data(x=x_tensor[i], edge_index=edge_index, num_nodes=num_nodes).to(device)
        data_x.append(data)
        all_edge_index.append(data.edge_index)
        i=i+1

    all_edge_index = torch.cat(all_edge_index, dim=1)
    neg_edge = negative_sampling(all_edge_index)
    data_loader = DataLoader(data_x, batch_size=int(len(data_x)))

    all_embeddings = []
    for batch in data_loader:
        features = batch.x.to(device)

        edge_index = batch.edge_index.to(device)

        # Initialize model and optimizer
        model = Model(backbone=configs.backbone, n_layers=configs.n_layers, in_features=dataset_set["max_feat_num"],
                      embed_features=configs.embed_features, hidden_features=configs.hidden_features,
                      n_heads=configs.n_heads, drop_edge=configs.drop_edge, drop_node=configs.drop_edge,
                      num_factors=configs.num_factors, dimensions=configs.dimensions, d_embeds=configs.d_embeds,
                      temperature=configs.temperature, device=device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, weight_decay=configs.w_decay)
        Riemann_embeds_getter = RiemannianFeature(features.shape[0], configs.dimensions,
                                                  configs.init_curvature, configs.num_factors,
                                                  learnable=configs.learnable).to(device)
        r_optim = RiemannianAdam(Riemann_embeds_getter.parameters(), lr=configs.lr_Riemann,
                                 weight_decay=configs.w_decay, stabilize=100)

        # Training
        for epoch in range(1, configs.epochs + 1):
            model.train()
            Riemann_embeds_getter.train()

            # Forward pass
            # embeddings, loss = model(features, edge_index, None, None, Riemann_embeds_getter)
            embeddings, loss = model(features, all_edge_index, None, None, Riemann_embeds_getter)

            # Backward pass
            r_optim.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            r_optim.step()
            optimizer.step()

            print(f"Epoch {epoch}: train_loss={loss.item()}")


        # Evaluation
        model.eval()
        Riemann_embeds_getter.eval()
        embeddings, _ = model(features, all_edge_index, None, None, Riemann_embeds_getter)

        # Convert embeddings to NumPy array and append to list
        embeddings_np = embeddings.detach().cpu().numpy()
        all_embeddings.append(embeddings_np)

    torch.save(Riemann_embeds_getter.state_dict(), f'riemann_embeds_grid.pth')
    test = RiemannianFeature(features.shape[0], configs.dimensions,
                                              configs.init_curvature, configs.num_factors,
                                              learnable=configs.learnable).to(device)
    state_dict = torch.load('riemann_embeds_grid.pth')

    # 使用加载的参数更新你的模型
    test.load_state_dict(state_dict)
    # Concatenate all embeddings into a single array
    all_embeddings = np.concatenate(all_embeddings, axis=0)

    # Save embeddings
    if configs.save_embeds:
        np.save(configs.save_embeds, all_embeddings)



    # param_model=RiemannianFeature(features.shape[0], configs.dimensions,
    #                                               configs.init_curvature, configs.num_factors,
    #                                               learnable=configs.learnable).to(device)
    # param_model.load_state_dict(torch.load('riemann_embeds.pth'))
    # manifolds=param_model.manifolds
    # c=[]
    #
    # for i in param_model.manifolds:
    #     c.append()
    # a=1

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='')
#
#         # Experiment settings
#     parser.add_argument('--downstream_task', type=str, default='NC', choices=['NC', 'LP', "Motif"])
#     parser.add_argument('--dataset', type=str, default='facebook', choices=['Cora', 'Citeseer', 'Pubmed', 'airport', 'amazon', 'facebook'])
#     parser.add_argument('--root_path', type=str, default='./datasets')
#     parser.add_argument('--version', type=str, default="run")
#     parser.add_argument('--save_embeds', type=str, default="./results/ego_small.npy")
#         # parser.add_argument('--log_path', type=str, default="./results/cls_Cora.log")
#
#         # Riemannian Embeds
#     parser.add_argument('--num_factors', type=int, default=1, help='number of product factors')
#     parser.add_argument('--dimensions', type=int, nargs='+', default=dataset_set["max_feat_num"], help='dimension of Riemannian embedding')
#     parser.add_argument('--d_embeds', type=int, default=dataset_set["max_feat_num"], help='dimension of laplacian features')
#     parser.add_argument('--init_curvature', type=float, default=-1, help='initial curvature')
#     parser.add_argument('--learnable', action='store_false')
#
#         # Contrastive Learning Module
#     parser.add_argument('--backbone', type=str, default='gcn', choices=['gcn', 'gat', 'sage'])
#     parser.add_argument('--epochs', type=int, default=100)
#     parser.add_argument('--hidden_features', type=int, default=16)
#     parser.add_argument('--embed_features', type=int, default=dataset_set["max_feat_num"], help='dimensions of graph embedding')
#     parser.add_argument('--n_layers', type=int, default=2)
#     parser.add_argument('--drop_node', type=float, default=0.0)
#     parser.add_argument('--drop_edge', type=float, default=0.0)
#     parser.add_argument('--lr', type=float, default=0.01)
#     parser.add_argument('--lr_Riemann', type=float, default=0.01)
#     parser.add_argument('--w_decay', type=float, default=0.)
#     parser.add_argument('--n_heads', type=int, default=4, help='number of attention heads')
#     parser.add_argument('--t', type=float, default=1., help='for Fermi-Dirac decoder')
#     parser.add_argument('--r', type=float, default=2., help='Fermi-Dirac decoder')
#     parser.add_argument('--temperature', type=float, default=0.2, help='temperature of contrastive loss')
#
#     args = parser.parse_args()
#
#     main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

        # Experiment settings
    parser.add_argument('--downstream_task', type=str, default='NC', choices=['NC', 'LP', "Motif"])
    parser.add_argument('--dataset', type=str, default='facebook', choices=['Cora', 'Citeseer', 'Pubmed', 'airport', 'amazon', 'facebook'])
    parser.add_argument('--root_path', type=str, default='./datasets')
    parser.add_argument('--version', type=str, default="run")
    parser.add_argument('--save_embeds', type=str, default="./results/ego_small.npy")
        # parser.add_argument('--log_path', type=str, default="./results/cls_Cora.log")

        # Riemannian Embeds
    parser.add_argument('--num_factors', type=int, default=3, help='number of product factors')
    parser.add_argument('--dimensions', type=int, nargs='+', default=10, help='dimension of Riemannian embedding')
    parser.add_argument('--d_embeds', type=int, default=16, help='dimension of laplacian features')
    parser.add_argument('--init_curvature', type=float, default=[-1,-1,-1], help='initial curvature')
    parser.add_argument('--learnable', action='store_false')

        # Contrastive Learning Module
    parser.add_argument('--backbone', type=str, default='gcn', choices=['gcn', 'gat', 'sage'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--hidden_features', type=int, default=16)
    parser.add_argument('--embed_features', type=int, default=10, help='dimensions of graph embedding')
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--drop_node', type=float, default=0.0)
    parser.add_argument('--drop_edge', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_Riemann', type=float, default=0.01)
    parser.add_argument('--w_decay', type=float, default=0.)
    parser.add_argument('--n_heads', type=int, default=4, help='number of attention heads')
    parser.add_argument('--t', type=float, default=1., help='for Fermi-Dirac decoder')
    parser.add_argument('--r', type=float, default=2., help='Fermi-Dirac decoder')
    parser.add_argument('--temperature', type=float, default=0.2, help='temperature of contrastive loss')

    args = parser.parse_args()

    main(args)