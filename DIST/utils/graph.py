import torch
import math
import math
import numpy as np
from torch_geometric.utils import (remove_self_loops, degree, 
                                   batched_negative_sampling)
from torch_geometric.utils.num_nodes import maybe_num_nodes

def split_batch(g):
    split = degree(g.batch[g.edge_index[0]], dtype=torch.long).tolist()
    edge_indices = torch.split(g.edge_index, split, dim=1)
    num_nodes = degree(g.batch, dtype=torch.long)
    cum_nodes = torch.cat([g.batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]])
    num_edges = torch.tensor([e.size(1) for e in edge_indices], dtype=torch.long).to(g.x.device)
    cum_edges = torch.cat([g.batch.new_zeros(1), num_edges.cumsum(dim=0)[:-1]])

    return edge_indices, num_nodes, cum_nodes, num_edges, cum_edges

        
def split_graph(data, edge_score, ratio):
    causal_edge_index = torch.LongTensor([[],[]]).to(data.x.device)
    causal_edge_weight = torch.tensor([]).to(data.x.device)
    causal_edge_attr = torch.tensor([]).to(data.x.device)
    conf_edge_index = torch.LongTensor([[],[]]).to(data.x.device)
    conf_edge_weight = torch.tensor([]).to(data.x.device)
    conf_edge_attr = torch.tensor([]).to(data.x.device)

    edge_indices, _, _, num_edges, cum_edges = split_batch(data)
    for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):
        n_reserve =  int(ratio * N)
        edge_attr = data.edge_attr[C:C+N]
        single_mask = edge_score[C:C+N]
        single_mask_detach = edge_score[C:C+N].detach().cpu().numpy()
        rank = np.argpartition(-single_mask_detach, n_reserve)
        idx_reserve, idx_drop = rank[:n_reserve], rank[n_reserve:]

        causal_edge_index = torch.cat([causal_edge_index, edge_index[:, idx_reserve]], dim=1)
        conf_edge_index = torch.cat([conf_edge_index, edge_index[:, idx_drop]], dim=1)

        causal_edge_weight = torch.cat([causal_edge_weight, single_mask[idx_reserve]])
        conf_edge_weight = torch.cat([conf_edge_weight,  -1 * single_mask[idx_drop]])

        causal_edge_attr = torch.cat([causal_edge_attr, edge_attr[idx_reserve]])
        conf_edge_attr = torch.cat([conf_edge_attr, edge_attr[idx_drop]])
    return (causal_edge_index, causal_edge_attr, causal_edge_weight), \
        (conf_edge_index, conf_edge_attr, conf_edge_weight)

def relabel(x, edge_index, batch, pos=None):
        
    num_nodes = x.size(0)
    sub_nodes = torch.unique(edge_index)
    x = x[sub_nodes]
    batch = batch[sub_nodes]
    row, col = edge_index
    # remapping the nodes in the explanatory subgraph to new ids.
    node_idx = row.new_full((num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
    edge_index = node_idx[edge_index]
    if pos is not None:
        pos = pos[sub_nodes]
    return x, edge_index, batch, pos