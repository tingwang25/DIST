import math
from torch_geometric.nn.inits import glorot
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax, remove_self_loops, add_self_loops
from torch_scatter import scatter
from torch_geometric.utils import dense_to_sparse
import numpy as np
from torch_geometric.nn import GraphConv, BatchNorm, global_max_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import degree

class EstimationGate(nn.Module):
    """The estimation gate module."""

    def __init__(self, node_emb_dim, time_emb_dim, history_dim):
        """
        node_emb_dim: node_embedding
        time_emb_dim: time_feature
        history_dim: embedded observation_data
        """
        super().__init__()
        self.fully_connected_layer_1 = nn.Linear(2 * node_emb_dim + time_emb_dim * 2, history_dim)
        self.activation = nn.ReLU()
        self.fully_connected_layer_2 = nn.Linear(history_dim, 1)

    def forward(self, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat, history_data):
        """Generate gate value in (0, 1) based on current node and time step embeddings to roughly estimating the proportion of the two hidden time series."""

        batch_size, seq_length, _, _ = time_in_day_feat.shape
        estimation_gate_feat = torch.cat([time_in_day_feat, day_in_week_feat, node_embedding_u.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length,  -1, -1), node_embedding_d.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length,  -1, -1)], dim=-1)
        hidden = self.fully_connected_layer_1(estimation_gate_feat)
        hidden = self.activation(hidden)
        # activation
        estimation_gate = torch.sigmoid(self.fully_connected_layer_2(hidden))[:, -history_data.shape[1]:, :, :]
        history_data = history_data * estimation_gate
        return history_data
    
class EstimationGate_D(nn.Module):
    """The estimation gate module for start node and end node respectively."""

    def __init__(self, node_emb_dim, time_emb_dim, history_dim):
        """
        node_emb_dim: node_embedding
        time_emb_dim: time_feature
        history_dim: embedded observation_data
        """
        super().__init__()
        self.fully_connected_layer_1 = nn.Linear(node_emb_dim + time_emb_dim * 2, history_dim)
        self.activation = nn.ReLU()
        self.fully_connected_layer_2 = nn.Linear(history_dim, 1)

    def forward(self, node_embedding, time_in_day_feat, day_in_week_feat, history_data):
        """Generate gate value in (0, 1) based on current node and time step embeddings to roughly estimating the proportion of the two hidden time series."""

        batch_size, seq_length, _, _ = time_in_day_feat.shape
        estimation_gate_feat = torch.cat([time_in_day_feat, day_in_week_feat, node_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length,  -1, -1)], dim=-1)
        hidden = self.fully_connected_layer_1(estimation_gate_feat)
        hidden = self.activation(hidden)
        estimation_gate = torch.sigmoid(self.fully_connected_layer_2(hidden))[:, -history_data.shape[1]:, :, :] 
        history_data = history_data * estimation_gate
        return history_data

class GraphLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, inter_dim=-1, **kwargs):
        super(GraphLayer, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.__alpha__ = None

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)

        self.att_i = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_i = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = nn.Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)

        zeros(self.att_em_i)
        zeros(self.att_em_j)

        zeros(self.bias)

    def forward(self, x, edge_index, embedding, return_attention_weights=False):
        """"""
        if torch.is_tensor(x):
            x = self.lin(x)
            x = (x, x)
        else:
            x = (self.lin(x[0]), self.lin(x[1]))

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x[1].size(self.node_dim))

        out = self.propagate(edge_index, x=x, embedding=embedding, edges=edge_index,
                             return_attention_weights=return_attention_weights)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_i, x_j, edge_index_i, size_i,
                embedding,
                edges,
                return_attention_weights):

        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        if embedding is not None:
            embedding_i, embedding_j = embedding[edge_index_i], embedding[edges[0]]
            embedding_i = embedding_i.unsqueeze(1).repeat(1, self.heads, 1)
            embedding_j = embedding_j.unsqueeze(1).repeat(1, self.heads, 1)

            key_i = torch.cat((x_i, embedding_i), dim=-1)
            key_j = torch.cat((x_j, embedding_j), dim=-1)

        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)

        # alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1)
        alpha = (x_i*self.att_i).sum(-1) + (x_j * self.att_j).sum(-1)

        alpha = alpha.view(-1, self.heads, 1)

        alpha = F.leaky_relu(alpha, self.negative_slope)

        self.node_dim = 0
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

        if return_attention_weights:
            self.__alpha__ = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1):
        super(GNNLayer, self).__init__()

        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None):
        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index

        out = self.bn(out)

        return self.relu(out)

class AttModel(torch.nn.Module):
    """
    This is a method for calculating zi and zv by ranking edge_score
    We define edge_score by node similarity
    """
    def __init__(self, args=None):
        super(AttModel, self).__init__()
        self.top_k_ratio = args.causal_ratio  # hyperparam
        self.num_nodes = args.num_nodes
        self.hid_dim = args.nhid*2
        self.device = args.device
        self.batch_size = args.batch_size
        self.use_distance = args.use_distance
        self.normalized_k = args.normalized_k
        self.input_length = args.input_length

        self.direction = args.direction
        self.edge_score = args.edge_score

        self.num_feat = 1

        self.estimation_gate= EstimationGate(node_emb_dim=self.hid_dim, time_emb_dim=self.hid_dim, history_dim=self.hid_dim)
        self.estimation_gate_d = EstimationGate_D(node_emb_dim=self.hid_dim, time_emb_dim=self.hid_dim, history_dim=self.hid_dim)

        # time embedding
        self.T_i_D_emb  = nn.Parameter(torch.empty(288, self.hid_dim))
        self.D_i_W_emb  = nn.Parameter(torch.empty(7, self.hid_dim))
        # node embeddings （trainable）
        self.node_emb_u = nn.Parameter(torch.empty(self.num_nodes, self.hid_dim))
        self.node_emb_d = nn.Parameter(torch.empty(self.num_nodes, self.hid_dim))

        self.mlp_cat = nn.Sequential(
            nn.Linear(self.hid_dim*2, self.hid_dim*4),
            nn.ReLU(),
            nn.Linear(self.hid_dim*4, 1)
        )

        self.mlp_had = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim*2),
            nn.ReLU(),
            nn.Linear(self.hid_dim*2, 1)
        )
        
        # start embedding layer for node feature
        self.embedding  = nn.Linear(self.num_feat, self.hid_dim)

        in_channel = self.input_length*self.hid_dim
        self.node_emb = nn.Linear(in_channel, self.hid_dim)
        self.node_emb_1 = nn.Linear(in_channel, self.hid_dim)   #source
        self.node_emb_2 = nn.Linear(in_channel, self.hid_dim)   #target
        # self.node_emb_12 = nn.Linear(2*self.hid_dim, self.hid_dim)   
        
        self.reset_parameter()
    
    def reset_parameter(self):
        nn.init.xavier_uniform_(self.node_emb_u)
        nn.init.xavier_uniform_(self.node_emb_d)
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)

    def forward(self, ori_data , ori_adj):
        """
        node: trainable params for cal edge
        Args:
            ori_data [B L N C] 
        Return:
            x: [B*N dim]
        """
        # prepare data
        # time slot embedding
        time_in_day_feat = self.T_i_D_emb[(ori_data[:, :, :, self.num_feat] * 288).type(torch.LongTensor)]    # [B, L, N, d]
        day_in_week_feat = self.D_i_W_emb[(ori_data[:, :, :, self.num_feat+1]).type(torch.LongTensor)]          # [B, L, N, d]
        # traffic signals
        history_data = ori_data[:, :, :, :self.num_feat]   # [B, L, N, 1]
        # Start embedding layer
        history_data   = self.embedding(history_data)   # [B, L, N, d]
        # node embeddings
        node_emb_u  = self.node_emb_u  # [N, d] 起
        node_emb_d  = self.node_emb_d  # [N, d] 终

        # cal graph structure
        adaptive_adj = F.softmax(F.relu(torch.mm(node_emb_d, node_emb_u.T)), dim=1)  # [1, N, N] with direction
        adaptive_adj[adaptive_adj < self.normalized_k] = 0  

        if self.use_distance == 1:
            adaptive_adj = ori_adj
        elif self.use_distance == 2:
            adaptive_adj = adaptive_adj + ori_adj
        N = torch.count_nonzero(adaptive_adj).item()  
        edge_index, edge_attr = dense_to_sparse(adaptive_adj)
        row, col = edge_index
        # local st graph for X_G+Temb+Semb
        if(self.direction == 1): #direct
            gated_history_data_1  = self.estimation_gate_d(node_emb_u, time_in_day_feat, day_in_week_feat, history_data)    # [B L N d] 
            gated_history_data_2  = self.estimation_gate_d(node_emb_d, time_in_day_feat, day_in_week_feat, history_data)    # [B L N d] 
            local_st_1 = gated_history_data_1.transpose(1,2) # [B N L d]  
            local_st_2 = gated_history_data_2.transpose(1,2) # [B N L d]  
            _, _, all_feature1, all_feature2 = local_st_1.shape
            local_st_1 = local_st_1.contiguous().view(-1, all_feature1*all_feature2)    # for batch node-level x [B*N L*d]
            local_st_2 = local_st_2.contiguous().view(-1, all_feature1*all_feature2)    # for batch node-level x [B*N L*d]

            x_1 = self.node_emb_1(local_st_1)    # x_embedding [B*N d] 
            x_2 = self.node_emb_2(local_st_2)    # x_embedding [B*N d] 
            if(self.edge_score==0):
                edge_rep = torch.cat([x_1[row], x_2[col]], dim=-1) 
                edge_score = self.mlp_cat(edge_rep).view(-1)   
            elif(self.edge_score==1):
                edge_rep = x_1[row] * x_2[col]  
                edge_score = self.mlp_had(edge_rep).view(-1)    
            # x = x_1 
            x = x_1 * x_2 
                     
        elif(self.direction == 0):
            gated_history_data  = self.estimation_gate(node_emb_u, node_emb_d, time_in_day_feat, day_in_week_feat, history_data)    # [B L N d] 
            local_st = gated_history_data.transpose(1,2) # [B N L d] 
            _, _, all_feature1, all_feature2 = local_st.shape
            local_st = local_st.contiguous().view(-1, all_feature1*all_feature2)    # for batch node-level x [B*N L*d]
            
            x = self.node_emb(local_st)    # x_embedding [B*N d] （
            if(self.edge_score==0):
                edge_rep = torch.cat([x[row], x[col]], dim=-1)  
                edge_score = self.mlp_cat(edge_rep).view(-1)   
            elif(self.edge_score==1):
                edge_rep = x[row] * x[col]  
                edge_score = self.mlp_had(edge_rep).view(-1)   
                 
        causal_edge_index = torch.LongTensor([[],[]]).to(self.device)
        conf_edge_index = torch.LongTensor([[],[]]).to(self.device)
        n_reserve =  int(self.top_k_ratio * N)  
        for i in range(self.batch_size):
            C = i*N 
            single_mask_detach = edge_score[C:C+N].detach().cpu().numpy()
            rank = np.argpartition(-single_mask_detach, n_reserve)
            idx_reserve, idx_drop = rank[:n_reserve], rank[n_reserve:]
            causal_edge_index = torch.cat([causal_edge_index, edge_index[:, idx_reserve]], dim=1)
            conf_edge_index = torch.cat([conf_edge_index, edge_index[:, idx_drop]], dim=1)

        return x, causal_edge_index, conf_edge_index
    
    
class Model(nn.Module):
    """
    Our proposed Disentangled Dynamic Graph Attention Networks
    this is a version for time feature
    """

    def __init__(self, args=None):
        super(Model, self).__init__()
        self.args = args
        self.num_nodes = args.num_nodes
        self.batch_size = args.batch_size

        in_dim, hid_dim = args.input_length, 2*args.nhid

        self.causal_mlp = nn.Sequential(
            nn.Linear(hid_dim, 2*hid_dim),
            nn.GELU(),
            nn.Linear(2*hid_dim, args.input_length)
        )
        
        self.conf_mlp = nn.Sequential(
            nn.Linear(hid_dim, 2*hid_dim),
            nn.GELU(),
            nn.Linear(2*hid_dim, args.input_length)
        )

        self.gnn_layer_causal = GNNLayer(hid_dim, hid_dim, inter_dim=hid_dim*2, heads=args.heads) # node level
        self.gnn_layer_conf = GNNLayer(hid_dim, hid_dim, inter_dim=hid_dim*2, heads=args.heads)   # node level

    def forward(self, x):
        """
        relabel the node for two part
        """

        casual_x = x
        conf_x = x.clone().detach() #[B N dim] 
        
        return casual_x, conf_x
    
    def get_causal_rep(self, x, causal_edge_index, all_embeddings=None):
        cs_out = self.gnn_layer_causal(x, causal_edge_index,
                                         embedding=all_embeddings)  # zi # [B*N hid_dim]
        # cs_out = self.gnn_layer(x, causal_edge_index,
        #                                 embedding=all_embeddings)  # zi # [B*N hid_dim]
        return cs_out.view(self.batch_size, self.num_nodes, -1)   # [B N hid_dim]
    
    def get_conf_rep(self, x, conf_edge_index, all_embeddings=None):
        # ss_out = self.gnn_layer(x, conf_edge_index,
        #                                 embedding=all_embeddings)  # zv # [B*N hid_dim]
        ss_out = self.gnn_layer_conf(x, conf_edge_index,
                                         embedding=all_embeddings)  # zv # [B*N hid_dim]
        return ss_out.view(self.batch_size, self.num_nodes, -1)   # [B N hid_dim]
    
    def get_causal_pred(self, x, causal_edge_index, all_embeddings=None):
        causal_graph_x = self.get_causal_rep( x, causal_edge_index, all_embeddings)
        pred = self.causal_mlp(causal_graph_x)
        return pred, causal_graph_x
    
    def get_conf_pred(self, x, conf_edge_index, all_embeddings=None):
        conf_graph_x = self.get_conf_rep( x, conf_edge_index, all_embeddings)
        pred = self.conf_mlp(conf_graph_x)
        return pred, conf_graph_x

    def get_comb_pred(self, causal_graph_x, conf_graph_x):
        # causal_graph_x = self.get_causal_rep( x, causal_edge_index, all_embeddings)
        # conf_graph_x = self.get_conf_rep( x, conf_edge_index, all_embeddings).detach() 
        causal_pred = self.causal_mlp(causal_graph_x)
        conf_pred = self.conf_mlp(conf_graph_x).detach()    
        return torch.sigmoid(conf_pred) * causal_pred
