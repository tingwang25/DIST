
import os
import traceback
import torch
from model.model import Model, AttModel
from dataset.dataloader import load_dataset, load_graph
import argparse
from train import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=1000, type=int)
parser.add_argument('--mode', default='train', type=str, help='the configuration to use')
parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--batch_size', default=256, type=int)  # metr&bay 256; pems08 256; pems07 128
parser.add_argument('--data_dir', default='data/', type=str, help='directory for datasets.')
parser.add_argument('--step_dir', default='h12/', type=str, help='directory for datasets.')
parser.add_argument('--input_length', default=12, type=int)

# dataset
parser.add_argument('--dataset', default='METR-LA', type=str)   # METR-LA, bay, PEMS08, PEMS04, PEMS07
parser.add_argument('--num_nodes', type=int, default=207, help='num of nodes')  # metr 207; bay 325; pems08 170; pems04 307; pems07 883
parser.add_argument('--alpha', default=1e-4, type=float, help='invariant loss') # metr&bay 1e-4, pems08 1e-6, pems04 1e-8; pems07 1e-8

parser.add_argument('--causal_ratio', default=0.8, type=float, help='causal_ratio r')    # metr&bay&pems04: 0.8; pems08 0.9; pems07: 0.8
parser.add_argument('--normalized_k', default=0.1, type=float, 
                    help='Entries that become lower than normalized_k after normalization are set to zero for sparsity.')    # k

# hyperparam
parser.add_argument('--lr_patience', default=20, type=int, help='learning rate adjustment') # metr&bay: 20, pems08: 10 pems04:20; pems07:20
parser.add_argument('--net_lr', default=0.005, type=float, help='learning rate for the predictor')   # metr 0.01; pems08 0.01; #pems04: 0.005 #pems07: 0.01
parser.add_argument('--intervention_mechanism',type=int,default=1, help='0 none ; 1 DIR; 2 DIR2; 3 DIR3')   
parser.add_argument('--use_distance', type=int, default=2, help='0 no distance; 1 only distance; 2 distance+dyadj') #
parser.add_argument('--var_ratio', default=1, type=float, help='alpha increase ratio for each epoch') 
parser.add_argument('--direction', type=int, default=1, help='0 indirect; 1 direct') #
parser.add_argument('--edge_score', type=int, default=1, help='0 mlp_cat; 1 mlp_hadamard;') #


parser.add_argument('--nhid', type=int, default=8, help='dim of hidden embedding') # metr&bay: 8 pems08:16 pems04:8 pems07:8
parser.add_argument('--debug', default=False, help='true means no log in file.')
parser.add_argument('--early_stop', default=True)  # None
parser.add_argument('--early_stop_patience', default=50)  # None
parser.add_argument('--n_obs', default=None, help='Only use this many observations. For unit testing.')  # None
parser.add_argument('--clip', default=5, type=int, help='gradient clip.')  # None
parser.add_argument('--weight_decay', type=float, default=5e-7, help='weight for L2 loss on basic models.')

parser.add_argument('--heads', type=int, default=1, help='attention heads.') # [1,2,3,4]

parser.add_argument('--best_path', default='result/best_model.pth', type=str)

args = parser.parse_args()

# load data

data = load_dataset(os.path.join(args.data_dir, args.dataset, args.step_dir), args.dataset, args.n_obs, args.batch_size)
ori_edge_index,_,ori_adj = load_graph(args.data_dir, args.dataset) 

# model
model = Model(args=args).to(args.device)
attModel = AttModel(args=args).to(args.device)

params = sum(p.numel() for p in model.parameters())
# print("FLOPs:", flops)
print("Params:", params) 

optimizer = torch.optim.Adam(
    list(model.parameters()) + 
    list(attModel.parameters()), 
    lr=args.net_lr,
    weight_decay=args.weight_decay
)
conf_opt = torch.optim.Adam(model.conf_mlp.parameters(), lr=args.net_lr)

lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau( 
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=args.lr_patience, 
    verbose=True, 
    threshold=0.0001, 
    threshold_mode='rel', 
    min_lr=0.000005, 
    eps=1e-08
    )

# start training

trainer = Trainer(
    model=model,
    attModel = attModel,
    optimizer=optimizer,
    conf_opt = conf_opt,
    data=data,
    ori_edge_index = ori_edge_index,
    ori_adj = torch.tensor(ori_adj).to(args.device),
    lr_scheduler=lr_scheduler,
    args=args
)

results = None
try:
    if args.mode == 'train':
        results = trainer.train() # best_eval_loss, best_epoch
    elif args.mode == 'test':
        # test
        state_dict = torch.load(
            args.best_path,
            map_location=torch.device(args.device)
        )
        model.load_state_dict(state_dict['model'])
        attModel.load_state_dict(state_dict['attModel'])
        print("Load saved model")
        results = trainer.test(model, attModel, data['test_loader'],ori_edge_index, torch.tensor(ori_adj).to(args.device), data['scaler'],data['_min'], data['_max'],
                    trainer.logger, trainer.args)
    else:
        raise ValueError
except:
    trainer.logger.info(traceback.format_exc())