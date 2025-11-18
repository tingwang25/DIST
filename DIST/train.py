import os
import time
import torch

import numpy as np

from utils.logger import get_logger, PD_Stats, get_log_dir
from utils.metrics import calc_metrics
from sklearn.metrics import mean_absolute_error

class Trainer(object):

    def __init__(self, data, ori_edge_index, model, attModel, optimizer, lr_scheduler, args, conf_opt = None, ori_adj = None):
        self.train_loader = data['train_loader']
        self.val_loader = data['val_loader']
        self.test_loader = data['test_loader']
        self.scaler = data['scaler']
        self._min = data['_min']
        self._max = data['_max']
        self.ori_edge_index = ori_edge_index
        self.ori_adj = ori_adj
        
        self.model = model
        self.attModel = attModel
        self.args = args
        self.optimizer = optimizer
        self.conf_opt = conf_opt
        self.lr_scheduler = lr_scheduler

        self.clip = self.args.clip

        # log
        args.log_dir = get_log_dir(args)
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.log_dir, debug=args.debug)
        self.best_path = os.path.join(args.log_dir, 'best_model.pth')

        # create a panda object to log loss and acc
        training_stats = PD_Stats(
            os.path.join(args.log_dir, 'stats.pkl'),
            ['epoch', 'train_loss', 'val_loss'],
        )
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.logger.info('Experiment configs are: {}'.format(args))

    def train(self):

        best_loss = float('inf')
        best_epoch = 0
        not_improved_count = 0
        start_time = time.time()

        for epoch in range(1, self.args.epochs + 1):

            all_loss = self.train_epoch(epoch)
            
            if all_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            # use val dataset to judge
            val_all_loss = self.val_epoch(epoch)

            if val_all_loss < best_loss:
                best_loss = val_all_loss
                best_epoch = epoch
                not_improved_count = 0
                # save the best state
                save_dict = {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "attModel": self.attModel.state_dict(),
                }
                if not self.args.debug:
                    self.logger.info('**************Current best model saved to {}'.format(self.best_path))
                    torch.save(save_dict, self.best_path)
            else:
                not_improved_count += 1
            
            

            self.lr_scheduler.step(val_all_loss)  

            #early stopping
            if self.args.early_stop and not_improved_count == self.args.early_stop_patience:
                self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                break

        training_time = time.time() - start_time
        self.logger.info("== Training finished.\n"
                            "Total training time: {:.2f} min\t"
                            "best loss: {:.4f}\t"
                            "best epoch: {}\t".format(
            (training_time / 60),
            best_loss,
            best_epoch))         
        
        # test
        state_dict = save_dict if self.args.debug else torch.load(
            self.best_path, map_location=torch.device(self.args.device))
        self.model.load_state_dict(state_dict['model'])
        self.attModel.load_state_dict(state_dict['attModel'])
        self.logger.info("== Test results.")
        test_results = self.test(self.model, self.attModel, 
                                 self.test_loader, self.ori_edge_index, 
                                 self.ori_adj, self.scaler,
                                 self._min, self._max,
                                 self.logger, self.args)
        results = {
            'best_val_loss': best_loss,
            'best_val_epoch': best_epoch,
            'test_results': test_results,
        }

        return results
    
    def train_epoch(self, epoch):

        alpha_prime = self.args.alpha * (epoch ** self.args.var_ratio) 
        all_loss, n_bw, all_env_loss = 0, 0, 0
        all_causal_loss, all_conf_loss = 0, 0

        self.model.train()

        self.train_loader.shuffle()

        start_time = time.time()

        for iter, (x, labels) in enumerate(self.train_loader.get_iterator()):
            n_bw += 1

            x = torch.Tensor(x).to(self.args.device)  # [B L N C]   #x graph-level
            labels = torch.Tensor(labels).squeeze().transpose(1,2).to(self.args.device)  # [B N L]   #y

            x_embedding, causal_edge_index, conf_edge_index = self.attModel(x, self.ori_adj)  # split graph # zi[B,N,dim=2nhid], zv[B,N,dim=2nhid]
            causal_x, conf_x = self.model(x_embedding) # [B,N,dim=2nhid] split nodes

            causal_out, casual_embedding = self.model.get_causal_pred(causal_x, causal_edge_index)  # Yc [B N L]
            conf_out, spurious_embedding = self.model.get_conf_pred(conf_x, conf_edge_index)  # Ys [B N L]

            if self._min == None:
                predicted_causal = self.scaler.inverse_transform(causal_out)    #f(zi)
                predicted_conf = self.scaler.inverse_transform(conf_out)    #f(zv)

                causal_loss,_,_ = calc_metrics(predicted_causal, labels, null_val=0.0)   #loss term 1
                conf_loss,_,_ = calc_metrics(predicted_conf, labels, null_val=0.0)
            else:   
                predicted_causal = self.scaler(causal_out.unsqueeze(-1).transpose(2, 3), self._max[0, 0, 0, 0], self._min[0, 0, 0, 0]).transpose(2, 3).squeeze(-1)
                predicted_conf = self.scaler(conf_out.unsqueeze(-1).transpose(2, 3), self._max[0, 0, 0, 0], self._min[0, 0, 0, 0]).transpose(2, 3).squeeze(-1)

                causal_loss,_,_ = calc_metrics(predicted_causal, labels, null_val=np.nan)   #loss term 1
                conf_loss,_,_ = calc_metrics(predicted_conf, labels, null_val=np.nan)


            env_loss = 0
            
            if self.args.intervention_mechanism == 1:   
                env = torch.tensor([]).to(self.args.device)
                for conf in spurious_embedding: # conf.shape [N, dim] 

                    rep_out = self.model.get_comb_pred(casual_embedding, conf.repeat(self.args.batch_size, 1, 1)) # y = sigmoid（ys~）*yc~ [B N L]
                    if self._min == None:
                        predicted_rep = self.scaler.inverse_transform(rep_out)    #f(zi,zv|do(s))
                        rep_loss,_,_ = calc_metrics(predicted_rep, labels, null_val=0.0)
                    else:   
                        predicted_rep = self.scaler(rep_out.unsqueeze(-1).transpose(2, 3), self._max[0, 0, 0, 0], self._min[0, 0, 0, 0]).transpose(2, 3).squeeze(-1)
                        rep_loss,_,_ = calc_metrics(predicted_rep, labels, null_val=np.nan)
                    env = torch.cat([env, rep_loss.unsqueeze(0)]) 
                env_mean = min(alpha_prime, 1) * env.mean()

                env_var = alpha_prime * torch.var(env)
                env_loss = env_mean + env_var


            self.optimizer.zero_grad()
            loss = causal_loss + env_loss   
            assert not torch.isnan(loss)
            loss.backward()
            if self.clip is not None:   
                torch.nn.utils.clip_grad_norm_(list(self.model.parameters())+list(self.attModel.parameters()), self.clip)
            
            self.conf_opt.zero_grad()
            conf_loss.backward()
            self.optimizer.step()
            self.conf_opt.step()

            all_conf_loss += conf_loss.item()
            all_causal_loss += causal_loss.item() # term 1
            all_env_loss += env_loss.item()    # term 2

        # each epoch
        endtime=time.time()
        print("train time: " + str(endtime-start_time))
        all_env_loss /= n_bw
        all_causal_loss /= n_bw
        all_conf_loss /= n_bw
        all_loss = all_causal_loss + all_env_loss

        self.logger.info('*******Train Epoch {}: all_loss:{:2.3f}=[Term1:{:2.3f}  Term2:{:2.6f}]'.format(epoch, all_loss, all_causal_loss, all_env_loss))

        return all_loss
    
    def val_epoch(self, epoch):
        alpha_prime = self.args.alpha * (epoch ** self.args.var_ratio) 
        all_loss, n_bw, all_env_loss = 0, 0, 0
        all_causal_loss, all_conf_loss = 0, 0

        self.model.eval()

        with torch.no_grad():
            for iter, (x, labels) in enumerate(self.val_loader.get_iterator()):
                n_bw += 1

                x = torch.Tensor(x).to(self.args.device)  # [B L N C]   #x graph-level
                labels = torch.Tensor(labels).squeeze().transpose(1,2).to(self.args.device)  # [B N L]   #y

                x_embedding, causal_edge_index, conf_edge_index = self.attModel(x, self.ori_adj)  # split graph # zi[B,N,dim=2nhid], zv[B,N,dim=2nhid]
                causal_x, conf_x = self.model(x_embedding) # [B,N,dim=2nhid] split nodes

                causal_out, casual_embedding = self.model.get_causal_pred(causal_x, causal_edge_index)  # Yc [B N L]
                conf_out, spurious_embedding = self.model.get_conf_pred(conf_x, conf_edge_index)  # Ys [B N L]

                if self._min == None:
                    predicted_causal = self.scaler.inverse_transform(causal_out)    #f(zi)
                    predicted_conf = self.scaler.inverse_transform(conf_out)    #f(zv)
                else:  
                    predicted_causal = self.scaler(causal_out.unsqueeze(-1).transpose(2, 3), self._max[0, 0, 0, 0], self._min[0, 0, 0, 0]).transpose(2, 3).squeeze(-1)
                    predicted_conf = self.scaler(conf_out.unsqueeze(-1).transpose(2, 3), self._max[0, 0, 0, 0], self._min[0, 0, 0, 0]).transpose(2, 3).squeeze(-1)

                causal_loss,_,_ = calc_metrics(predicted_causal, labels, null_val=0.0)   #loss term 1
                conf_loss,_,_ = calc_metrics(predicted_conf, labels, null_val=0.0)

                env_loss = 0
            
                if self.args.intervention_mechanism == 1:
                    env = torch.tensor([]).to(self.args.device)
                    for conf in spurious_embedding:
                        rep_out = self.model.get_comb_pred(casual_embedding, conf.repeat(self.args.batch_size, 1, 1)) # y = sigmoid（ys~）*yc~ [B N L]

                        if self._min == None:
                            predicted_rep = self.scaler.inverse_transform(rep_out)    #f(zi,zv|do(s))
                        else:   
                            predicted_rep = self.scaler(rep_out.unsqueeze(-1).transpose(2, 3), self._max[0, 0, 0, 0], self._min[0, 0, 0, 0]).transpose(2, 3).squeeze(-1)

                        rep_loss,_,_ = calc_metrics(predicted_rep, labels, null_val=0.0)
                        env = torch.cat([env, rep_loss.unsqueeze(0)])
                    env_mean = min(alpha_prime, 1) * env.mean()
                    env_var = alpha_prime * torch.var(env) 
                    env_loss = env_mean + env_var
                
                all_causal_loss += causal_loss.item() # term 1
                all_env_loss += env_loss.item()    # term 2

            # each epoch
            all_env_loss /= n_bw
            all_causal_loss /= n_bw
            all_loss = all_causal_loss + all_env_loss

            self.logger.info('*******Val Epoch {}: all_loss:{:2.3f}=[Term1:{:2.3f}  Term2:{:2.6f}]'.format(epoch, all_loss, all_causal_loss, all_env_loss))

            return all_loss
        
    @staticmethod
    def test(model,attModel, dataloader,ori_edge_index, ori_adj, scaler, min, max, logger, args):
        all_causal_loss,all_causal_mape, all_causal_rmse = [0.],[0.], [0.]
        all_conf_loss,all_conf_mape, all_conf_rmse = [0.],[0.], [0.]
        n_bw = 0

        start_time = time.time()

        model.eval()
        with torch.no_grad():
            for iter, (x, labels) in enumerate(dataloader.get_iterator()):
                x = torch.Tensor(x).to(args.device)  # [B L N C]   #x graph-level
                labels = torch.Tensor(labels).squeeze().transpose(1,2).to(args.device)  # [B N L]   #y

                x_embedding, causal_edge_index, conf_edge_index = attModel(x, ori_adj)  # split graph # zi[B,N,dim=2nhid], zv[B,N,dim=2nhid]
                causal_x, conf_x = model(x_embedding) # [B,N,dim=2nhid] split nodes

                causal_out, casual_embedding = model.get_causal_pred(causal_x, causal_edge_index)  # Yc [B N L]
                conf_out, spurious_embedding = model.get_conf_pred(conf_x, conf_edge_index)  # Ys [B N L]

                if min == None:
                    predicted_causal = scaler.inverse_transform(causal_out)    #f(zi)
                    predicted_conf = scaler.inverse_transform(conf_out)    #f(zv)

                else:   
                    predicted_causal = scaler(causal_out.unsqueeze(-1).transpose(2, 3), max[0, 0, 0, 0], min[0, 0, 0, 0]).transpose(2, 3).squeeze(-1)
                    predicted_conf = scaler(conf_out.unsqueeze(-1).transpose(2, 3), max[0, 0, 0, 0], min[0, 0, 0, 0]).transpose(2, 3).squeeze(-1)

                causal_loss,causal_mape, causal_rmse = calc_metrics(predicted_causal, labels, null_val=0.0)   #loss term 1
                conf_loss,conf_mape, conf_rmse = calc_metrics(predicted_conf, labels, null_val=0.0)

                all_causal_loss.append(causal_loss)  # term 1
                all_conf_loss.append(conf_loss)    # term 2
                all_causal_mape.append(causal_mape)
                all_conf_mape.append(conf_mape)
                all_conf_rmse.append(conf_rmse)
                all_causal_rmse.append(causal_rmse)

        endtime=time.time()
        print("inference time: " + str(endtime-start_time))

        test_results = []
        test_results.append([torch.tensor(all_causal_loss).mean(), torch.tensor(all_causal_mape).mean(), torch.tensor(all_causal_rmse).mean()])
        logger.info("test_result, mae: [{:.3f}, {:.3f}], mape: [{:.4f}, {:.4f}], rmse: [{:.3f}, {:.3f}]".format(
                    torch.tensor(all_causal_loss).mean(), torch.tensor(all_conf_loss).mean() , 
                    torch.tensor(all_causal_mape).mean(),torch.tensor(all_conf_mape).mean(), 
                    torch.tensor(all_causal_rmse).mean(),torch.tensor(all_conf_rmse).mean() ))
        
        return test_results