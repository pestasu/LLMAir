import os
import time
import warnings
import ipdb
import numpy as np
import pandas as pd
import logging
import torch.distributed as dist
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt

from exp.exp_basic import Exp_Basic
from utils import metrics, graph
from utils.tools import adjust_learning_rate
from utils.data_loader import StandardScaler, get_dataloader
from models import *

warnings.filterwarnings('ignore')
my_logger = 'lazy'
logger = logging.getLogger(my_logger)

class Exp_llmair(Exp_Basic):
    def __init__(self, args, ii):
        super(Exp_llmair, self).__init__(args)
        self.cur_exp = ii
        self.alpha_alg = args.alpha_alg
        self.alpha_rec = args.alpha_rec
        self.train_loader, self.valid_loader, self.test_loader = self.dataloader['train'], self.dataloader['valid'], self.dataloader['test']

    def _save_model(self, save_path, cur_exp):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = f'final_model_{cur_exp}.pt'
        state_dict = self.model.state_dict()
        llm_params = 'backbone_llm'
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith(llm_params)}
        if self.args.prefix_alignment:
            word_params = 'word_embeddings'
            filtered_state_dict = {k: v for k, v in filtered_state_dict.items() if not k.startswith(word_params)}
        torch.save(filtered_state_dict, os.path.join(save_path, filename))
        return filename

    def _load_model(self, save_path, cur_exp):
        filename = f'final_model_{cur_exp}.pt'
        state_dict = torch.load(os.path.join(save_path, filename))
        
        model_state_dict = self.model.state_dict()
        llm_params = 'backbone_llm'
        word_params = 'word_embeddings'

        # Filter out the keys that should not be overwritten in the LLM
        filtered_state_dict = {k: v for k, v in state_dict.items() if k not in model_state_dict or (llm_params not in k and (not self.args.prefix_alignment or word_params not in k))}
        
        model_state_dict.update(filtered_state_dict)
        self.model.load_state_dict(model_state_dict)
        
        return filename

    def early_stop(self, epoch, best_loss):
        logger.info(f'Early stop at epoch {epoch}, loss = {best_loss:.6f}')
        # np.savetxt(os.path.join(self.pt_dir, f'val_loss_{self.cur_exp}.txt'), [best_loss], fmt='%.4f', delimiter=',')

    def train_batch(self, x, y):
        '''
        the training process of a batch
        '''   
        self.optimizer.zero_grad()

        x = x.to(self.device)
        y = y[..., :1].to(self.device)

        output = self.model(x)
        pred, true = self._inverse_transform([output['pred'], y])
        loss_pred = self.loss_fn(pred, true) 
        loss_rec = self.loss_fn(output['rec'], x[...,:1]) 
        loss_align = output['align']

        loss = loss_pred + self.alpha_alg*loss_align + self.alpha_rec*loss_rec
        # loss = loss_pred 

        loss.backward()
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         print(f'{name} grad: {param.grad.norm().item()}')
                
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                    max_norm=self.max_grad_norm)
        self.optimizer.step()

        return loss.item(), loss_pred.item(), loss_align.item(), loss_rec.item()

    def train(self, setting):
        train_steps = len(self.train_loader)

        self.saved_epoch = -1
        self.val_losses = [np.inf]
        time_now = time.time()
        for epoch in range(self.train_epochs):
            self.model.train()

            iter_count = 0
            train_losses, pred_losses, align_losses, rec_losses = [], [], [], []
            # train_losses = []
            if epoch - self.saved_epoch > self.patience:
                self.early_stop(epoch, min(self.val_losses))
                np.savetxt(os.path.join(self.pt_dir, f'val_loss_{self.cur_exp}.txt'), self.val_losses, fmt='%.4f', delimiter=',')
                break

            logger.info('------start training!------')
            start_time = time.time()
            for i, (batch_x, batch_y) in enumerate(self.train_loader):
                iter_count += 1
                loss, pred_loss, align_loss, rec_loss = self.train_batch(batch_x, batch_y)

                train_losses.append(loss)
                pred_losses.append(pred_loss)
                align_losses.append(align_loss)
                rec_losses.append(rec_loss)

                if (i + 1) % 100 == 0:
                    logger.info(f'\titers: {i+1}, epoch: {epoch+1} | loss: {loss:.7f}')
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.train_epochs - epoch) * train_steps - i)
                    logger.info(f'\tspeed: {speed:.4f}s/iter | left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                if iter_count % self.save_iter == 0:
                    val_loss, _ = self.valid(epoch)
                    # logger.info(f'Epoch [{epoch}/{self.train_epochs}]({iter_count}) | val_mae:{val_loss:.4f} | train_loss:{np.mean(train_losses):.4f}')
                    logger.info(f'Epoch [{epoch}/{self.train_epochs}]({iter_count}) | val_mae:{val_loss:.4f} | train_loss:{loss:.4f}, train_mae:{pred_loss:.4f}, train_align:{align_loss:.4f}, train_rec:{rec_loss:.4f}')
                    # logger.info(f'Epoch [{epoch}/{self.train_epochs}]({iter_count}) | val_mae:{val_loss:.4f} | train_loss:{np.mean(train_losses):.4f}, train_mae:{np.mean(pred_losses):.4f}, train_rec:{np.mean(rec_losses):.4f}, train_align:{np.mean(align_losses):.4f}')
            
            end_time = time.time()
            logger.info(f'{epoch}-epoch complete')
            logger.info('------evaluating now!------')

            val_loss, val_time = self.valid(epoch)
            # logger.info(f'Epoch [{epoch}/{self.train_epochs}]({iter_count}) | val_mae:{val_loss:.4f}, val_time:{val_time:.1f}s | train_mae:{np.mean(train_losses):.4f}')
            # logger.info(f'Epoch [{epoch}/{self.train_epochs}]({iter_count}) | val_mae:{val_loss:.4f} | train_loss:{np.mean(train_losses):.4f}, train_mae:{np.mean(pred_losses):.4f}, train_align:{np.mean(align_losses):.4f}')
            logger.info(f'Epoch [{epoch}/{self.train_epochs}]({iter_count}) | val_mae:{val_loss:.4f} | train_loss:{np.mean(train_losses):.4f}, train_mae:{np.mean(pred_losses):.4f}, train_rec:{np.mean(rec_losses):.4f}, train_align:{np.mean(align_losses):.4f}')
            if self.lr_adj == 'cosine':
                scheduler.step()
                print(f'lr = {self.optimizer.param_groups[0]["lr"]:.10f}')
            else:
                adjust_learning_rate(self.optimizer, epoch+1, self.args)

    def valid(self, epoch):
        preds, trues = [], []
        self.model.eval()
        total_time = 0
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.valid_loader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y[..., :1].to(self.device)

                time_now = time.time()

                output = self.model(batch_x)
                pred, true = self._inverse_transform([output['pred'], batch_y])
                preds.append(pred.cpu())
                trues.append(true.cpu())

                total_time += time.time() - time_now

        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        val_loss = self.loss_fn(preds, trues)

        if val_loss < np.min(self.val_losses):
            saved_model_file = self._save_model(self.pt_dir, self.cur_exp)
            logger.info(f'Valid loss decrease: {np.min(self.val_losses)} -> {val_loss}, saving to {saved_model_file}')
            self.val_losses.append(val_loss)
            self.saved_epoch = epoch

            # test in each epoch
            self.test()

        self.model.train()

        return val_loss, total_time

    def test(self, is_test=False):
        if is_test:
            logger.info(f'------------Test process~load model({self.args.model}-{self.args.version})------------')
            self._load_model(self.pt_dir, self.cur_exp)

        preds, trues = [], []
        preds2, trues2 = [], []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.test_loader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y[..., :1].to(self.device)

                output = self.model(batch_x)
                pred, true = self._inverse_transform([output['pred'], batch_y])
                preds.append(pred.cpu())
                trues.append(true.cpu())
                
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)

        maes = []
        rmses = []
        mae, rmse = metrics.compute_all_metrics(preds, trues)
        logger.info(f'***** Average Horizon, Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f} *****')
        maes.append(mae)
        rmses.append(rmse)
        if self.horizon == 24: 
            for i in range(0, self.horizon, 8):
                pred = preds[:,i: i + 8]
                true = trues[:,i: i + 8]
                result = metrics.compute_all_metrics(pred, true)
                maes.append(result[0])
                rmses.append(result[1])

            logger.info(f'***** (0-7) 1-8h Test MAE: {maes[1]:.4f}, Test RMSE: {rmses[1]:.4f} *****')
            logger.info(f'***** (8-15) 9-16h Test MAE: {maes[2]:.4f}, Test RMSE: {rmses[2]:.4f} *****')
            logger.info(f'***** (16-23) 17-24h Test MAE: {maes[3]:.4f}, Test RMSE: {rmses[3]:.4f} *****')

            results = pd.DataFrame(columns=['Time','Test MAE', 'Test RMSE'], index=range(5))
            Time_list=['Average','1-8h','9-16h','17-24h', 'SuddenChange']

            results = pd.DataFrame(columns=['Time','Test MAE', 'Test RMSE'], index=range(5))
            Time_list=['Average','1-40min','41-80min','81-120min', 'SuddenChange']

            for i in range(4):
                results.iloc[i, 0]= Time_list[i]
                results.iloc[i, 1]= maes[i]
                results.iloc[i, 2]= rmses[i]
        
        else:
            print('The output length is not 24 !!!')

        mask_sudden_change = metrics.sudden_changes_mask(trues, datapath=self.args.data_root_path, null_val=0.0, threshold_start=75, threshold_change=20, horizon=self.horizon)
        results.iloc[4, 0] = Time_list[4]
        mae_sc, rmse_sc = metrics.compute_sudden_change(mask_sudden_change, preds, trues, null_value=0.0)
        results.iloc[4, 1:] = [mae_sc, rmse_sc]
        logger.info(f'***** Sudden Changes MAE: {mae_sc:.4f}, Test RMSE: {rmse_sc:.4f} *****')
    
        results.to_csv(os.path.join(self.pt_dir, f'metrics_{self.cur_exp}.csv'), index = False)

        return results