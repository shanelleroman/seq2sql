import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from net_utils import run_lstm, col_name_encode


class AggPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, max_col_num, use_ca):
        super(AggPredictor, self).__init__()
        self.use_ca = use_ca
        self.N_h = N_h
        # choose how many ag ops
        self.agg_n_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.agg_num_col_att = nn.Linear(N_h, 1)
        # choose which aggs for a given number
        self.agg_item_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.agg_num_col2hid1 = nn.Linear(N_h, 2*N_h)
        self.agg_num_col2hid2 = nn.Linear(N_h, 2*N_h)
        self.agg_num_att = nn.Linear(N_h, 1)
        self.sel_num_out = nn.Sequential(nn.Linear(N_h, N_h),
                nn.Tanh(), nn.Linear(N_h, max_col_num)) # NOTE: might have to change the third dimension
        if use_ca:
            print "Using column attention on aggregator predicting"
            self.agg_col_name_enc = nn.LSTM(input_size=N_word,
                    hidden_size=N_h/2, num_layers=N_depth,
                    batch_first=True, dropout=0.3, bidirectional=True)
            self.agg_att = nn.Linear(N_h, N_h)
        else:
            print "Not using column attention on aggregator predicting"
            self.agg_att = nn.Linear(N_h, 1)
        self.agg_out = nn.Sequential(nn.Linear(N_h, N_h),
                nn.Tanh(), nn.Linear(N_h, 6)) # there are 6 types of aggregators 
        #- there will be more - get james to enumerate them
        self.softmax = nn.Softmax()

    def forward(self, x_emb_var, x_len, col_inp_var=None, col_name_len=None,
            col_len=None, col_num=None, gt_sel=None):
        B = len(x_emb_var)
        max_x_len = max(x_len)
        # # deicde how many aggregators to use
        e_num_agg, agg_num = col_name_encode(col_inp_var, col_name_len,
                col_len, self.agg_n_lstm)
        num_agg_att_val = self.agg_num_col_att(e_num_agg).squeeze() 
        for idx, num in enumerate(col_num):
            if num < max(col_num):
                num_agg_att_val[idx, num:] = -100
        # get a probability distribution of how many columns likely to be selected
        num_agg_att = self.softmax(num_agg_att_val)
        K_num_agg = (e_num_agg * num_agg_att.unsqueeze(2)).sum(1) # not really sure what this is doing

        agg_num_h1 = self.agg_num_col2hid1(K_num_agg).view(
                B, -1, self.N_h/2).transpose(0, 1).contiguous() # not really sure what the second dimension should be - previously was 4
        agg_num_h2 = self.agg_num_col2hid2(K_num_agg).view(
                B, -1, self.N_h/2).transpose(0, 1).contiguous()
        h_num_enc, _ = run_lstm(self.agg_n_lstm, x_emb_var, x_len,
                hidden=(agg_num_h1, agg_num_h2))
        num_att_val = self.agg_num_att(h_num_enc).squeeze()

        for idx, num in enumerate(x_len):
            if num < max_x_len:
                num_att_val[idx, num:] = -100
        num_att = self.softmax(num_att_val)

        K_agg_num = (h_num_enc * num_att.unsqueeze(2).expand_as(
            h_num_enc)).sum(1)
        agg_num_score = self.sel_num_out(K_agg_num)

        # decide which aggregators to use
        h_enc, _ = run_lstm(self.agg_item_lstm, x_emb_var, x_len)
        if self.use_ca:
            e_col, _ = col_name_encode(col_inp_var, col_name_len, 
                    col_len, self.agg_col_name_enc)
            chosen_sel_idx = torch.LongTensor(gt_sel)
            aux_range = torch.LongTensor(range(len(gt_sel)))
            if x_emb_var.is_cuda:
                chosen_sel_idx = chosen_sel_idx.cuda()
                aux_range = aux_range.cuda()
            chosen_e_col = e_col[aux_range, chosen_sel_idx]
            att_val = torch.bmm(self.agg_att(h_enc), 
                    chosen_e_col.unsqueeze(2)).squeeze()
        else:
            att_val = self.agg_att(h_enc).squeeze()

        for idx, num in enumerate(x_len):
            if num < max_x_len:
                att_val[idx, num:] = -100
        att = self.softmax(att_val)

        K_agg = (h_enc * att.unsqueeze(2).expand_as(h_enc)).sum(1)
        agg_item_score = self.agg_out(K_agg)
        return (agg_num_score, agg_item_score)
