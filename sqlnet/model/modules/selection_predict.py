import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from net_utils import run_lstm, col_name_encode

def debug_print(var_name, var_val):
    print var_name + ": " 
    print var_val

class SelPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, max_col_num, max_tok_num, use_ca):
        super(SelPredictor, self).__init__()
        self.use_ca = use_ca
        self.max_tok_num = max_tok_num
        self.sel_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        if use_ca:
            print "Using column attention on selection predicting"
            self.sel_att = nn.Linear(N_h, N_h)
        else:
            print "Not using column attention on selection predicting"
            self.sel_att = nn.Linear(N_h, 1)

        self.max_col_num = max_col_num
        self.sel_col_name_enc = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.sel_col_num_name_enc = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.N_h = N_h
        self.sel_num_col_att = nn.Linear(N_h, 1)
        self.sel_num_att = nn.Linear(N_h, 1)
        self.sel_col_att = nn.Linear(N_h, 1)
        self.sel_num_col2hid1 = nn.Linear(N_h, 2*N_h)
        self.sel_num_col2hid2 = nn.Linear(N_h, 2*N_h)
        self.sel_out_K = nn.Linear(N_h, N_h)
        self.sel_out_col = nn.Linear(N_h, N_h)
        self.sel_num_out = nn.Sequential(nn.Linear(N_h, N_h),
                nn.Tanh(), nn.Linear(N_h, max_col_num)) # NOTE: might have to change the third dimension
        self.sel_col_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))
        self.softmax = nn.Softmax()
        self.col_num_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2, num_layers=N_depth, batch_first=True,
        dropout=0.3, bidirectional=True)

    def forward_mult(self, x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num):
        # Predict the number of conditions
        # First use column embeddings to calculate the initial hidden unit
        # Then run the LSTM and predict condition number.
        # exit(1)
        B = len(x_len)
        max_x_len = max(x_len)
        e_num_col, col_num = col_name_encode(col_inp_var, col_name_len,
                col_len, self.sel_col_num_name_enc)
        num_col_att_val = self.sel_num_col_att(e_num_col).squeeze() 
        for idx, num in enumerate(col_num):
            if num < max(col_num):
                num_col_att_val[idx, num:] = -100
        # get a probability distribution of how many columns likely to be selected
        num_col_att = self.softmax(num_col_att_val)
        K_num_col = (e_num_col * num_col_att.unsqueeze(2)).sum(1) # not really sure what this is doing

        sel_num_h1 = self.sel_num_col2hid1(K_num_col).view(
                B, -1, self.N_h/2).transpose(0, 1).contiguous() # not really sure what the second dimension should be - previously was 4
        sel_num_h2 = self.sel_num_col2hid2(K_num_col).view(
                B, -1, self.N_h/2).transpose(0, 1).contiguous()
        h_num_enc, _ = run_lstm(self.col_num_lstm, x_emb_var, x_len,
                hidden=(sel_num_h1, sel_num_h2))
        num_att_val = self.sel_num_att(h_num_enc).squeeze()

        for idx, num in enumerate(x_len):
            if num < max_x_len:
                num_att_val[idx, num:] = -100
        num_att = self.softmax(num_att_val)

        K_sel_num = (h_num_enc * num_att.unsqueeze(2).expand_as(
            h_num_enc)).sum(1)
        sel_num_score = self.sel_num_out(K_sel_num)

        #Predict the columns of conditions
        e_sel_col, _ = col_name_encode(col_inp_var, col_name_len, col_len,
                self.sel_col_name_enc)
        h_col_enc, _ = run_lstm(self.sel_lstm, x_emb_var, x_len)
        if self.use_ca:
            h_enc, _ = run_lstm(self.sel_lstm, x_emb_var, x_len)
            sel_att_val = torch.bmm(e_sel_col, self.sel_att(h_enc).transpose(1, 2))
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    att_val[idx, num:] = -100
            att = self.softmax(att_val.view((-1, max_x_len))).view(
                    B, -1, max_x_len)

            K_sel_expand = (h_enc.unsqueeze(1) * att.unsqueeze(3)).sum(2)
        else:
            # col_att_val = self.cond_col_att(h_col_enc).squeeze()
            # for idx, num in enumerate(x_len):
            #     if num < max_x_len:
            #         col_att_val[idx, num:] = -100
            # col_att = self.softmax(col_att_val)
            # K_cond_col = (h_col_enc *
            #         col_att_val.unsqueeze(2)).sum(1).unsqueeze(1)
            sel_att_val = self.sel_col_att(h_col_enc).squeeze()
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    sel_att_val[idx, num:] = -100
            sel_att = self.softmax(sel_att_val)
            # print 'att_probabilities', sel_att
            K_sel_col = (h_col_enc * sel_att_val.unsqueeze(2)).sum(1).unsqueeze(1)
        sel_col_score = self.sel_col_out(self.sel_out_K(K_sel_col) + self.sel_out_col(e_sel_col)).squeeze()
        # print 'sel_col_score', sel_col_score
        max_col_num = max(col_num)
        for b, num in enumerate(col_num):
            if num < max_col_num:
                sel_col_score[b, num:] = -100
        sel_score = (sel_num_score, sel_col_score)
        return sel_score

    def forward(self, x_emb_var, x_len, col_inp_var,
            col_name_len, col_len, col_num):
        B = len(x_emb_var)
        max_x_len = max(x_len)

        e_col, _ = col_name_encode(col_inp_var, col_name_len,
                col_len, self.sel_col_name_enc)

        if self.use_ca:
            h_enc, _ = run_lstm(self.sel_lstm, x_emb_var, x_len)
            att_val = torch.bmm(e_col, self.sel_att(h_enc).transpose(1, 2))
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    att_val[idx, :, num:] = -100
            att = self.softmax(att_val.view((-1, max_x_len))).view(
                    B, -1, max_x_len)
            K_sel_expand = (h_enc.unsqueeze(1) * att.unsqueeze(3)).sum(2)
        else:
            h_enc, _ = run_lstm(self.sel_lstm, x_emb_var, x_len)
            att_val = self.sel_att(h_enc).squeeze()
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    att_val[idx, num:] = -100
            att = self.softmax(att_val)
            K_sel = (h_enc * att.unsqueeze(2).expand_as(h_enc)).sum(1)
            K_sel_expand=K_sel.unsqueeze(1)

        sel_score = self.sel_out( self.sel_out_K(K_sel_expand) + \
                self.sel_out_col(e_col) ).squeeze() 
        debug_print('sel_score.size()',sel_score.size() )
        max_col_num = max(col_num)
        for idx, num in enumerate(col_num):
            if num < max_col_num:
                sel_score[idx, num:] = -100

        return sel_score
