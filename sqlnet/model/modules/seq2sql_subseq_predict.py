# *- coding: utf-8 -*-
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from net_utils import run_lstm
import logging

class Seq2SQLSubSeqPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, max_col_num, max_tok_num, gpu, start_end_indices=(0,1)):
        super(Seq2SQLSubSeqPredictor, self).__init__()
        logging.info("Seq2SQL subsequence prediction")
        self.N_h = N_h
        self.max_tok_num = 400
        self.max_col_num = max_col_num
        self.gpu = gpu

        self.enc_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.dec_lstm = nn.LSTM(input_size=self.max_tok_num,
                hidden_size=N_h, num_layers=N_depth,
                batch_first=True, dropout=0.3)

        self.seq_out_g = nn.Linear(N_h, N_h) # gold standard
        self.seq_out_h = nn.Linear(N_h, N_h) # for the hidden states
        self.seq_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1)) # just one SEQUENCE that we are predicting - each word is one CHOICE

        self.softmax = nn.Softmax() # get the probability distribution
        self.start_end_indices = start_end_indices


    def gen_gt_batch(self, tok_seq, gen_inp=True):
        # If gen_inp: generate the input token sequence (removing <END>)
        # Otherwise: generate the output token sequence (removing <BEG>)
        # print ('tok_seq', tok_seq)
        logging.info('method subsequence_pred gen_gt_batch')
        B = len(tok_seq)
        ret_len = np.array([len(one_tok_seq)-1 for one_tok_seq in tok_seq])
        max_len = max(ret_len)
        ret_array = np.zeros((B, max_len, self.max_tok_num), dtype=np.float32)
        for b, one_tok_seq in enumerate(tok_seq):
            # print('one_tok_seq', one_tok_seq)
            out_one_tok_seq = one_tok_seq[:-1] if gen_inp else one_tok_seq[1:]
            logging.warning('generated_gt_sel_decoder_seq {0}'.format(out_one_tok_seq))
            # print ('out_one_tok_seq', out_one_tok_seq)
            for t, tok_id in enumerate(out_one_tok_seq):
                ret_array[b, t, tok_id] = 1
        ret_inp = torch.from_numpy(ret_array)
        if self.gpu:
            ret_inp = ret_inp.cuda()
        ret_inp_var = Variable(ret_inp) #[B, max_len, max_tok_num]
        # print('ret_inp_var', ret_inp_var)
        # print('ret_inp_var.size()', ret_inp_var.size())
        return ret_inp_var, ret_len

    def forward(self, x_emb_var, x_len, col_inp_var, col_name_len, col_len,
            col_num, gt_index_seq=None, gt_query=None, reinforce=None): # check what is gt_query
        logging.info('method subsequence_pred forward')
        max_x_len = max(x_len)
        # print ('max_x_len', max_x_len)
        B = len(x_len)

        h_enc, hidden = run_lstm(self.enc_lstm, x_emb_var, x_len)
        # print('h_enc.size()', h_enc.size())
        
        decoder_hidden = tuple(torch.cat((hid[:2], hid[2:]),dim=2) 
                for hid in hidden) 
        # print('decoder_hidden[0].size()', decoder_hidden[0].size())
        # print('decoder_hidden[1].size()', decoder_hidden[1].size())
        if gt_index_seq is not None:
            logging.info('gold sequence provided')
            logging.info('gt_sel_gt_index_seq {0}'.format(gt_index_seq))
            gt_tok_seq, gt_tok_len = self.gen_gt_batch(gt_index_seq, gen_inp=True) # get rid of <SELECT>
            # print('gt_index_seq', gt_tok_seq)
            # gt_tok_seq: SELECT_index agg_index , col_index ... last_col_index
            # print('gt_tok_seq.size()', gt_tok_seq.size())
            # print ('gt_tok_seq', gt_tok_seq)
            # print('gt_tok_len', gt_tok_len)
            # print('decoder_hidden', decoder_hidden)
            g_s, _ = run_lstm(self.dec_lstm, gt_tok_seq, gt_tok_len, decoder_hidden)
            # logging.info('pred_sel_decoder_seq {0}'.format(g_s))
            # print('g_s.size()', g_s.size())

            h_enc_expand = h_enc.unsqueeze(1)
            g_s_expand = g_s.unsqueeze(2)

            seq_score = self.seq_out( self.seq_out_h(h_enc_expand) +
                    self.seq_out_g(g_s_expand) ).squeeze()
            # print('seq_score.size()', seq_score.size())
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    seq_score[idx, :, num:] = -100 # make sure the padded numbers have ~0 probab
        else:
            logging.info('gold sequence not provided')
            h_enc_expand = h_enc.unsqueeze(1)
            scores = []
            choices = []
            done_set = set()

            t = 0
            init_inp = np.zeros((B, 1, self.max_tok_num), dtype=np.float32) # initialize input because no golden to pass in
            # print ('self.max_tok_num', self.max_tok_num)
            init_inp[:,0,self.start_end_indices[0]] = 1   #Set the SELECT token - this needs to change - will need to pass in as a parameter when I extend for other things
            # 2 = index of SELECT
            # first input
            #RNN <=> update cur_inp each time, pass hidden and update it! 
            if self.gpu:
                cur_inp = Variable(torch.from_numpy(init_inp).cuda())
            else:
                cur_inp = Variable(torch.from_numpy(init_inp)) 
            cur_h = decoder_hidden
            while len(done_set) < B and t < 100:
                g_s, cur_h = self.dec_lstm(cur_inp, cur_h)
                # print('g_s.size()', g_s.size())
                g_s_expand = g_s.unsqueeze(2)

                cur_seq_score = self.seq_out(self.seq_out_h(h_enc_expand) +
                        self.seq_out_g(g_s_expand)).squeeze()
                # print('cur_seq_score.size()', cur_seq_score.size())
                for b, num in enumerate(x_len):
                    if num < max_x_len:
                        cur_seq_score[b, num:] = -100
                scores.append(cur_seq_score) # score for this particular word

                if not reinforce:
                    _, ans_tok_var = cur_seq_score.view(B, max_x_len).max(1)
                    ans_tok_var = ans_tok_var.unsqueeze(1)
                else:
                    ans_tok_var = self.softmax(cur_seq_score).multinomial() # get the probability distribution for the scores
                    choices.append(ans_tok_var) # list of the different probability distributions
                ans_tok = ans_tok_var.data.cpu()
                if self.gpu:  #To one-hot
                    cur_inp = Variable(torch.zeros(
                        B, self.max_tok_num).scatter_(1, ans_tok, 1).cuda())
                else:
                    cur_inp = Variable(torch.zeros(
                        B, self.max_tok_num).scatter_(1, ans_tok, 1))
                cur_inp = cur_inp.unsqueeze(1)
                # print('ans_tok.squeeze()', ans_tok.squeeze())
                for idx, tok in enumerate(ans_tok.squeeze()):
                    if tok == self.start_end_indices[1]:  #Find the <END> token
                        done_set.add(idx) # index of the the <END> token for this sequence => this much closer to finishing!
                t += 1

            seq_score = torch.stack(scores, 1)

        if reinforce:
            return seq_score, choices
        else:
            return seq_score


    # def forward(self, x_emb_var, x_len, col_inp_var, col_name_len, col_len,
    #         col_num, gt_where, gt_query, reinforce):
    #     max_x_len = max(x_len)
    #     B = len(x_len)

    #     h_enc, hidden = run_lstm(self.enc_lstm, x_emb_var, x_len)
    #     decoder_hidden = tuple(torch.cat((hid[:2], hid[2:]),dim=2) 
    #             for hid in hidden)
    #     if gt_where is not None:
    #         gt_tok_seq, gt_tok_len = self.gen_gt_batch(gt_where, gen_inp=True)
    #         g_s, _ = run_lstm(self.cond_decoder,
    #                 gt_tok_seq, gt_tok_len, decoder_hidden)

    #         h_enc_expand = h_enc.unsqueeze(1)
    #         g_s_expand = g_s.unsqueeze(2)
    #         #
    #         cond_score = self.cond_out( self.cond_out_h(h_enc_expand) +
    #                 self.cond_out_g(g_s_expand) ).squeeze()
    #         for idx, num in enumerate(x_len):
    #             if num < max_x_len:
    #                 cond_score[idx, :, num:] = -100
    #     else:
    #         h_enc_expand = h_enc.unsqueeze(1)
    #         scores = []
    #         choices = []
    #         done_set = set()

    #         t = 0
    #         init_inp = np.zeros((B, 1, self.max_tok_num), dtype=np.float32)
    #         init_inp[:,0,7] = 1   #Set the <BEG> token - this needs to change
    #         if self.gpu:
    #             cur_inp = Variable(torch.from_numpy(init_inp).cuda())
    #         else:
    #             cur_inp = Variable(torch.from_numpy(init_inp))
    #         cur_h = decoder_hidden
    #         while len(done_set) < B and t < 100:
    #             g_s, cur_h = self.cond_decoder(cur_inp, cur_h)
    #             g_s_expand = g_s.unsqueeze(2)

    #             cur_cond_score = self.cond_out(self.cond_out_h(h_enc_expand) +
    #                     self.cond_out_g(g_s_expand)).squeeze()
    #             for b, num in enumerate(x_len):
    #                 if num < max_x_len:
    #                     cur_cond_score[b, num:] = -100
    #             scores.append(cur_cond_score)

    #             if not reinforce:
    #                 _, ans_tok_var = cur_cond_score.view(B, max_x_len).max(1)
    #                 ans_tok_var = ans_tok_var.unsqueeze(1)
    #             else:
    #                 ans_tok_var = self.softmax(cur_cond_score).multinomial()
    #                 choices.append(ans_tok_var)
    #             ans_tok = ans_tok_var.data.cpu()
    #             if self.gpu:  #To one-hot
    #                 cur_inp = Variable(torch.zeros(
    #                     B, self.max_tok_num).scatter_(1, ans_tok, 1).cuda())
    #             else:
    #                 cur_inp = Variable(torch.zeros(
    #                     B, self.max_tok_num).scatter_(1, ans_tok, 1))
    #             cur_inp = cur_inp.unsqueeze(1)

    #             for idx, tok in enumerate(ans_tok.squeeze()):
    #                 if tok == 1:  #Find the <END> token
    #                     done_set.add(idx)
    #             t += 1

    #         cond_score = torch.stack(scores, 1)

    #     if reinforce:
    #         return cond_score, choices
    #     else:
    #         return cond_score
