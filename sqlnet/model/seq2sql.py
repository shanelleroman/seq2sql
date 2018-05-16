# *- coding: utf-8 -*-
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from modules.word_embedding import WordEmbedding, Type_Pred
from modules.aggregator_predict import AggPredictor
from modules.selection_predict import SelPredictor
from modules.seq2sql_subseq_predict import Seq2SQLSubSeqPredictor
from modules.seq2sql_condition_predict import Seq2SQLCondPredictor
import logging


# This is a re-implementation based on the following paper:

# Victor Zhong, Caiming Xiong, and Richard Socher. 2017.
# Seq2SQL: Generating Structured Queries from Natural Language using
# Reinforcement Learning. arXiv:1709.00103

def debug_print(var_name, var_val):
    print var_name + ": " 
    print var_val

def remove_duplicates(sel_toks):
# get rid of duplicates next to each other
    if not sel_toks:
        return sel_toks
    sel_toks_new = []
    for i in range(len(sel_toks) - 1):
        if sel_toks[i] != sel_toks[i + 1]:
            sel_toks_new.append(sel_toks[i])
            if i == len(sel_toks) - 2:
                sel_toks_new.append(sel_toks[i + 1])
    if not sel_toks_new:
        sel_toks_new = [sel_toks[0]]
    return sel_toks_new

def safe_index_get(lst, word, default=-2):
    try:
        return lst.index(word)
    except:
        return default





class Seq2SQL(nn.Module):
    def __init__(self, word_emb, N_word, N_h=100, N_depth=2,
                 gpu=False, trainable_emb=False):
        super(Seq2SQL, self).__init__()
        self.trainable_emb = trainable_emb

        self.gpu = gpu # GPU = what?
        self.N_h = N_h # what is this?
        self.N_depth = N_depth # what is this??

        self.max_col_num = 45
        self.max_tok_num = 200
        self.AGG_SQL_TOK= ['MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        # SELECT, END, '', MAX, MIN = new_index correct!!
        self.SEL_SQL_TOK = ['SELECT', '<END>'] + self.AGG_SQL_TOK # w
        #gt_sel_seq = [0, ..., 1]
        self.SQL_TOK = ['NT', 'BTWN', 'EQL', 'GT', 'LT', 'GTEQL', 'LTEQL', 'NTEQL', 'IN', 'LKE', 'IS', 'XST']\
         + ['WHERE', 'AND', 'OR', '<END>']
         #WHERE = 12, END = 14
        self.GROUPBY_SQL_TOK = ['GROUPBY', '<END>'] + self.AGG_SQL_TOK + \
        ['NT', 'BTWN', 'EQL', 'GT', 'LT', 'GTEQL', 'LTEQL', 'NTEQL', 'IN', 'LKE', 'IS', 'XST']
        self.ORDERBY_SQL_TOK = ['ORDERBY', '<END>'] + self.AGG_SQL_TOK + ['0', '1', 'LIMIT']
        self.COND_OPS = ['NT', 'BTWN', 'EQL', 'GT', 'LT', 'GTEQL', 'LTEQL', 'NTEQL', 'IN', 'LKE', 'IS', 'XST']

        #Word embedding
        if trainable_emb:

            self.agg_embed_layer = WordEmbedding(word_emb, N_word, gpu,
                                                 self.SQL_TOK, our_model=False,
                                                 trainable=trainable_emb)
            self.sel_embed_layer = WordEmbedding(word_emb, N_word, gpu,
                                                 self.SQL_TOK, our_model=False,
                                                 trainable=trainable_emb)
            self.cond_embed_layer = WordEmbedding(word_emb, N_word, gpu,
                                                  self.SQL_TOK, our_model=False,
                                                  trainable=trainable_emb)
        else:
            self.embed_layer = WordEmbedding(word_emb, N_word, gpu,
                                             self.SQL_TOK, trainable=trainable_emb)

        #Predict aggregator
        self.agg_pred = Seq2SQLSubSeqPredictor(N_word, N_h, N_depth, self.max_col_num, self.max_tok_num, gpu)

        # self.agg_pred = AggPredictor(N_word, N_h, N_depth, use_ca=False)

        #Predict selected column
        # self.sel_pred = SelPredictor(N_word, N_h, N_depth, self.max_tok_num,
                                     # use_ca=False)
        self.sel_pred = Seq2SQLSubSeqPredictor(N_word, N_h, N_depth, self.max_col_num, self.max_tok_num, gpu)


        #Predict GROUPBY columns
        self.groupby_pred = Seq2SQLSubSeqPredictor(N_word, N_h, N_depth, self.max_col_num, self.max_tok_num, gpu)

        #Predict ORDERBY columns
        self.orderby_pred = Seq2SQLSubSeqPredictor(N_word, N_h, N_depth, self.max_col_num, self.max_tok_num, gpu)

        #Predict number of cond
        self.cond_pred = Seq2SQLCondPredictor(N_word, N_h, N_depth, self.max_col_num, self.max_tok_num, gpu)


        self.CE = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax()
        self.bce_logit = nn.BCEWithLogitsLoss()

        if gpu:
            self.cuda()

    def generate_SQL_query(pred_queries, idxes, sql_data, table_data, st, end, table_ids):
        # logging.error('pred_queries[0]: {0}'.format(pred_queries[0]))
        # logging.error('table_id[0]: {0}'.format(table_ids[0]))
        # logging.error('table_data: {0}'.format(json.dumps(table_data[table_ids[0]]['table_data'], indent=4)))
        # # generate SELECT
        # database_info = table_data[table_ids[0]]
        # query = pred_queries[0]
        # sql_query  = 'SELECT'
        # for i, x in enumerate(query['agg']):
        #     if x != 0:
        #         sql_query += ' ' + self.AGG_SQL_TOK[x]
        #     sql_query += ' ' + database_info['col_map'][query['sel'][i]]
        # logging.error('sql_query: {0}'.format(sql_query)) 
        # exit(1)
        pass

    def search(self, source, target, start=0, end=None, forward=True):
        """Naive search for target in source."""
        m = len(source)
        if target is None:
            return -1
        n = len(target)
        if end is None:
            end = m
        else:
            end = min(end, m)
            # target is empty, or longer than source, so obviously can't be found.
            return -1
        if forward:
            x = range(start, end-n+1)
        else:
            x = range(end-n, start-1, -1)
        for i in x:
            if source[i:i+n] == target:
                return i
        return -1

    def generate_gt_sel_seq(self, q, col, query, ans_seq):
    # NOTE: these numbers are in terms of the overall all_toks!!!!
        logging.error('method generate_gt_sel_seq')
        gt_sel_seq = []
        ret_seq = []
        for cur_q, cur_col, cur_query, mini_seq in zip(q, col, query, ans_seq):
            all_toks = self.SEL_SQL_TOK + cur_col + [None] + cur_q + [None]
            logging.warning('all_toks: {0}'.format(all_toks))

            # get aggregators
            cur_seq = [all_toks.index('SELECT')]

            for i, x in enumerate(mini_seq[0]):
                logging.warning('orig_agg_index {0}'.format(mini_seq[0][i]))
                
                if mini_seq[0][i] != 0:
                    cur_seq.append(2 + mini_seq[0][i]) # index for the agg
                    logging.warning('new_agg_index {0}'.format(1 + mini_seq[0][i]))
                    logging.warning('new_agg_op: {0}'.format(all_toks[1 + mini_seq[0][i]]))
                    logging.warning('actual_agg_op: {0}'.format(self.AGG_SQL_TOK[mini_seq[0][i] - 1]))                    # orig index = 2 => 4
                
                index = len(self.SEL_SQL_TOK) + mini_seq[1][i] # get the index for the normal word in all_toks without the expanded columns
                logging.warning('index: {0}'.format(index))
                logging.warning('new_val: {0}'.format(all_toks[index]))
                cur_seq.append(index)
            cur_seq.append(all_toks.index('<END>'))

            logging.warning('cur_sel_query_indices: {0}'.format(cur_seq))
            ret_seq.append(cur_seq)

        return ret_seq

    def generate_gt_order_seq(self, q, col, query, ans_seq):
         # NOTE: these numbers are in terms of the overall all_toks!!!!
        logging.error('method generate_gt_order_seq')
        gt_order_seq = []
        ret_seq = []
        for cur_q, cur_col, cur_query, mini_seq in zip(q, col, query, ans_seq):
            all_toks = self.ORDERBY_SQL_TOK + cur_col + [None] + cur_q + [None]
            logging.info('all_toks: {0}'.format(all_toks))

            logging.info('cur_q: {0}'.format(cur_q))
            logging.info('cur_col: {0}'.format(cur_col))
            logging.error('cur_query: {0}'.format(cur_query))
            logging.info('mini_seq: {0}'.format(mini_seq))
            cur_seq = [all_toks.index('ORDERBY')]
            for i, x in enumerate(mini_seq[9]):
                # INDEX FOR AGG_ORDER
                if mini_seq[9][i] != 0:
                    my_lst = [''] + self.AGG_SQL_TOK
                    logging.warning('mini_seq[9]: {0}'.format(mini_seq[9][i]))
                    logging.info('actual AGG_ORDER: {0}'.format(my_lst[mini_seq[9][i]]))
                    logging.info('predicted AGG_ORDER: {0}'.format(all_toks[mini_seq[9][i] + 1]))
                    assert my_lst[mini_seq[9][i]] == all_toks[mini_seq[9][i] + 1], \
                    'predicted: {0}, actual {1}'.format(all_toks[mini_seq[9][i] + 1], my_lst[mini_seq[9][i]])
                    cur_seq.append(mini_seq[9][i] + 2)
                # INDEX FOR COL_ODER
                logging.warning('col_index: {0}'.format(mini_seq[10][i]))
                cur_seq.append(all_toks.index(cur_col[mini_seq[10][i]]))
            # INDEX FOR  ASC/DESC
            if mini_seq[12] != -1:
                logging.warning('mini_seq_12: {0}'.format(mini_seq[12]))
                my_lst = ['0', '1'] # 0 = DESC
                cur_seq.append(all_toks.index(my_lst[mini_seq[12]]))
                logging.error('ASC/DESC: {0}'.format(my_lst[int(mini_seq[12])]))
            if mini_seq[13]:
                logging.error('non null limit!')
                cur_seq.append(all_toks.index('LIMIT'))

            cur_seq.append(all_toks.index('<END>'))
            logging.error('gt_order_seq: {0}'.format(cur_seq))
            ret_seq.append(cur_seq)
        return ret_seq






    def generate_gt_where_seq(self, q, col_seq, cond_tuples):
        ret_seq = []
        for cur_q, ind_col_seq, cond_tuple in zip(q, col_seq, cond_tuples):

            all_toks = self.SQL_TOK + \
                   ind_col_seq + [''] + cur_q + ['']
            logging.warning('all_toks: {0}'.format(all_toks))
            logging.warning('cond: {0}'.format(cond_tuple))
            cond = [all_toks.index('WHERE')] # append index WHERE
            for ind_cond in cond_tuple:
                col_toks = ind_col_seq[ind_cond[0]] #list of the tokens that represent the column_token_sequence
                cond.append(all_toks.index(col_toks))
                cond.append(all_toks.index(self.COND_OPS[ind_cond[1]]))
            cond.append(all_toks.index('<END>'))
            logging.warning('appended {0}'.format(cond))
            ret_seq.append(cond)
        return ret_seq



    def generate_gt_group_seq(self, q, col, query, ans_seq):
        gt_group_seq = []
        ret_seq = []
        for cur_q, cur_col, cur_query, mini_seq in zip(q, col, query, ans_seq):
            # [u'*', ',', u'col_tok_1', 'col_tok_2', ',' + ...]
            all_toks = self.GROUPBY_SQL_TOK + cur_col + [None] + cur_q + [None]
            logging.warning('all_toks: {0}'.format(all_toks))
            logging.warning('mini_seq: {0}'.format(mini_seq))
            logging.warning('cur_q: {0}'.format(cur_q))
            logging.warning('cur_col: {0}'.format(cur_col))
            logging.warning('cur_query: {0}'.format(cur_query))
            cur_seq = [all_toks.index('GROUPBY')]
            for i, x in enumerate(mini_seq[6]): # 6 = groupby component
                index = len(self.GROUPBY_SQL_TOK) + mini_seq[6][i] # get the index for the normal word in all_toks without the expanded columns
                logging.warning('mini_seq_6_i: {0}'.format(mini_seq[6][i]))
                logging.warning('self_sql_tok: {0}'.format(self.GROUPBY_SQL_TOK))
                logging.warning('got word {0} for index{1}'.format(all_toks[index], index))
                logging.warning('having is: {0}'.format(mini_seq[8]))
                cur_seq.append(index) # append the individual indices for the column's words tokenized
                if mini_seq[8]:
                    cur_seq.append(mini_seq[8][0] + 1) # AGG INDEX
                    # make sure correct agg_tok
                    if mini_seq[8][0] != 0:
                        my_lst = [''] + self.AGG_SQL_TOK
                        assert all_toks[mini_seq[8][0] + 1] == my_lst[mini_seq[8][0]], 'predicted: {0} actual: {1}'.format(all_toks[mini_seq[8][0] + 2] , my_lst[mini_seq[8][0]])
                        logging.warning('predicted AGG TOK is: {0}'.format(all_toks[mini_seq[8][0] + 2]))
                        cur_seq.append(all_toks.index(cur_col[mini_seq[8][1]]))
                    logging.warning('predicted col tok is: {0}'.format(cur_col[mini_seq[8][1]]))
                    cur_seq.append(all_toks.index(self.COND_OPS[mini_seq[8][2]]))
                    logging.warning('predicted cond op is: {0}'.format(self.COND_OPS[mini_seq[8][2]]))

                
            cur_seq.append(all_toks.index('<END>'))
            logging.warning('cur_group_query_indices: {0}'.format(cur_seq))
            logging.warning('--------')
            ret_seq.append(cur_seq)
        return ret_seq


        


    def forward(self, q, col, col_num, pred_entry,
                gt_where = None, gt_cond=None, reinforce=False, gt_sel=None, gt_groupby=None, gt_orderby=None):
        logging.info('method seq2sqlforward')
        B = len(q)
        pred_agg, pred_sel, pred_cond, pred_groupby, pred_orderby = pred_entry

        agg_score = None
        sel_score = None
        cond_score = None
        groupby_score = None
        if self.trainable_emb:
  
            if pred_sel:
                x_emb_var, x_len = self.sel_embed_layer.gen_x_batch(q, col)
                batch = self.sel_embed_layer.gen_col_batch(col)
                col_inp_var, col_name_len, col_len = batch
                max_x_len = max(x_len)
                sel_score = self.sel_pred(x_emb_var, x_len, col_inp_var,col_name_len, col_len, col_num, gt_sel)

            if pred_cond:
                x_emb_var, x_len = self.cond_embed_layer.gen_x_batch(q, col)
                batch = self.cond_embed_layer.gen_col_batch(col)

                col_inp_var, col_name_len, col_len = batch
                max_x_len = max(x_len)
                cond_score = self.cond_pred(x_emb_var, x_len, col_inp_var,
                                            col_name_len, col_len, col_num,gt_where, gt_cond, reinforce=reinforce)
        else:
            # this creates the embeddings for the natural language question
            
            col_name_len = -1 #TODO fix
            
            if pred_cond:
                x_emb_var, x_len = self.embed_layer.gen_x_batch(q, col, is_q=True) 
                # [[col_tok_1, col_tok_2]] => generate average for each column
                col_inp_var, col_len = self.embed_layer.gen_x_batch(col, col, is_list=True)
                max_x_len = max(x_len)
                cond_score = self.cond_pred(x_emb_var, x_len, col_inp_var,
                                            col_name_len, col_len, col_num, gt_where, gt_cond,\
                                              reinforce=reinforce)

            if pred_sel:
                x_emb_var, x_len = self.embed_layer.gen_x_batch(q, col, is_q=True, type_pred=Type_Pred.sel) 
                # [[col_tok_1, col_tok_2]] => generate average for each column
                col_inp_var, col_len = self.embed_layer.gen_x_batch(col, col, is_list=True, type_pred=Type_Pred.sel)
                max_x_len = max(x_len)
                sel_score = self.sel_pred(x_emb_var, x_len, col_inp_var,
                                          col_name_len, col_len, col_num, gt_index_seq=gt_sel)

            if pred_groupby:
                x_emb_var, x_len = self.embed_layer.gen_x_batch(q, col, is_q=True, type_pred=Type_Pred.group) 
                # [[col_tok_1, col_tok_2]] => generate average for each column
                col_inp_var, col_len = self.embed_layer.gen_x_batch(col, col, is_list=True, type_pred=Type_Pred.group)
                max_x_len = max(x_len)
                groupby_score = self.groupby_pred(x_emb_var, x_len, col_inp_var,
                                          col_name_len, col_len, col_num, gt_index_seq=gt_groupby)

            if pred_orderby:
                x_emb_var, x_len = self.embed_layer.gen_x_batch(q, col, is_q=True, type_pred=Type_Pred.order) 
                # [[col_tok_1, col_tok_2]] => generate average for each column
                col_inp_var, col_len = self.embed_layer.gen_x_batch(col, col, is_list=True, type_pred=Type_Pred.order)
                max_x_len = max(x_len)
                orderby_score = self.orderby_pred(x_emb_var, x_len, col_inp_var,
                                          col_name_len, col_len, col_num, gt_index_seq=gt_orderby)





            

        return (sel_score, cond_score, groupby_score, orderby_score)

    def loss(self, score, truth_num, pred_entry, gt_where, gt_sel, gt_groupby, gt_orderby):
        pred_agg, pred_sel, pred_cond, pred_groupby, pred_orderby= pred_entry
        sel_score, cond_score, groupby_score, orderby_score = score
        loss = 0
        if pred_cond:
            for b in range(len(gt_where)):
                logging.warning('gt_where: {0}'.format(gt_where[b]))
                if self.gpu:
                    cond_truth_var = Variable(
                        torch.from_numpy(np.array(gt_where[b][1:])).cuda())
                    
                else:
                    cond_truth_var = Variable(
                        torch.from_numpy(np.array(gt_where[b][1:])))
                cond_pred_score = cond_score[b, :len(gt_where[b])-1]
                loss += ( self.CE(
                    cond_pred_score, cond_truth_var) / len(gt_where) )
        if pred_sel:
            sel_truth = gt_sel
            # logging.warning('sel_truth: {0}'.format(sel_truth))
            for b in range(len(sel_truth)):
                if self.gpu:
                    sel_truth_var = Variable(
                        torch.from_numpy(np.array(sel_truth[b][1:])).cuda()) # 1: - get rid of the first token
                else:
                    sel_truth_var = Variable(
                        torch.from_numpy(np.array(sel_truth[b][1:])))
                # logging.warning('sel_truth_var.size(): {0}'.format(sel_truth_var.size()))
                sel_pred_score = sel_score[b, :len(sel_truth[b]) - 1] # get rid of the last token
                # logging.warning('sel_pred_score.size(): {0}'.format(sel_pred_score.size()))
                loss += ( self.CE(
                    sel_pred_score, sel_truth_var)) / len(sel_truth)
        if pred_groupby:
            groupby_truth = gt_groupby

            for b in range(len(groupby_truth)):
                if self.gpu:
                    groupby_truth_var = Variable(
                        torch.from_numpy(np.array(groupby_truth[b][1:])).cuda()) # 1: - get rid of the first token
                else:
                    groupby_truth_var = Variable(
                        torch.from_numpy(np.array(groupby_truth[b][1:])))

                groupby_pred_score = groupby_score[b, :len(groupby_truth[b]) - 1] # get rid of the last token
                # Reshape variables
                groupby_truth_var = groupby_truth_var.squeeze()
                groupby_pred_score = groupby_pred_score.squeeze()

                if len(groupby_pred_score.size()) == 1:
                    groupby_pred_score = groupby_pred_score.unsqueeze(0)

                loss += ( self.CE(
                    groupby_pred_score, groupby_truth_var)) / len(groupby_truth)


        if pred_orderby:
            orderby_truth = gt_orderby

            for b in range(len(orderby_truth)):
                if self.gpu:
                    orderby_truth_var = Variable(
                        torch.from_numpy(np.array(orderby_truth[b][1:])).cuda()) # 1: - get rid of the first token
                else:
                    orderby_truth_var = Variable(
                        torch.from_numpy(np.array(orderby_truth[b][1:])))

                orderby_pred_score = orderby_score[b, :len(orderby_truth[b]) - 1] # get rid of the last token
                # Reshape variables
                orderby_truth_var = orderby_truth_var.squeeze()
                orderby_pred_score = orderby_pred_score.squeeze()

                if len(orderby_pred_score.size()) == 1:
                    orderby_pred_score = orderby_pred_score.unsqueeze(0)

                loss += ( self.CE(
                    orderby_pred_score, orderby_truth_var)) / len(orderby_truth)            
        

        return loss

    def reinforce_backward(self, score, rewards):
        agg_score, sel_score, cond_score = score

        cur_reward = rewards[:]
        eof = self.SQL_TOK.index('<END>')
        for t in range(len(cond_score[1])):
            reward_inp = torch.FloatTensor(cur_reward).unsqueeze(1)
            if self.gpu:
                reward_inp = reward_inp.cuda()
            cond_score[1][t].reinforce(reward_inp)

            for b in range(len(rewards)):
                if cond_score[1][t][b].data.cpu().numpy()[0] == eof:
                    cur_reward[b] = 0
        torch.autograd.backward(cond_score[1], [None for _ in cond_score[1]])
        return
    def merge_tokens(self, tok_list, raw_tok_str):
        tok_str = raw_tok_str.lower()
        alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$('
        special = {'-LRB-':'(', '-RRB-':')', '-LSB-':'[', '-RSB-':']',
                   '``':'"', '\'\'':'"', '--':u'\u2013'}
        ret = ''
        double_quote_appear = 0
        for raw_tok in tok_list:
            if not raw_tok:
                continue
            tok = special.get(raw_tok, raw_tok)
            if tok == '"':
                double_quote_appear = 1 - double_quote_appear

            if len(ret) == 0:
                pass
            elif len(ret) > 0 and ret + ' ' + tok in tok_str:
                ret = ret + ' '
            elif len(ret) > 0 and ret + tok in tok_str:
                pass
            elif tok == '"':
                if double_quote_appear:
                    ret = ret + ' '
            elif tok[0] not in alphabet:
                pass
            elif (ret[-1] not in ['(', '/', u'\u2013', '#', '$', '&']) and \
                 (ret[-1] != '"' or not double_quote_appear):
                ret = ret + ' '
            ret = ret + tok
        return ret.strip()

    def gen_order_query(self, col, orderby_score, b, q, raw_col, raw_q, verbose=False, train=True):
        logging.error('gen_order_query')
        all_toks = self.ORDERBY_SQL_TOK + col[b] + [''] + q[b] + ['']
        logging.info('all_toks: {0}'.format(all_toks))


        order_toks = []
        for i, order_score_idx in enumerate(orderby_score[b].data.cpu().numpy()):
            order_tok = np.argmax(order_score_idx)

            try:
                order_val = all_toks[order_tok] # get all the words
                logging.error('order_val {0}'.format(order_val))
            except:
                break
            
            if order_val == '<END>':
                logging.error('found EXIT!!!')
                break
            elif not train and i > 5:
                logging.error('cut it off short!!')
                break
            order_toks.append(order_val) #list of words
        logging.error('order_toks: {0}'.format(order_toks))
        order_query = [[], [], -1]
        looking_for = 'agg' # agg, column, order
        my_agg_lst = [''] + self.AGG_SQL_TOK
        i = 0
        len_order_toks = len(order_toks)
        if 'LIMIT' in order_toks:
            limit_query = True
        else:
            limit_query = None

        while i < len_order_toks:
            tok = order_toks[i]
            if looking_for == 'agg':

                if tok in self.AGG_SQL_TOK: # AGG sql tok
                    order_query[0].append(safe_index_get(self.AGG_SQL_TOK, order_toks[0], default=0)) # aggregator  
                    looking_for = 'column'
                    i += 1
                elif tok in col[b]: # COLUMN sql tok - append 0 for AGG sql
                    order_query[0].append(0)
                    order_query[1].append(col[b].index(tok))
                    looking_for = 'agg'
                    i += 1
                else:
                    looking_for = 'column'

                    # don't move token
            elif looking_for == 'column':
                if tok in col[b]:
                    order_query[1].append(col[b].index(tok))
                    i += 1
                else:
                    looking_for = 'order'
            else:
                my_lst = ['0', '1']
                order_query[2] = safe_index_get(my_lst, tok, default=0)  if order_query[0] else safe_index_get(my_lst, tok, default=-1)# ASC / DESC 
                i += 1

        if not train:
            order_query[0] = order_query[0][:2] # constrain to two items
            order_query[1] = order_query[1][:2]
            len_order_query_0 = len(order_query[0])
            len_order_query_1 = len(order_query[1])
            if len_order_query_0 < len_order_query_1:
                for i in range(len_order_query_1 - len_order_query_0):
                    order_query[0].append(0)
            elif len_order_query_1 < len_order_query_0: 
                order_query[0] = order_query[0][:len_order_query_1]
            
        logging.error('order_query: {0}'.format(order_query))
        return order_query, limit_query



    def gen_group_query(self, col, group_score, b, q, raw_col, raw_q, verbose=False, train=True):
        all_toks = self.GROUPBY_SQL_TOK + \
                   col[b] + ['']  + q[b] + [''] # should I delete q --> issue is that I get the wrong dimensions, should I change the embeddings??

        logging.warning('all_toks: {0}'.format(all_toks))
        group_toks = []
        group_query = []


        to_idx = col[b] # possible columns to 

        logging.info('to_idx {0}'.format(to_idx))
        index_having_cond_tok = -1 
        for i, group_score_idx in enumerate(group_score[b].data.cpu().numpy()):
            group_tok = np.argmax(group_score_idx)

            try:
                group_val = all_toks[group_tok] # get all the words
                # logging.error('group_val {0}'.format(group_val))
                if group_val in self.COND_OPS:
                    index_having_cond_tok = i
            except:
                break
            
            if group_val == '<END>':
                logging.error('found EXIT!!!')
                break
            elif not train and i > 3:
                logging.error('cut it off short!!')
                break
            group_toks.append(group_val) # get the column names
        logging.error('group_toks: {0}'.format(group_toks))
        # splice based on the first index of having
        if index_having_cond_tok != -1:
            # gets everything up to [having indices] if it makes sense, otherwise, just saves the first token as the group token
            group_splice = group_toks[:index_having_cond_tok - 1] if index_having_cond_tok  > 0 else group_toks[:1] 
            having_splice = group_toks[index_having_cond_tok - 1: ] if  index_having_cond_tok  > 0 else group_toks[1:]
        else:
            group_splice = [group_toks[0]] if len(group_toks) > 0 else group_toks
            having_splice = group_toks[1:]

        having_query = []
        if not train:
            # group_toks = remove_duplicates(group_toks)
            if group_splice:
                group_splice = [group_splice[0]] # get the first one 
            if '*' in group_splice:
                group_splice = []



        for tok in group_splice:
            if tok in col[b]:
                group_query.append(to_idx.index(tok))
        if len(having_splice) >= 3:
            having_query.append(safe_index_get(self.AGG_SQL_TOK, having_splice[0], default=0) + 1) # agg ops
            having_query.append(safe_index_get(to_idx, having_splice[1])) # column name
            having_query.append(safe_index_get(self.COND_OPS, having_splice[2]))
        elif len(having_splice) == 2:
            having_query.append(0)
            having_query.append(safe_index_get(to_idx, having_splice[0])) # column name
            having_query.append(safe_index_get(self.COND_OPS, having_splice[1]))
        else:
            having_query = []
        if not train:
            for i, x in enumerate(having_query):
                if x == -2:
                    if i == 0:
                        having_query[i] = 0 # predict no AGG
                    elif i == 1:
                        having_query[i] = np.random.choice(range(len(to_idx[1:])), 1)[0] # predict a random column index
                    else:
                        having_query[i] = np.random.choice(range(len(self.COND_OPS)), 1)[0]
            

        logging.error('group_query: {0}, having_query: {1}'.format(group_query, having_query))
        return  group_query, having_query

    def gen_sel_query(self, col, sel_score, b, q, raw_col, raw_q, verbose=False, train=True):
        selects = []
        logging.error('gen_sel_query')
        # the tokens with col[b] [[col_tok, col_tok], [col_tok, col_tok]]
        all_toks = self.SEL_SQL_TOK + col[b] + [''] + q[b] + ['']

        sel_toks = []
        sel_query = []
        agg_query = []

        # indices refer to the original ones listed in the dataset
        to_idx = col[b] 



        agg_sql_tok_lower = [x.lower() for x in self.AGG_SQL_TOK]
        for i, sel_score_idx in enumerate(sel_score[b].data.cpu().numpy()):
            sel_tok = np.argmax(sel_score_idx)
            logging.info('sel_tok: {0}'.format(sel_tok))
            try:
                # get the words 
                sel_val = all_toks[sel_tok] # get all the words
                logging.info('sel_val {0}'.format(sel_val))
            except:
                break
            
            if sel_val == '<END>':
                logging.info('found EXIT!!!')
                break
            elif not train and i >= 10: # constraining the number of select columns to Tao
                #TODO: ask Tao 
                break

            sel_toks.append(sel_val) # get the column names
        if verbose:
            print sel_toks
        logging.error('sel_toks: {0}'.format(sel_toks))
        sel_query = []    
        for tok in sel_toks:

            if tok in to_idx:
                sel_query.append(to_idx.index(tok))
            elif tok in self.AGG_SQL_TOK:
                logging.warning('tok: {0}'.format(tok))
                # logging.warning('potentially appened: {0}'.format(self.AGG_SQL_TOK.index(tok)))
                # logging.warning('actually appending: {0}'.format(self.AGG_SQL_TOK.index(tok) + 1))
                # logging.warning('alternate list: {0}'.format([''] + self.AGG_SQL_TOK))
                my_lst = self.AGG_SQL_TOK
                logging.warning('index_other option: {0}'.format(my_lst.index(tok)))
                agg_query.append(my_lst.index(tok)) # agg token
            else:
                logging.error('unknown pred col: {0}'.format(tok))
                if not train and len(sel_query) == 0:
                    sel_query.append(0)
        logging.error('sel_indices: {0}'.format(sel_query))

        if not train:
            len_agg_query = len(agg_query)
            
            if len(sel_query) == 0:
                sel_query.append(0)
            len_sel_query = len(sel_query)
            if len_agg_query < len_sel_query:
                for i in range(len_sel_query - len_agg_query):
                    agg_query.append(0)
                assert len_sel_query == len(agg_query)
            elif len_agg_query > len_sel_query:
                agg_query = agg_query[:len_sel_query]
        return  agg_query, sel_query

 

    def gen_where_query(self, col, cond_score, b, q, raw_col, raw_q, verbose=False, train=True):
        logging.warning('gen_where_query_train: {0}'.format(train))
        conds = []
        all_toks = self.SQL_TOK + \
                   col[b] + [''] + q[b] + ['']
        logging.warning('all_toks: {0}'.format(all_toks))


        cond_toks = []
        logging.warning('method gen_where_query')
        first_ind_q = all_toks.index(q[b][0])
        logging.warning('all_toks: {0}'.format(all_toks))

        for i, where_score in enumerate(cond_score[b].data.cpu().numpy()):
            cond_tok = np.argmax(where_score)
            # print('where_score', where_score)
            cond_val = all_toks[cond_tok]
            if not train and i > 8:
                break # restrain to 8 predicted tokens (3 clauses with 2 tokens each + 2 conj)
            if cond_val == '<END>':

                break
            cond_toks.append(cond_val)

        if verbose:
            print cond_toks
        # if len(cond_toks) > 0:
        #     cond_toks = cond_toks[1:] # get rid of the WHERE
        st = 0
        try:
            logging.warning('cond_toks: {0}'.format(cond_toks))
        except:
            pass
        to_add = None
        conj = []
        while st < len(cond_toks):
            cur_cond = [None, None, None]
            if 'AND' in cond_toks[st:]:
                ed = cond_toks[st:].index('AND') + st
                to_add = 'AND'
            elif 'OR' in  cond_toks[st:]:
                ed = cond_toks[st:].index('OR') + st
                to_add = 'OR'
            else:
                ed =len(cond_toks)
            try:
                logging.warning('cond_tok_st_ed: {0}'.format(cond_toks[st:ed]))
            except:
                pass
            if 'NT' in cond_toks[st:ed] and 'EQL' in cond_toks[st:ed]:
                op_prev = cond_toks[st:ed].index('NT') + st
                op = op_prev + 1
                cur_cond[1] = 7
            elif 'EQL' in cond_toks[st:ed]:
                op_prev = cond_toks[st:ed].index('EQL') + st
                cur_cond[1] = 2
                op = op_prev
            elif 'GT' in cond_toks[st:ed]:
                op_prev = cond_toks[st:ed].index('GT') + st
                cur_cond[1] = 3
                op = op_prev
            elif 'LT' in cond_toks[st:ed]:
                op_prev = cond_toks[st:ed].index('LT') + st
                cur_cond[1] = 4
                op = op_prev
            elif 'NT' in cond_toks[st:ed]:
                op_prev = cond_toks[st:ed].index('NT') + st
                cur_cond[1] = 0
                op = op_prev
            elif 'GTEQL' in cond_toks[st:ed]:
                op_prev = cond_toks[st:ed].index('GTEQL') + st
                cur_cond[1] = 5
                op = op_prev
            elif 'LTEQL' in cond_toks[st:ed]:
                op_prev = cond_toks[st:ed].index('LTEQL') + st
                cur_cond[1] = 6
                op = op_prev
            elif 'IN' in cond_toks[st:ed]:
                op_prev = cond_toks[st:ed].index('IN') + st
                cur_cond[1] = 8
                op = op_prev
            elif 'LKE' in cond_toks[st:ed]:
                op_prev = cond_toks[st:ed].index('LKE') + st
                cur_cond[1] = 9
                op = op_prev
            elif 'IS' in cond_toks[st:ed]:
                op_prev = cond_toks[st:ed].index('IS') + st
                cur_cond[1] = 10
                op = op_prev
            elif 'XST' in cond_toks[st:ed]:
                op_prev = cond_toks[st:ed].index('XST') + st
                cur_cond[1] = 11
                op = op_prev
            elif 'BTWN' in cond_toks[st:ed]:
                op_prev = cond_toks[st:ed].index('BTWN') + st
                cur_cond[1] = 1
                op = op_prev
            else:
                op = st
                op_prev = st + 1
                cur_cond[1] = 15
                # cur_cond[1] = 15

            sel_col = cond_toks[st:op_prev]
            ''' Possible sel_col:
            - [] ==> index(COND_OP) = 0
            - [['col_tok_1', 'cond_tok_2'], ['cond_tok_1', 'cond_tok_2']] => predict too many columns
            - [[col_tok_1, col_tok_2]]
            '''
            if len(sel_col) != 1 or sel_col[0] not in col[b]:
                cur_cond[0] = -1
                logging.warning('len_sel_col is not 1: {0}'.format(sel_col))
            else:
                cur_cond[0] = col[b].index(sel_col[0])
            if not train:
                if cur_cond[1] != 15 and cur_cond[0] != -1:
                    # impose additional heuristics for dev to improve the dev accuracy
                    conds.append(cur_cond)
            else:
                conds.append(cur_cond)
            if to_add is not None:
                conj.append(to_add)
            st = ed + 1

        # remove all until the right length
        len_conj = len(conds) - 1 if len(conds) >= 2 else 0
        conj = conj[:len_conj]
        logging.warning('conds {0}'.format(conds))
        return  conds, conj

    def check_acc(self, vis_info, pred_queries, gt_queries, pred_entry):
        def pretty_print(vis_data):
            print 'question:', vis_data[0]
            print 'headers: (%s)'%(' || '.join(vis_data[1]))
            print 'query:', vis_data[2]

        def gen_cond_str(conds, header):
            if len(conds) == 0:
                return 'None'
            cond_str = []
            for cond in conds:
                cond_str.append(
                    header[cond[0]] + ' ' + self.COND_OPS[cond[1]] + \
                    ' ' + unicode(cond[2]).lower())
            return 'WHERE ' + ' AND '.join(cond_str)

        pred_agg, pred_sel, pred_cond, pred_group, pred_orderby = pred_entry
        logging.warning('---------')
        logging.warning('check_acc')
        B = len(gt_queries)

        tot_err = agg_err = sel_err = cond_err = cond_num_err = sel_num_err = sel_val_err = agg_num_err = agg_val_err \
        = cond_col_err = cond_op_err = cond_val_err = cond_conj_err = 0.0
        tot_cor = agg_cor = sel_cor = cond_cor = cond_num_cor = sel_num_cor = sel_val_cor = agg_num_cor = agg_val_cor \
        = cond_col_cor = cond_op_cor = cond_val_cor = cond_conj_cor = 0.0
        tot_count = agg_count = sel_count = group_count = cond_count = 0.0
        group_cor = cond_val_cor
        having_count = having_cor = 0.0
        order_count = order_cor = 0.0
        limit_count = limit_cor = 0.0
        agg_ops = ['None', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        for b, (pred_qry, gt_qry) in enumerate(zip(pred_queries, gt_queries)):
            logging.error('------')
            tot_count += 1
            good = True
            if pred_agg:
                agg_pred = [x for x in pred_qry['agg']]
                agg_gt = [x for x in gt_qry['agg']]

                flag = True
                if not(set(agg_pred) == set(agg_gt) == {0}):
                    logging.error('pred_agg: {0}, gt_agg: {1}'.format(agg_pred, agg_gt))
                    agg_count += 1
                    # going to count this batch
                    if agg_pred == agg_gt:
                        agg_cor += 1
                    else:
                        good = False
                else:
                    logging.error('!!!!!!!!!!')



            if pred_sel:
                sel_pred = pred_qry['sel']
                sel_gt = gt_qry['sel']


                sel_pred_sorted = sorted(sel_pred)
                sel_gt_sorted = sorted(sel_gt)
                if not(sel_pred_sorted == sel_gt_sorted == [0]): # don't count [0]
                    logging.error('pred_sel: {0}, gt_sel: {1}'.format(sel_pred_sorted, sel_gt_sorted))
                    sel_count += 1
                    if sel_pred_sorted == sel_gt_sorted:
                        sel_cor += 1
                    else:
                        good = False
                else:
                    logging.error('!!!!!!!!!!')



            if pred_group:
                group_pred = pred_qry['group']
                group_gt = gt_qry['group'][0] if not gt_qry['group'][0] else [gt_qry['group'][0]]  # ignore HAVING for now 
                if not (group_pred == group_gt == []):
                    logging.error('pred_group: {0}, gt_group: {1}'.format(group_pred, group_gt))
                    group_count += 1
                    if group_pred == group_gt:
                        group_cor += 1
                    else:
                        good = False
                else:
                    logging.error('!!!!!!!!!!')

                # having accuracy
                having_pred = pred_qry['having']
                # logging.warning('gt_qry: {0}'.format(json.dumps(gt_qry, indent=4)))
                having_gt = gt_qry['group'][1][:-1] if len(gt_qry['group']) == 2 else []
                logging.error('having_gt: {0}'.format(having_gt))
                logging.error('gt_qry[group]: {0}'.format(gt_qry['group']))
                if not (having_pred == having_gt == []):
                    logging.error('pred_having: {0}, gt_having: {1}'.format(having_pred, having_gt))
                    having_count += 1
                    if having_pred == having_gt:
                        having_cor += 1
                    else:
                        good = False
                else:
                    logging.error('!!!!!!!!!!')

            if pred_orderby:
                order_pred = pred_qry['order']
                order_gt = gt_qry['order']
                if not (order_gt == order_pred == [[], [], -1]):
                    logging.error('pred_order: {0}, gt_order: {1}'.format(order_pred, order_gt))
                    order_count += 1
                    if order_gt == order_pred:
                        order_cor += 1
                    else:
                        good = False
                else:
                    logging.error('!!!!!!!!!!')

                limit_pred = pred_qry['limit']
                limit_gt = gt_qry['limit']
                if limit_pred or limit_gt:
                    logging.error('pred_limit: {0}, gt_limit: {1}'.format(limit_pred, limit_gt))
                    limit_count += 1
                    if limit_pred == limit_gt:
                        limit_cor += 1
                    else:
                        good = False



            if pred_cond:
                cond_pred = [x[:-1] for x in sorted(pred_qry['conds'])]
                cond_gt = [x[:-1] for x in sorted(gt_qry['cond'])]


                if not(cond_pred == cond_gt == []):
                    logging.error('pred_cond: {0}, gt_cond: {1}'.format(cond_pred, cond_gt))
                    cond_count += 1
                    if cond_pred == cond_gt and gt_qry['conj'] == pred_qry['conj']:
                        cond_cor += 1
                    else:
                        good = False
                else:
                    logging.error('!!!!!!!!!!')


            if good:
                tot_cor += 1

        # cond_err_breakdown = (cond_num_err, cond_op_err, cond_val_err)
        err_breakdown = np.zeros((4, 4))
        # err_breakdown[0,:2] = (agg_num_err, agg_val_err)
        # err_breakdown[1,:2] = (sel_num_err, sel_val_err)
        # err_breakdown[2,:] = (cond_num_err, cond_op_err, cond_val_err, cond_conj_err)
        # err_breakdown[3, 1] = group_err
        # logging.warning('err_breakdown: {0}'.format(err_breakdown))
        return np.array([(agg_cor, sel_cor, cond_cor, group_cor, having_cor, order_cor, limit_cor), \
            (agg_count, sel_count, cond_count, group_count, having_count, order_count, limit_count)]), \
        np.array((tot_cor, tot_count)), err_breakdown
    

           

    def gen_query(self, score, q, col, raw_q, raw_col, pred_entry,
                  reinforce=False, verbose=False, train=True):
        logging.error('gen_query_train: {0}'.format(train))
        pred_agg, pred_sel, pred_cond, pred_groupby, pred_orderby = pred_entry
        sel_score, cond_score, groupby_score, orderby_score = score

        ret_queries = []
        # if pred_agg:
        #     B = len(agg_score)
        if pred_sel:
            B = len(sel_score)
        elif pred_cond:
            B = len(cond_score[0]) if reinforce else len(cond_score)
        elif pred_groupby:
            B = len(groupby_score)
        elif pred_orderby:
            B = len(orderby_score)
        for b in range(B):
            cur_query = {}
            # raw_col[b] = [(0, [u'*']), (1, [u'template', u'type', u'code']), \
            # (2, [u'template', u'type', u'description']), (3, [u'template', u'id'])]
            if pred_sel:
                cur_query['sel'] = np.argmax(sel_score[b].data.cpu().numpy())
                try:
                    logging.warning('raw_q[b]: {0}'.format(unicode(raw_q[b])))
                except:
                    pass
                cur_query['agg'], cur_query['sel'] =  self.gen_sel_query(col, sel_score, b, q, raw_col[b], raw_q, train=train)
                # logging.debug('cur_query[sel]: %s', str(cur_query['sel']))
                # logging.debug('cur_query[agg]: %s', str(cur_query['agg']))
            if pred_cond:
                cur_query['conds'], cur_query['conj'] = self.gen_where_query(col, cond_score, b, q, raw_col[b], raw_q, train=train)
                logging.warning('cur_query_conds: %s', str(cur_query['conds']))
                logging.warning('cur_query_conj: %s', str(cur_query['conj'])) 
            if pred_groupby:
                cur_query['group'], cur_query['having'] = self.gen_group_query(col, groupby_score, b, q, raw_col[b], raw_q, train=train)
            if pred_orderby:
                cur_query['order'], cur_query['limit'] = self.gen_order_query(col, orderby_score, b, q, raw_col[b], raw_q, train=train)           
            
            ret_queries.append(cur_query)


        return ret_queries
