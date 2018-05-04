# *- coding: utf-8 -*-
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from modules.word_embedding import WordEmbedding
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

def convert_tok_to_combined_indices(orig_col_lst, expanded_lst, indices):
    # indices refer to the expanded list which follows the following form:
    # [tok in each_col + ',']
    # [['boat', 'id'], ['table', 'id']] => ['boat', 'id', ',', 'table', 'id', ',']
    pass



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
        self.AGG_SQL_TOK= ['MAX', 'MIN', 'AVG', 'COUNT', 'SUM']
        self.SEL_SQL_TOK = ['SELECT', '<END>', ','] + self.AGG_SQL_TOK # w
        #gt_sel_seq = [0, ..., 1]
        self.SQL_TOK = ['NT', 'BTWN', 'EQL', 'GT', 'LT', 'GTEQL', 'LTEQL', 'NTEQL', 'IN', 'LKE', 'IS', 'XST']\
         + ['WHERE', 'AND', 'OR', '<END>']
         #WHERE = 12, END = 14
        self.GROUPBY_SQL_TOK = ['GROUPBY', '<END>', ',']
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

        #Predict number of cond
        self.cond_pred = Seq2SQLCondPredictor(
            N_word, N_h, N_depth, self.max_col_num, self.max_tok_num, gpu)


        self.CE = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax()
        self.bce_logit = nn.BCEWithLogitsLoss()
        if gpu:
            self.cuda()

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
        logging.info('method generate_gt_sel_seq')
        gt_sel_seq = []
        ret_seq = []
        for cur_q, cur_col, cur_query, mini_seq in zip(q, col, query, ans_seq):
            connect_col = [tok for col_tok in cur_col for tok in col_tok +[',']] # get the column names
            # [u'*', ',', u'col_tok_1', 'col_tok_2', ',' + ...]
            logging.info('connect_col: {0}'.format(connect_col))
            all_toks = self.SEL_SQL_TOK + connect_col + [None] + cur_q + [None]
            # logging.info('cur_col: {0}'.format(cur_col))
            # cur_col_list = [col.split(' ') for col in cur_col]
            # logging.info('cur_col_list: {0}'.format(cur_col_list))
            all_toks_condense = self.SEL_SQL_TOK + cur_col + [None] + cur_q + [None]
            logging.info('all_toks_gt: {0}'.format(all_toks))
            logging.info('all_toks_condense: {0}'.format(all_toks_condense))
            # get aggregators
            cur_seq = [all_toks.index('SELECT')]
            print('cur_seq', cur_seq)
            # print('cur_q', cur_q)
            # logging.info('cur_sel_gt_query{0}'.format(cur_query))
            # logging.info('mini_sel_gt_seq{0}'.format(mini_seq))
            for i, x in enumerate(mini_seq[0]):
                if mini_seq[0][i] != 0:
                    logging.info('orig_agg_index {0}'.format(mini_seq[0][i]))
                    cur_seq.append(all_toks.index(',')+ mini_seq[0][i]) 
                    # get the overall agg index - not plus 1 because all indices are already one off because deleted ''
                    # print('changed to ', all_toks.index(',')+ mini_seq[0][i])
                # cur_seq.append(len(self.SEL_SQL_TOK) + mini_seq[1][i]) # get the overall column index
                index = len(self.SEL_SQL_TOK) + mini_seq[1][i] # get the index for the normal word in all_toks without the expanded columns
                column_words = all_toks_condense[index] # list of words that are the column tokenized
                logging.warning('got word {0} for index{1}'.format(column_words, index))
                first_index = self.search(all_toks, column_words)
                if first_index != -1:
                    for i in range(first_index, first_index + len(column_words)):
                        cur_seq.append(i)
                        # logging.info('appended {0}'.format(i))
                    cur_seq.append(all_toks.index(','))
                # logging.info('appended , with index {0}'.format(all_toks.index(',')))
                    
                # cur_seq.append(all.toks.index(','))
            cur_seq = cur_seq[:-1] # delete the last ,
            cur_seq.append(all_toks.index('<END>'))
            # print('mini_seq[0] - agg', mini_seq[0])
            # print('mini_seq[1] - agg', mini_seq[1])
            # print('cur_seq_final', cur_seq)
            logging.warning('cur_sel_query_indices: {0}'.format(cur_seq))
            ret_seq.append(cur_seq)
        # print('gt_sel_seq',ret_seq)
        return ret_seq

    def clean_where_query(self, cur_where_query):
        logging.info('method clean_where_query')
        import re
        logging.info('cur_where_query: {0}'.format(cur_where_query))
        op_index = 100
        cond_ops = [x.lower() for x in self.COND_OPS]
        if 'order' in cur_where_query and 'by' == cur_where_query[cur_where_query.index('order') + 1]:
            cur_where_query = cur_where_query[:cur_where_query.index('order')] # gets rid of order by 
        if 'group' in cur_where_query and 'by' == cur_where_query[cur_where_query.index('group') + 1]:
            cur_where_query = cur_where_query[:cur_where_query.index('group')] # gets rid of group by 
        if 'union' in cur_where_query:
            cur_where_query = cur_where_query[:cur_where_query.index('union')] # gets rid of union
        if 'intersect' in cur_where_query:
            cur_where_query = cur_where_query[:cur_where_query.index('intersect')] # gets rid of intersect
        if 'limit' in cur_where_query:
            cur_where_query = cur_where_query[:cur_where_query.index('limit')] # gets rid of limit
        cur_where_query = filter(lambda a: a != '``' and a != '\'\'' and a !=  '\'', cur_where_query) # get rid of all quotations
        cur_where_query_copy = []
        for i, x in enumerate(cur_where_query):
            x_lowered = x.lower()
            if x == '=':
                if i > 0 and cur_where_query[i - 1] == '<':
                    cur_where_query_copy[-1] = 'lteql' # ['<', '='] => ['lteql']
                if i > 0 and cur_where_query[i - 1] == '>':
                    cur_where_query_copy[-1] = 'gteql' # ['>', '='] => ['gteql']
                else:
                    cur_where_query_copy.append('eql')
            elif x == '>':
                cur_where_query_copy.append('gt')
            elif x == '<':
                cur_where_query_copy.append('lt')
            elif x == '!' or x_lowered == 'not':
                cur_where_query_copy.append('nt')
            elif x_lowered == 'between':
                cur_where_query_copy.append('btwn')
            elif x_lowered == 'like':
                cur_where_query_copy.append('lke')
            elif x_lowered == 'exist':
                cur_where_query_copy.append('xst')
            else:
                cur_where_query_copy.append(x)
        cur_where_query = cur_where_query_copy
        logging.info('cur_where_query: {0}'.format(cur_where_query))
        quote_patt = re.compile('(\')(.*)')
        cur_where_query = map(lambda tok: quote_patt.search(tok).group(2) if quote_patt.search(tok) is not None \
                    else tok, cur_where_query)
        logging.info('cur_where_query_cleaned: {0}'.format(cur_where_query))
        t_pat = re.compile('(t\d+\.)(.*)')
        cur_where_query =  map(lambda tok: t_pat.search(tok).group(2) if t_pat.search(tok) is not None \
                    else tok, cur_where_query) # get rid of t1.colname -> colname
        # splice or / and 
        or_and_indices = [i for i, x in enumerate(cur_where_query) if x == 'or' or x == 'and']
        or_and_indices.append(len(cur_where_query) - 1)
        logging.info('or_and_indices: {0}'.format(or_and_indices))
        st = 0
        spliced = []
        for index in or_and_indices:
            mini_seq = cur_where_query[st:index] # doesn't include the or/and
            op_index = -1
            for i, x in enumerate(mini_seq):
                if x in cond_ops:
                    op_index = i
                    break
            spliced.extend(mini_seq[:i + 1]) # get rid of the value portion
            spliced.append(cur_where_query[index]) # or/and
            st = index + 1
        spliced = spliced[:-1]
        logging.info('spliced: {0}'.format(spliced))

        return spliced


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



    # def generate_gt_where_seq(self,  col, query):
    #     # data format
    #     # <BEG> WHERE cond1_col cond1_op cond1
    #     #         AND cond2_col cond2_op cond2
    #     #         AND ... <END>
    #     # WHERE cond1_col cond1_op cond1
    #     #         AND cond2_col cond2_op cond2
    #     #         AND ... <END>
    #     # SELECT agg_tok col ,  col , ... <END>
    #     # GROUPBY col ,  col , ... HAVING ... <END>
    #     # ORDERBY agg col1 ASC/DESC limit num <END> 
    #     logging.warning('method generate_gt_where_seq')
    #     ret_seq = []
    #     logging.info('q: {0}'.format(q))
    #     logging.info('col: {0}'.format(col))
    #     logging.info('quer: {0}'.format(query))
    #     for cur_q, cur_col, cur_query in zip(q, col, query):
    #         connect_col = [tok for col_tok in cur_col for tok in col_tok +[',']]
    #         # logging.info('connect_col: {0}'.format(connect_col))
    #         all_toks = self.SQL_TOK + connect_col + [None] + cur_q + [None]
    #         for i in range(len(all_toks)):
    #             if all_toks[i]:
    #                 all_toks[i] = all_toks[i].lower()
    #         logging.debug(all_toks)
    #         cur_query = [x.lower() for x in cur_query]
    #         cur_seq = [all_toks.index('where')]
    #         # logging.info('cur_query: {0}'.format(cur_query))
    #         if 'where' in cur_query:
    #             cur_where_query = cur_query[cur_query.index('where') + 1:]
    #             logging.warning('cur_where_query: %s', cur_where_query)
    #             cur_where_query = self.clean_where_query(cur_where_query)
    #             logging.warning('cur_where_query cleaned: %s', cur_where_query)
    #             # logging.info('all_toks: {0}'.format(all_toks))
    #             for item in cur_where_query:
    #                 if item in all_toks:
    #                     cur_seq += [all_toks.index(item.lower())]
    #                     logging.warning('item: {0} index:{1}'.format(item, all_toks.index(item.lower())))
    #                 else:
    #                     logging.warning('Not found in all_toks {0}'.format(item))
    #                     break
    #                     # assert False, "%s" % item
    #             # cur_seq = cur_seq + map(lambda tok:all_toks.index(tok)
    #             #                         if tok in all_toks else 0, cur_where_query)
    #             # logging.debug('cur_seq: %s', cur_seq)
    #         cur_seq.append(all_toks.index('<end>'))
    #         ret_seq.append(cur_seq)
    #         logging.warning('generate_gt_where_seq: {0}'.format(ret_seq))
    #         # exit(1) 

    #     return ret_seq

    def generate_gt_group_seq(self, q, col, query, ans_seq):
        gt_group_seq = []
        ret_seq = []
        for cur_q, cur_col, cur_query, mini_seq in zip(q, col, query, ans_seq):
            connect_col = [tok for col_tok in cur_col for tok in col_tok +[',']] # get the column names and tokenize
            # [u'*', ',', u'col_tok_1', 'col_tok_2', ',' + ...]

            all_toks = self.GROUPBY_SQL_TOK + connect_col + [None] + cur_q + [None]
            all_toks_condense = self.GROUPBY_SQL_TOK + cur_col + [None] + cur_q + [None]
            cur_seq = [all_toks.index('GROUPBY')]
            for i, x in enumerate(mini_seq[6]): # 6 = groupby component
                index = len(self.SEL_SQL_TOK) + mini_seq[6][i] # get the index for the normal word in all_toks without the expanded columns
                column_words = all_toks_condense[index] # list of words that are the column tokenized
                logging.warning('got word {0} for index{1}'.format(column_words, index))
                first_index = self.search(all_toks, column_words)
                if first_index != -1:
                    logging.warning('all_toks: {0}'.format(all_toks))
                    logging.warning('first_index: {0}'.format(first_index))
                    for i in range(first_index, first_index + len(column_words)):
                        cur_seq.append(i) # append the individual indices for the column's words tokenized
            cur_seq.append(all_toks.index('<END>'))
            logging.warning('cur_group_query_indices: {0}'.format(cur_seq))
            ret_seq.append(cur_seq)
        return ret_seq
        # print('gt_sel_seq',ret_seq)

        


    def forward(self, q, col, col_num, pred_entry,
                gt_where = None, gt_cond=None, reinforce=False, gt_sel=None, gt_groupby=None):
        logging.info('method seq2sqlforward')
        B = len(q)
        pred_agg, pred_sel, pred_cond, pred_groupby = pred_entry

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
            x_emb_var, x_len = self.embed_layer.gen_x_batch(q, col, is_q=True) # this creates the embeddings for the natural language question
            # batch = self.embed_layer.gen_col_batch(col, ) # [[col_tok_1, col_tok_2]] => generate average for each column
            col_inp_var, col_len = self.embed_layer.gen_x_batch(col, col, is_list=True)
            col_name_len = -1 #TODO fix
            max_x_len = max(x_len)
            if pred_cond:
                cond_score = self.cond_pred(x_emb_var, x_len, col_inp_var,
                                            col_name_len, col_len, col_num, gt_where, gt_cond,\
                                              reinforce=reinforce)

            if pred_sel:
                logging.info('seq2sqlforward non-trainable embeddings')
                logging.info('gt_sel: {0}'.format(gt_sel))
                sel_score = self.sel_pred(x_emb_var, x_len, col_inp_var,
                                          col_name_len, col_len, col_num, gt_index_seq=gt_sel)

            if pred_groupby:
                logging.info('gt_groupby: {0}'.format(gt_groupby))
                groupby_score = self.groupby_pred(x_emb_var, x_len, col_inp_var,
                                          col_name_len, col_len, col_num, gt_index_seq=gt_groupby)


            

        return (sel_score, cond_score, groupby_score)

    def loss(self, score, truth_num, pred_entry, gt_where, gt_sel, gt_groupby):
        pred_agg, pred_sel, pred_cond, pred_groupby = pred_entry
        sel_score, cond_score, groupby_score = score
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
            # logging.warning('groupby_truth: {0}'.format(groupby_truth))
            for b in range(len(groupby_truth)):
                if self.gpu:
                    groupby_truth_var = Variable(
                        torch.from_numpy(np.array(groupby_truth[b][1:])).cuda()) # 1: - get rid of the first token
                else:
                    groupby_truth_var = Variable(
                        torch.from_numpy(np.array(groupby_truth[b][1:])))
                # logging.warning('groupby_truth_var.size(): {0}'.format(groupby_truth_var.size()))
                # logging.warning('groupby_truth_var: {0}'.format(groupby_truth_var))
                groupby_pred_score = groupby_score[b, :len(groupby_truth[b]) - 1] # get rid of the last token
                # Reshape variables
                groupby_truth_var = groupby_truth_var.squeeze()
                groupby_pred_score = groupby_pred_score.squeeze()
                # logging.warning('groupby_pred_score_len: {0}'.format(len(groupby_pred_score.size())))
                if len(groupby_pred_score.size()) == 1:
                    groupby_pred_score = groupby_pred_score.unsqueeze(0)
                # logging.warning('groupby_pred_score.size(): {0}'.format(groupby_pred_score.size()))
                # logging.warning('groupby_truth_var.size(): {0}'.format(groupby_truth_var.size()))
                loss += ( self.CE(
                    groupby_pred_score, groupby_truth_var)) / len(groupby_truth)
        

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

    def gen_group_query(self, col, group_score, b, q, raw_col, raw_q, verbose=False):
        all_toks = self.GROUPBY_SQL_TOK + \
                   [x for toks in col[b] for x in
                    toks +[',']] + [''] + q[b] + [''] # should I delete q --> issue is that I get the wrong dimensions, should I change the embeddings??

        group_toks = []
        group_query = []


        to_idx = [x.lower() for x in raw_col + [',']]# possible columns to 

        logging.info('raw_col {0}'.format(raw_col))

        agg_sql_tok_lower = [x.lower() for x in self.AGG_SQL_TOK]
        for group_score_idx in group_score[b].data.cpu().numpy():
            group_tok = np.argmax(group_score_idx)

            try:
                group_val = all_toks[group_tok] # get all the words
                logging.warning('group_val {0}'.format(group_val))
            except:
                break
            
            if group_val == '<END>':
                logging.warning('found EXIT!!!')
                break
            group_toks.append(group_val) # get the column names
        if verbose:
            print group_toks

        group_toks = [x.lower() for x in group_toks]
        logging.warning('orig group toks {0}'.format(group_toks))
        group_toks = remove_duplicates(group_toks)


        pred_col = self.merge_tokens(group_toks, raw_q[b] + ' || ' + ' || '.join(raw_col))
        group_merged_toks = [pred_col]
        
        logging.warning('group_merged_toks: {0}'.format(group_merged_toks))
        # check if the word is a substring in a column
        poss_indices = [i for i, x in enumerate(col[b]) if pred_col in x and pred_col != '']
        logging.warning('poss_indices: {0}'.format(poss_indices))
        for index in poss_indices:
            logging.warning ('col_b_ index: {0}'.format(col[b][index]))
            word_seq = col[b][index]
            for word in word_seq:
                if word in to_idx:
                    group_query.append(to_idx.index(word)) # column names
        # if pred_col in to_idx:
        #     group_query.append(to_idx.index(pred_col)) # column name
        # else:
        #     logging.warning('unknown pred col: %s', str(pred_col))
        #     # check if the word is a substring in a column
        #     poss_indices = [i for i, x in enumerate(group_merged_toks) if pred_col in x and pred_col != '']
        #     if len(poss_indices) > 0 and 0 not in group_query:
        #         group_query.append(np.random.choice(poss_indices, 1)[0])
        #     pass

        logging.warning('pred sel_query indices: {0}'.format(group_query))
        return  group_query

    def gen_sel_query(self, col, sel_score, b, q, raw_col, raw_q, verbose=False):
        selects = []
        all_toks = self.SEL_SQL_TOK + \
                   [x for toks in col[b] for x in
                    toks +[',']] + [''] + q[b] + [''] # should I delete q --> issue is that I get the wrong dimensions, should I change the embeddings??
        logging.warning('col[b]: {0}'.format(col[b]))
        sel_toks = []
        sel_query = []


        to_idx = [x.lower() for x in raw_col + [',']]# possible columns to 

        logging.info('raw_col {0}'.format(raw_col))
        logging.info('all_toks: {0}'.format(all_toks))
        logging.info('all_toks_len: {0}'.format(len(all_toks)))

        agg_sql_tok_lower = [x.lower() for x in self.AGG_SQL_TOK]
        for sel_score_idx in sel_score[b].data.cpu().numpy():
            sel_tok = np.argmax(sel_score_idx)
            logging.info('sel_tok: {0}'.format(sel_tok))
            try:
                sel_val = all_toks[sel_tok] # get all the words
                logging.info('sel_val {0}'.format(sel_val))
            except:
                break
            
            if sel_val == '<END>':
                logging.info('found EXIT!!!')
                break
            sel_toks.append(sel_val) # get the column names
        if verbose:
            print sel_toks

        sel_toks = [x.lower() for x in sel_toks]

        sel_merged_toks = []
        agg_query = []
        if ',' in sel_toks:
            while sel_toks:
                try:
                    index_comma = sel_toks.index(',') # get index of ','
                except:
                    index_comma = len(sel_toks)
                # get all the words until first comma
                pred_col = self.merge_tokens(sel_toks[:index_comma], raw_q[b] + ' || ' + ' || '.join(raw_col))
                # merged column names
                sel_merged_toks.append(pred_col)
                sel_toks = sel_toks[index_comma + 1:] # move to the next one
        else:
            sel_toks_new = remove_duplicates(sel_toks)
            logging.warning('sel_toks_new: {0}'.format(sel_toks_new))
            for tok in sel_toks_new:
                pred_col = self.merge_tokens(tok, raw_q[b] + ' || ' + ' || '.join(raw_col))
                pred_col.replace(' ', '')
                sel_merged_toks.append(pred_col)
            # sel_merged_toks = [pred_col]
        
        logging.warning('sel_merged_toks: {0}'.format(sel_merged_toks))
        for pred_col in sel_merged_toks:

            if pred_col in to_idx:
                sel_query.append(to_idx.index(pred_col)) # column name
                if len(agg_query) != len(sel_query):
                    agg_query.append(0)
            elif pred_col in agg_sql_tok_lower:
                agg_query.append(agg_sql_tok_lower.index(pred_col)) # agg token
            else:
                logging.warning('unknown pred col: %s', str(pred_col))
                # check if the word is a substring in a column
                poss_indices = [i for i, x in enumerate(col[b]) if pred_col in x and pred_col != '']
                logging.warning('poss_indices: {0}'.format(poss_indices))
                for index in poss_indices:
                    logging.warning ('col_b_ index: {0}'.format(col[b][index]))
                    word_seq = col[b][index]
                    for word in word_seq:
                        if word in to_idx:
                            sel_query.append(to_idx.index(word))
                # if len(poss_indices) > 0 and 0 not in sel_query:
                #     sel_query.append(np.random.choice(poss_indices, 1)[0])
                
        if not sel_query:
            logging.warning('No items predicted in sel_query: predict them all')
            sel_query.append(0)
        if not agg_query:
            agg_query.append(0)

        if len(sel_query) != len(agg_query):
            logging.info('len sel_query: %d len_agg_query: %d', len(sel_query), len(agg_query))
        logging.warning('pred sel_query indices: {0}'.format(sel_query))
        logging.warning('pred agg_query indices: {0}'.format(agg_query))
        return  agg_query, sel_query

 

    def gen_where_query(self, col, cond_score, b, q, raw_col, raw_q, verbose=False, train=False):
        conds = []
        all_toks = self.SQL_TOK + \
                   col[b] + [''] + q[b] + ['']
        logging.warning('all_toks: {0}'.format(all_toks))


        cond_toks = []
        logging.error('method gen_where_query')
        first_ind_q = all_toks.index(q[b][0])
        logging.warning('all_toks: {0}'.format(all_toks))

        for i, where_score in enumerate(cond_score[b].data.cpu().numpy()):
            cond_tok = np.argmax(where_score)
            # print('where_score', where_score)
            cond_val = all_toks[cond_tok]
            if cond_val == '<END>':
                
                break
            cond_toks.append(cond_val)

        if verbose:
            print cond_toks
        # if len(cond_toks) > 0:
        #     cond_toks = cond_toks[1:] # get rid of the WHERE
        st = 0
        try:
            logging.error('cond_toks: {0}'.format(cond_toks))
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
                logging.error('len_sel_col is not 1: {0}'.format(sel_col))
            else:
                cur_cond[0] = col[b].index(sel_col[0])

            conds.append(cur_cond)
            if to_add is not None:
                conj.append(to_add)
            st = ed + 1

        # remove all until the right length
        len_conj = len(conds) - 1 if len(conds) >= 2 else 0
        conj = conj[:len_conj]
        logging.error('conds {0}'.format(conds))
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

        pred_agg, pred_sel, pred_cond, pred_group = pred_entry
        logging.warning('---------')
        logging.warning('check_acc')
        B = len(gt_queries)

        tot_err = agg_err = sel_err = cond_err = cond_num_err = sel_num_err = sel_val_err = agg_num_err = agg_val_err \
        = cond_col_err = cond_op_err = cond_val_err = cond_conj_err = 0.0
        tot_cor = agg_cor = sel_cor = cond_cor = cond_num_cor = sel_num_cor = sel_val_cor = agg_num_cor = agg_val_cor \
        = cond_col_cor = cond_op_cor = cond_val_cor = cond_conj_cor = 0.0
        tot_count = agg_count = sel_count = group_count = cond_count = 0.0
        group_cor = cond_val_cor
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
                    if set(agg_pred) == set(agg_gt):
                        agg_cor += 1
                    else:
                        
                        good = False

                

                # if flag and len(set(agg_pred)) != len(set(agg_gt)):
                #     flag = False
                #     agg_num_err += 1

                # if flag and set(agg_pred) != set(agg_gt):
                #     flag = False
                #     agg_val_err += 1

                # if not flag:
                #     agg_err += 1
                #     good = False
                #     logging.warning('agg_pred: %s', str(agg_pred))
                #     logging.warning('agg_gt: %s', str(agg_gt))

            if pred_sel:
                sel_pred = pred_qry['sel']
                sel_gt = gt_qry['sel']

                # flag = True
                # if len(sel_pred) != len(sel_gt): # if length is off
                #     flag = False
                #     logging.warning('wrong number of columns selected')
                #     sel_num_err += 1
                sel_pred_sorted = sorted(sel_pred)
                sel_gt_sorted = sorted(sel_gt)
                if not(sel_pred_sorted == sel_gt_sorted == [0]): # don't count [0]
                    logging.error('pred_sel: {0}, gt_sel: {1}'.format(sel_pred_sorted, sel_gt_sorted))
                    sel_count += 1
                    if sel_pred_sorted == sel_gt_sorted:
                        sel_cor += 1
                    else:
                        
                        good = False

                # for idx in range(len(sel_pred)):
                #     if not flag:
                #         break
                #     if flag and sel_pred_sorted[idx] != sel_gt_sorted[idx]:
                #         flag = False
                #         logging.warning('wrong columns!')
                #         sel_val_err += 1
                #         #sel_err += 1

                # if not flag:
                #     sel_err += 1
                #     logging.warning('incremented sel_err')
                #     logging.warning('sel_pred: {0}'.format(sel_pred))
                #     logging.warning('sel_gt: {0}'.format(sel_gt))
                #     good = False

            if pred_group:
                group_pred = pred_qry['group']
                group_gt = gt_qry['group'][0] # ignore HAVING for now
                if not (group_pred == group_gt == []):
                    logging.error('pred_group: {0}, gt_group: {1}'.format(group_pred, group_gt))
                    group_count += 1
                    if group_pred == group_gt:
                        group_cor += 1
                    else:
                        
                        good = False

                # if group_pred != group_gt:
                #     group_err += 1
                #     good = False
                #     logging.warning('incremented group_err')
                #     logging.warning('group_pred: {0}, group_gt: {1}'.format(group_pred, group_gt))


            if pred_cond:
                cond_pred = [x[:-1] for x in sorted(pred_qry['conds'])]
                cond_gt = [x[:-1] for x in sorted(gt_qry['cond'])]

                # # clean column names
                # for idx in range(len(cond_gt)):
                #     try:
                #         cond_gt[idx][2] = cond_gt[idx][2].replace('\'', '').replace('\"', '')
                #     except AttributeError as e:
                #         pass

                if not(cond_pred == cond_gt == []):
                    logging.error('pred_cond: {0}, gt_cond: {1}'.format(cond_pred, cond_gt))
                    cond_count += 1
                    if cond_pred == cond_gt and gt_qry['conj'] == pred_qry['conj']:
                        cond_cor += 1
                    else:

                        good = False
                # flag = True


                # # penalize if they predict the wrong number of conditions
                # if len(cond_pred) != len(cond_gt): 
                #     flag = False
                #     cond_num_err += 1
                #     logging.warning('wrong number of columns')

                # # penalize if they predict the different columns
                # if flag and set(
                #         x[0] for x in cond_pred) != set(x[0] for x in cond_gt): 
                #     flag = False
                #     cond_col_err += 1
                #     logging.warning('wrong column selected')


                # for idx in range(len(cond_pred)):
                #     if not flag:
                #         break
                #     # get indices of all possible matches
                #     gt_idx_lst = [i for i, y in enumerate([x[0] for x in cond_gt]) if y == cond_pred[idx][0]]
                #     condition_op = False
                #     # make sure there is one where there is the right operation
                #     for gt_idx in gt_idx_lst:
                #         if cond_gt[gt_idx][1] == cond_pred[idx][1]:
                #             condition_op = True

                #     if flag and not condition_op:
                #         flag = False
                #         cond_op_err += 1
                #         logging.warning('Wrong condition op')

                

                # # check right values for conditions
                # for idx in range(len(cond_pred)):
                #     if not flag:
                #         break
                #     gt_idx_lst = [i for i, y in enumerate([x[0] for x in cond_gt]) if y == cond_pred[idx][0]]
                #     value_correct = False
                #     for gt_idx in gt_idx_lst:
                #         if unicode(cond_gt[gt_idx][2]).lower() == unicode(cond_pred[idx][2]).lower():
                #             value_correct = True

                #     if flag and not value_correct:
                #         flag = False
                #         cond_val_err += 1
                #         logging.warning('Wrong values!')   
                # check correct conjunctions for the columns
                # if flag and gt_qry['conj'] != pred_qry['conj']:
                #     cond_conj_err += 1
                #     flag = False
                # if not flag:
                #     cond_err += 1
                #     logging.warning('cond_pred: {0}, cond_gt: {1}'.format(cond_pred, cond_gt))    
                #     good = False
                # else:
                #     logging.warning('predicted correctly cond!')
                #     logging.warning('cond_pred: {0}, cond_gt: {1}'.format(cond_pred, cond_gt))
 
            if good:
                tot_cor += 1

        # cond_err_breakdown = (cond_num_err, cond_op_err, cond_val_err)
        err_breakdown = np.zeros((4, 4))
        # err_breakdown[0,:2] = (agg_num_err, agg_val_err)
        # err_breakdown[1,:2] = (sel_num_err, sel_val_err)
        # err_breakdown[2,:] = (cond_num_err, cond_op_err, cond_val_err, cond_conj_err)
        # err_breakdown[3, 1] = group_err
        # logging.warning('err_breakdown: {0}'.format(err_breakdown))
        return np.array([(agg_cor, sel_cor, cond_cor, group_cor), (agg_count, sel_count, cond_count, group_count)]), \
        np.array((tot_cor, tot_count)), err_breakdown
    

           

    def gen_query(self, score, q, col, raw_q, raw_col, pred_entry,
                  reinforce=False, verbose=False):

        pred_agg, pred_sel, pred_cond, pred_groupby = pred_entry
        sel_score, cond_score, groupby_score = score

        ret_queries = []
        # if pred_agg:
        #     B = len(agg_score)
        if pred_sel:
            B = len(sel_score)
        elif pred_cond:
            B = len(cond_score[0]) if reinforce else len(cond_score)
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
                cur_query['agg'], cur_query['sel'] =  self.gen_sel_query(col, sel_score, b, q, raw_col[b], raw_q)
                # logging.debug('cur_query[sel]: %s', str(cur_query['sel']))
                # logging.debug('cur_query[agg]: %s', str(cur_query['agg']))
            if pred_cond:
                cur_query['conds'], cur_query['conj'] = self.gen_where_query(col, cond_score, b, q, raw_col[b], raw_q)
                logging.warning('cur_query_conds: %s', str(cur_query['conds']))
                logging.warning('cur_query_conj: %s', str(cur_query['conj'])) 
            if pred_groupby:
                cur_query['group'] = self.gen_group_query(col, cond_score, b, q, raw_col[b], raw_q)
                logging.error('cur_query[group]: {0}'.format(cur_query['group']))            
            # print('cur_query', cur_query)
            ret_queries.append(cur_query)


        return ret_queries
