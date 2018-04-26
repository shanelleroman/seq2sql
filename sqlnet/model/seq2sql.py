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
         #WHERE = 13, END = 16
        self.SQL_TOK = ['EQL', 'WHERE', 'AND', 'OR', # what is the <UNK> stand for???
                       '<END>' , 'GT', 'LT', 'NT'] # EQL = , GT > , LT <, <=, >=, != # should be all the SQL tokens #TODO: add back <UNK>
        # gt_where_seq == [2, 1] or [2, 0, 10, 0, 42, 0, 36, 18, 1]
        #7 = <BEG>, 1 = <END>
        # NEED TO EMULATE FOR AGG_SQL_TOK
        self.COND_OPS = ['EQL', 'GT', 'LT', 'NT']

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
                                             self.SQL_TOK, our_model=False,
                                             trainable=trainable_emb)

        #Predict aggregator
        self.agg_pred = Seq2SQLSubSeqPredictor(N_word, N_h, N_depth, self.max_col_num, self.max_tok_num, gpu)
        # self.agg_pred = AggPredictor(N_word, N_h, N_depth, use_ca=False)

        #Predict selected column
        # self.sel_pred = SelPredictor(N_word, N_h, N_depth, self.max_tok_num,
                                     # use_ca=False)
        self.sel_pred = Seq2SQLSubSeqPredictor(N_word, N_h, N_depth, self.max_col_num, self.max_tok_num, gpu)

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
        n = len(target)
        if end is None:
            end = m
        else:
            end = min(end, m)
        if n == 0 or (end-start) < n:
            # target is empty, or longer than source, so obviously can't be found.
            return None
        if forward:
            x = range(start, end-n+1)
        else:
            x = range(end-n, start-1, -1)
        for i in x:
            if source[i:i+n] == target:
                return i
        return None

    def generate_gt_sel_seq(self, q, col, query, ans_seq):
    # NOTE: these numbers are in terms of the overall all_toks!!!!
        gt_sel_seq = []
        ret_seq = []
        for cur_q, cur_col, cur_query, mini_seq in zip(q, col, query, ans_seq):
            connect_col = [tok for col_tok in cur_col for tok in col_tok+[',']]
            all_toks = self.SEL_SQL_TOK + connect_col + [None] + cur_q + [None]
            all_toks_condense = self.SEL_SQL_TOK + cur_col + [None] + cur_q + [None]
            logging.warning('all_toks_gt: {0}'.format(all_toks))
            logging.warning('all_toks_condense: {0}'.format(all_toks_condense))
            # get aggregators
            cur_seq = [all_toks.index('SELECT')]
            # print('cur_seq', cur_seq)
            # print('cur_q', cur_q)
            logging.warning('cur_sel_gt_query{0}'.format(cur_query))
            logging.warning('mini_sel_gt_seq{0}'.format(mini_seq))
            for i, x in enumerate(mini_seq[0]):
                if mini_seq[0][i] != 0:
                    logging.warning('orig_agg_index', mini_seq[0][i])
                    cur_seq.append(all_toks.index(',')+ mini_seq[0][i]) 
                    # get the overall agg index - not plus 1 because all indices are already one off because deleted ''
                    # print('changed to ', all_toks.index(',')+ mini_seq[0][i])
                # cur_seq.append(len(self.SEL_SQL_TOK) + mini_seq[1][i]) # get the overall column index
                index = len(self.SEL_SQL_TOK) + mini_seq[1][i] # get the index for the normal word in all_toks without the expanded columns
                column_words = all_toks_condense[index] # list of words that are the column tokenized
                logging.warning('got word{0} for index{1}'.format(column_words, index))
                first_index = self.search(all_toks, column_words)
                for i in range(first_index, first_index + len(column_words)):
                    cur_seq.append(i)
                    logging.warning('appended {0}'.format(i))
                cur_seq.append(all_toks.index(','))
                logging.warning('appended , with index {0}'.format(all_toks.index(',')))
                    
                # cur_seq.append(all.toks.index(','))
            cur_seq = cur_seq[:-1] # delete the last ,
            cur_seq.append(all_toks.index('<END>'))
            # print('mini_seq[0] - agg', mini_seq[0])
            # print('mini_seq[1] - agg', mini_seq[1])
            # print('cur_seq_final', cur_seq)
            logging.warning('cur_sel_query_indices: %s', cur_seq)
            ret_seq.append(cur_seq)
        # print('gt_sel_seq',ret_seq)
        return ret_seq

    def clean_where_query(self, cur_where_query):
        import re
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
        if '=' in cur_where_query:
            cur_where_query[cur_where_query.index('=')] = 'eql'
        if '>' in cur_where_query:
            cur_where_query[cur_where_query.index('>')] = 'gt'
        if '<' in cur_where_query:
            cur_where_query[cur_where_query.index('<')] = 'lt'
        if '!' in cur_where_query:
            cur_where_query[cur_where_query.index('!')] = 'nt'

        t_pat = re.compile('(t\d+\.)(.*)')
        return map(lambda tok: t_pat.search(tok).group(2) if t_pat.search(tok) is not None \
                    else tok, cur_where_query) # get rid of t1.colname -> colname


    def generate_gt_where_seq(self, q, col, query):
        
        # data format
        # <BEG> WHERE cond1_col cond1_op cond1
        #         AND cond2_col cond2_op cond2
        #         AND ... <END>
        # WHERE cond1_col cond1_op cond1
        #         AND cond2_col cond2_op cond2
        #         AND ... <END>
        # SELECT agg_tok col ,  col , ... <END>
        # GROUPBY col ,  col , ... HAVING ... <END>
        # ORDERBY agg col1 ASC/DESC limit num <END> 
        ret_seq = []
        for cur_q, cur_col, cur_query in zip(q, col, query):
            connect_col = [tok for col_tok in cur_col for tok in col_tok+[',']]
            all_toks = self.SQL_TOK + connect_col + [None] + cur_q + [None]
            for i in range(len(all_toks)):
                if all_toks[i]:
                    all_toks[i] = all_toks[i].lower()
            logging.debug(all_toks)
            cur_seq = [all_toks.index('where')]
            if 'where' in cur_query:
                cur_where_query = cur_query[cur_query.index('where') + 1:]
                # logging.debug('cur_where_query: %s', cur_where_query)
                cur_where_query = self.clean_where_query(cur_where_query)
                # logging.warning('cur_where_query: %s', cur_where_query)
                # logging.warning('all_toks: {0}'.format(all_toks))
                for item in cur_where_query:
                    if item in all_toks:
                        
                        cur_seq += [all_toks.index(item.lower())]
                        # logging.warning('item: {0} index:{1}'.format(item, all_toks.index(item.lower())))
                    else:
                        logging.info('Not found in all_toks {0}'.format(item))
                        # assert False, "%s" % item
                # cur_seq = cur_seq + map(lambda tok:all_toks.index(tok)
                #                         if tok in all_toks else 0, cur_where_query)
                # logging.debug('cur_seq: %s', cur_seq)
            cur_seq.append(all_toks.index('<end>'))
            ret_seq.append(cur_seq)
        # logging.warning('generate_gt_where_seq: %s', ret_seq) 
        return ret_seq


    def forward(self, q, col, col_num, pred_entry,
                gt_where = None, gt_cond=None, reinforce=False, gt_sel=None):
        B = len(q)
        pred_agg, pred_sel, pred_cond = pred_entry

        agg_score = None
        sel_score = None
        cond_score = None
        if self.trainable_emb:
            # if pred_agg:

            #     x_emb_var, x_len = self.agg_embed_layer.gen_x_batch(q, col)
            #     batch = self.agg_embed_layer.gen_col_batch(col)
            #     col_inp_var, col_name_len, col_len = batch
            #     max_x_len = max(x_len)
            #     agg_score = self.agg_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num)

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
            x_emb_var, x_len = self.embed_layer.gen_x_batch(q, col) # this creates the embeddings for the natural language question
            batch = self.embed_layer.gen_col_batch(col)
            col_inp_var, col_name_len, col_len = batch
            max_x_len = max(x_len)
            if pred_cond:
                cond_score = self.cond_pred(x_emb_var, x_len, col_inp_var,
                                            col_name_len, col_len, col_num, gt_where, gt_cond,\
                                              reinforce=reinforce)

            # if pred_agg:
            #     print 'agg_pred'
            #     agg_score = self.agg_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num)


            if pred_sel:
                sel_score = self.sel_pred(x_emb_var, x_len, col_inp_var,
                                          col_name_len, col_len, col_num, gt_index_seq=gt_sel)

            

        return (sel_score, cond_score)

    def loss(self, score, truth_num, pred_entry, gt_where, gt_sel):
        pred_agg, pred_sel, pred_cond = pred_entry
        sel_score, cond_score = score
        loss = 0
        # if pred_agg:
        #     agg_truth = map(lambda x:x[0], truth_num)
        #     data = torch.from_numpy(np.array(agg_truth))
        #     if self.gpu:
        #         agg_truth_var = Variable(data.cuda())
        #     else:
        #         agg_truth_var = Variable(data)

        #     loss += self.CE(agg_score, agg_truth_var)
        if pred_cond:
            for b in range(len(gt_where)):
                if self.gpu:
                    cond_truth_var = Variable(
                        torch.from_numpy(np.array(gt_where[b][1:])).cuda())
                    
                else:
                    cond_truth_var = Variable(
                        torch.from_numpy(np.array(gt_where[b][1:])))
                cond_pred_score = cond_score[b, :len(gt_where[b])-1]
                # print('cond_score[b, :].size()',cond_score[b, :].size())
                # print('gt_where', gt_where)
                # print('gt_where[b]', gt_where[b])
                # print('len(gt_where[b])-1', len(gt_where[b])-1)
                # print('gt_where.size()', len(gt_where))
                # print ('cond_score.size()', cond_score.size())
                # print('cond_pred_score.size()', cond_pred_score.size())
                # print('cond_truth_var.size()', cond_truth_var.size())
                loss += ( self.CE(
                    cond_pred_score, cond_truth_var) / len(gt_where) )
        if pred_sel:
            sel_truth = gt_sel
            # logging.debug('sel_truth: %s', str(sel_truth))
            # print('sel_score.size()', sel_score.size())
            for b in range(len(sel_truth)):
                if self.gpu:
                    sel_truth_var = Variable(
                        torch.from_numpy(np.array(sel_truth[b][1:])).cuda()) # 1: - get rid of the first token
                else:
                    sel_truth_var = Variable(
                        torch.from_numpy(np.array(sel_truth[b][1:])))
                sel_pred_score = sel_score[b, :len(sel_truth[b]) - 1] # get rid of the last token
                # print('sel_truth[b]', sel_truth[b])
                # print('sel_pred_score.size()', sel_pred_score.size())
                # print('sel_truth_var.size()', sel_truth_var.size())
                loss += ( self.CE(
                    sel_pred_score, sel_truth_var)) / len(sel_truth)
            
            # print('sel_truth', sel_truth)
            # exit(1)
            # data = torch.from_numpy(sel_truth)
            # if self.gpu:
            #     sel_truth_var = Variable(data).cuda()
            # else:
            #     sel_truth_var = Variable(data)

            # loss += self.CE(sel_score, sel_truth_var)

        

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

    def gen_sel_query(self, col, sel_score, b, q, raw_col, raw_q, verbose=False):
        selects = []
        all_toks = self.SEL_SQL_TOK + \
                   [x for toks in col[b] for x in
                    toks+[',']] + [''] + q[b] + [''] # should I delete q --> issue is that I get the wrong dimensions, should I change the embeddings??
        # print('all_toks', all_toks)
        # print('all_toks_len', len(all_toks))
        sel_toks = []
        sel_query = []

        # print('sel_score.size()', sel_score.size())
        # print ('sel_score[b].size()', sel_score[b].size())
        # print ('sel_score', sel_score[b].data.cpu().numpy())
        to_idx = [x.lower() for x in raw_col[b]] + [',']# possible columns to predict
        agg_query = []
        agg_sql_tok_lower = [x.lower() for x in self.AGG_SQL_TOK]
        for sel_score_idx in sel_score[b].data.cpu().numpy():
            # print('sel_score_idx', sel_score_idx)
            sel_tok = np.argmax(sel_score_idx)
            # print('max: ', np.max(sel_score_idx)) # should I maybe get the next highest???
            logging.warning('sel_tok {0}'.format(sel_tok))
            sel_val = all_toks[sel_tok] # get all the words
            logging.warning('sel_val {0}'.format(sel_val))
            if sel_val == '<END>':
                logging.warning('found EXIT!!!')
                # exit(1)
                break
            sel_toks.append(sel_val) # get the column names
        if verbose:
            print sel_toks
        # if len(sel_toks) > 0:
        #     sel_toks = sel_toks[1:] # gets rid of <SEL>
        sel_toks = [x.lower() for x in sel_toks]
        logging.warning('sel_toks: %s', str(sel_toks))
        logging.warning('to_idx: %s', str(to_idx))
        logging.warning('self.agg_sql_tok: %s', str(self.AGG_SQL_TOK))
        sel_merged_toks = []
        if ',' in sel_toks:
            while sel_toks:
                try:
                    index_comma = sel_toks.index(',') # get index of ','
                except:
                    index_comma = len(sel_toks)
                pred_col = self.merge_tokens(sel_toks[:index_comma], raw_q[b] + ' || ' + \
                                        ' || '.join(raw_col[b])) # get all the words until first comma
                logging.warning('merged pred_col {0}'.format(pred_col)) # merged column names
                sel_merged_toks.append(pred_col)
                sel_toks = sel_toks[index_comma + 1:] # move to the next one
        else:
            pred_col = self.merge_tokens(sel_toks, raw_q[b] + ' || ' + \
                                        ' || '.join(raw_col[b]))
            sel_merged_toks = [pred_col]
        
        for pred_col in sel_merged_toks:
            logging.warning('looking for {0}'.format(pred_col))
            if pred_col in to_idx:
                sel_query.append(to_idx.index(pred_col)) # column name
                logging.warning('appended {0}'.format(to_idx.index(pred_col)))
                if len(agg_query) != len(sel_query):
                    agg_query.append(0)
            elif pred_col in agg_sql_tok_lower:
                agg_query.append(agg_sql_tok_lower.index(pred_col)) # agg token
            else:
                logging.warning('unknown pred col: %s', str(pred_col))
                pass
                # if 0 not in sel_query:
                #     sel_query.append(0)
        if not sel_query:
            logging.warning('No items predicted in sel_query: predict them all')
            sel_query.append(0)
        if not agg_query:
            agg_query.append(0)

        if len(sel_query) != len(agg_query):
            logging.warning('len sel_query: %d len_agg_query: %d', len(sel_query), len(agg_query))
        # assert len(sel_query) == len(agg_query), " len_sel_query: %d len_agg_query: %d" % (len(sel_query), len(agg_query))
        logging.warning('pred sel_query indices: {0}'.format(sel_query))
        return  agg_query, sel_query


    def gen_where_query(self, col, cond_score, b, q, raw_col, raw_q, verbose=False):
        conds = []
        all_toks = self.SQL_TOK + \
                   [x for toks in col[b] for x in
                    toks+[',']] + [''] + q[b] + ['']
        cond_toks = []

        # print 'making where'
        # print 'raw_col', raw_col
        for where_score in cond_score[b].data.cpu().numpy():
            cond_tok = np.argmax(where_score)
            # print('where_score', where_score)
            # logging.warning('cond_tok: %s', str(cond_tok))
            cond_val = all_toks[cond_tok]
            if cond_val == '<END>':
                break
            cond_toks.append(cond_val)

        if verbose:
            print cond_toks
        # if len(cond_toks) > 0:
        #     cond_toks = cond_toks[1:] # get rid of the WHERE
        st = 0
        # logging.warning('cond_toks: %s', str(cond_toks))
        while st < len(cond_toks):
            print ('st', st)
            cur_cond = [None, None, None]
            ed = len(cond_toks) if 'and' not in cond_toks[st:] \
                 else cond_toks[st:].index('and') + st
            print ('ed', ed)
            if 'NT' in cond_toks[st:ed] and 'EQL' in cond_toks[st:ed]:
                op_prev = cond_toks[st:ed].index('NT') + st
                op = op_prev + 1
                cur_cond[1] = 8
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
            else:
                op_prev = op = st
                cur_cond[1] = 0
            sel_col = cond_toks[st:op_prev]
            # logging.warning('sel_col: %s', str(sel_col))
            to_idx = [x.lower() for x in raw_col[b]]
            pred_col = self.merge_tokens(sel_col, raw_q[b] + ' || ' + \
                                    ' || '.join(raw_col[b]))
            # logging.warning('pred_col: %s', str(pred_col))
            if pred_col in to_idx:
                cur_cond[0] = to_idx.index(pred_col)
            else:
                cur_cond[0] = 0
            cur_cond[2] = self.merge_tokens(cond_toks[op+1:ed], raw_q[b])
            # logging.warning('cur_cond: {0}'.format(cur_cond))
            conds.append(cur_cond)
            st = ed + 1
        logging.debug('conds', conds)
        return  conds

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

        pred_agg, pred_sel, pred_cond = pred_entry

        B = len(gt_queries)

        tot_err = agg_err = sel_err = cond_err = cond_num_err = \
                  cond_col_err = cond_op_err = cond_val_err = 0.0
        agg_ops = ['None', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        for b, (pred_qry, gt_qry) in enumerate(zip(pred_queries, gt_queries)):
            good = True
            if pred_agg:
                agg_pred = [x for x in pred_qry['agg']]
                agg_gt = [x for x in gt_qry['agg']]
                logging.info('agg_pred: %s', str(agg_pred))
                logging.info('agg_gt: %s', str(agg_gt))
                flag = True
                if len(agg_pred) != len(agg_gt): # if length is off
                    flag = False
                    # agg_err += 1

                # if flag and set(
                #         agg_pred) != set(agg_gt):
                #     flag = False
                #     agg_col_err += 1
                agg_pred_sorted = agg_pred
                agg_gt_sorted = agg_gt
                for idx in range(len(agg_pred)):
                    if not flag:
                        break
                    if flag and agg_pred_sorted[idx] != agg_gt_sorted[idx]:
                        flag = False
                        # agg_err += 1

                if not flag:
                    agg_err += 1
                    good = False

            if pred_sel:
                sel_pred = pred_qry['sel']
                sel_gt = gt_qry['sel']
                logging.warning('sel_pred: %s', str(sel_pred))
                logging.warning('sel_gt: %s', str(sel_gt))
                flag = True
                if len(sel_pred) != len(sel_gt): # if length is off
                    flag = False
                    logging.warning('wrong number of columns selected')
                    #sel_err += 1
                sel_pred_sorted = sorted(sel_pred)
                sel_gt_sorted = sorted(sel_gt)
                for idx in range(len(sel_pred)):
                    if not flag:
                        break
                    if flag and sel_pred_sorted[idx] != sel_gt_sorted[idx]:
                        flag = False
                        logging.warning('wrong columns!')
                        #sel_err += 1

                if not flag:
                    sel_err += 1
                    logging.warning('incremented sel_err')
                    good = False

            if pred_cond:
                cond_pred = pred_qry['conds']
                logging.info('gt_qry[cond]: %s', str(gt_qry['cond']))
                for i, item in enumerate(gt_qry['cond']):
                    if len(item) > 3:
                        logging.info('item before: %s', str(item))
                        gt_qry['cond'][i] = item[1:]
                        logging.info('item after: %s', str(gt_qry['cond'][i]))
                cond_gt = gt_qry['cond']
                # logging.warning('cond_pred_qry: {0}'.format(cond_pred))
                # logging.warning('cond_gt_qry: {0}'.format(cond_gt))
                flag = True
                # print('cond_err', cond_err)
                if len(cond_pred) != len(cond_gt): # penalize if they predict the wrong number of conditions
                    flag = False
                    cond_num_err += 1
                    logging.warning('wrong number of columns')
                # print('cond_num_err', cond_num_err)

                if flag and set(
                        x[0] for x in cond_pred) != set(x[0] for x in cond_gt): # penalize if they predict the different columns
                    flag = False
                    cond_col_err += 1
                    logging.info('wrong column selected')
                # print('cond_col_err', cond_col_err)

                for idx in range(len(cond_pred)):
                    if not flag:
                        break
                    gt_idx = tuple(x[0] for x in cond_gt).index(cond_pred[idx][0])
                    # print ('gt_idx', gt_idx)
                    if flag and cond_gt[gt_idx][1] != cond_pred[idx][1]:
                        flag = False
                        cond_op_err += 1
                        logging.info('Wrong condition op')
                # print('cond_op_err', cond_op_err)

                for idx in range(len(cond_pred)):
                    if not flag:
                        break
                    gt_idx = tuple(x[0] for x in cond_gt).index(cond_pred[idx][0])
                    if flag and unicode(cond_gt[gt_idx][2]).lower() != \
                       unicode(cond_pred[idx][2]).lower():
                        flag = False
                        cond_val_err += 1
                        logging.info('Wrong values!!')
                # print('cond_val_err', cond_val_err)
                

                if not flag:
                    cond_err += 1
                    logging.warning('Error increased by one')     
                    good = False
                # cond_err += cond_val_err + cond_op_err + cond_col_err + cond_num_err
                # print('cond_err', cond_err)
 
            if not good:
                tot_err += 1
        logging.warning('tot_err: %d', tot_err)
        logging.warning('cond_err: {0}'.format(cond_err)) # should be 2! 
        return np.array((agg_err, sel_err, cond_err)), tot_err

    

           

    def gen_query(self, score, q, col, raw_q, raw_col, pred_entry,
                  reinforce=False, verbose=False):
        pred_agg, pred_sel, pred_cond = pred_entry
        sel_score, cond_score = score

        ret_queries = []
        # if pred_agg:
        #     B = len(agg_score)
        if pred_sel:
            B = len(sel_score)
        elif pred_cond:
            B = len(cond_score[0]) if reinforce else len(cond_score)
        for b in range(B):
            cur_query = {}
            # if pred_agg:
            #     cur_query['agg'] = np.argmax(agg_score[b].data.cpu().numpy())
                # print ('cur_query_agg', cur_query['agg'])
            if pred_sel:
                cur_query['sel'] = np.argmax(sel_score[b].data.cpu().numpy())
                cur_query['agg'], cur_query['sel'] =  self.gen_sel_query(col, sel_score, b, q, raw_col, raw_q)
                logging.debug('cur_query[sel]: %s', str(cur_query['sel']))
                logging.debug('cur_query[agg]: %s', str(cur_query['agg']))
            if pred_cond:
                cur_query['conds'] = self.gen_where_query(col, cond_score, b, q, raw_col, raw_q)
                logging.debug('cur_query_conds: %s', str(cur_query['conds']))
            # print('cur_query', cur_query)
            ret_queries.append(cur_query)

        return ret_queries
