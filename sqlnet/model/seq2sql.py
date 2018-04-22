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
        self.AGG_SQL_TOK = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        self.SEL_SQL_TOK = ['<UNK>', '<END>', ',', 'SEL'] # w
        self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', # what is the <UNK> stand for???
                        'EQL', 'GT', 'LT', '<BEG>'] # EQL = , GT > , LT <
        self.COND_OPS = ['EQL', 'GT', 'LT']

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

    def generate_gt_sel_seq(self, q, col, query):
        print('q', q)
        print('col', col)
        print('query', query)
        exit(1)

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
            cur_seq = [all_toks.index('<BEG>')]
            if 'where' in cur_query:
                cur_where_query = cur_query[cur_query.index('where'):]
                cur_seq = cur_seq + map(lambda tok:all_toks.index(tok)
                                        if tok in all_toks else 0, cur_where_query)
            cur_seq.append(all_toks.index('<END>'))
            ret_seq.append(cur_seq)
        return ret_seq


    def forward(self, q, col, col_num, pred_entry,
                gt_where = None, gt_cond=None, reinforce=False, gt_sel=None):
        B = len(q)
        pred_sel, pred_cond = pred_entry

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
                print 'pred_cond'
                print('gt_where', gt_where)
                print('gt_cond', gt_cond)
                cond_score = self.cond_pred(x_emb_var, x_len, col_inp_var,
                                            col_name_len, col_len, col_num,
                                            gt_where, gt_cond,
                                            reinforce=reinforce)

            # if pred_agg:
            #     print 'agg_pred'
            #     agg_score = self.agg_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num)


            if pred_sel:
                print ('gt_sel', gt_sel)
                sel_score = self.sel_pred(x_emb_var, x_len, col_inp_var,
                                          col_name_len, col_len, col_num)

            

        return (sel_score, cond_score)

    def loss(self, score, truth_num, pred_entry, gt_where):
        pred_sel, pred_cond = pred_entry
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
            sel_truth = map(lambda x:x[1], truth_num)
            print('sel_truth', sel_truth)
            print('sel_score.size()', sel_score.size())
            for b in range(len(sel_truth)):
                if self.gpu:
                    sel_truth_var = Variable(
                        torch.from_numpy(np.array(sel_truth[b])).cuda())
                else:
                    sel_truth_var = Variable(
                        torch.from_numpy(np.array(sel_truth[b])))
                sel_pred_score = sel_score[b, :len(sel_truth[b])]
                # print('sel_truth[b]', sel_truth[b])
                # print('sel_pred_score.size()', sel_pred_score.size())
                # print('sel_truth_var.size()', sel_truth_var.size())
                self.CE(sel_pred_score, sel_truth_var)
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

    def gen_sel_query(self, col, sel_score, b, q, raw_col, raw_q, verbose=True):
        selects = []

        all_toks = self.SEL_SQL_TOK + self.AGG_SQL_TOK + \
                   [x for toks in col[b] for x in
                    toks+[',']] + [''] + q[b] + [''] # should I delete q --> issue is that I get the wrong dimensions, should I change the embeddings??

        print('all_toks_len', len(all_toks))
        sel_toks = []

        print('sel_score.size()', sel_score.size())
        print ('sel_score[b].size()', sel_score[b].size())
        print ('sel_score', sel_score[b].data.cpu().numpy())
        for sel_score_idx in sel_score[b].data.cpu().numpy():
            sel_tok = np.argmax(sel_score_idx)
            print('sel_tok', sel_tok)
            # print ('sel_tok', sel_tok)
            # print ('max', np.max(sel_score_idx))
            sel_val = all_toks[sel_tok]
            print ('sel_val', sel_val)
            if sel_val == '<END>':
                print 'found EXIT!!!'
                # exit(1)
                break
            sel_toks.append(sel_tok)
        # print ('q[b]', q[b])
        # print ('sel_toks', sel_toks)
        if verbose:
            print sel_toks
        if len(sel_toks) > 0:
            sel_toks = sel_toks[1:]
        return  sel_toks


    def gen_where_query(self, col, cond_score, b, q, raw_col, raw_q, verbose=True):
        conds = []
        all_toks = self.SQL_TOK + \
                   [x for toks in col[b] for x in
                    toks+[',']] + [''] + q[b] + ['']
        cond_toks = []

        # print 'making where'
        # print 'raw_col', raw_col
        for where_score in cond_score[b].data.cpu().numpy():
            cond_tok = np.argmax(where_score)
            cond_val = all_toks[cond_tok]
            if cond_val == '<END>':
                break
            cond_toks.append(cond_val)

        if verbose:
            print cond_toks
        if len(cond_toks) > 0:
            cond_toks = cond_toks[1:]
        st = 0
        # print 'cond_toks', cond_toks
        while st < len(cond_toks):
            # print ('st', st)
            cur_cond = [None, None, None]
            ed = len(cond_toks) if 'and' not in cond_toks[st:] \
                 else cond_toks[st:].index('and') + st
            # print ('ed', ed)
            if '=' in cond_toks[st:ed]:
                op = cond_toks[st:ed].index('=') + st
                cur_cond[1] = 0
            elif '>' in cond_toks[st:ed]:
                op = cond_toks[st:ed].index('>') + st
                cur_cond[1] = 1
            elif '<' in cond_toks[st:ed]:
                op = cond_toks[st:ed].index('<') + st
                cur_cond[1] = 2
            else:
                op = st
                cur_cond[1] = 0
            sel_col = cond_toks[st:op]
            # print ('sel_col', sel_col)
            to_idx = [x.lower() for x in raw_col[b]]
            pred_col = self.merge_tokens(sel_col, raw_q[b] + ' || ' + \
                                    ' || '.join(raw_col[b]))
            # print ('pred_col', pred_col)
            if pred_col in to_idx:
                cur_cond[0] = to_idx.index(pred_col)
            else:
                cur_cond[0] = 0
            cur_cond[2] = self.merge_tokens(cond_toks[op+1:ed], raw_q[b])
            conds.append(cur_cond)
            st = ed + 1
        print ('conds', conds)
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

        pred_sel, pred_cond = pred_entry

        B = len(gt_queries)

        tot_err = agg_err = sel_err = cond_err = cond_num_err = \
                  cond_col_err = cond_op_err = cond_val_err = 0.0
        agg_ops = ['None', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        for b, (pred_qry, gt_qry) in enumerate(zip(pred_queries, gt_queries)):
            good = True
            print 'pred_qry', pred_qry
            print 'gt_qry', gt_qry
            # if pred_agg:
            #     agg_pred = pred_qry['agg']
            #     agg_gt = gt_qry['agg']
            #     if [agg_pred] != agg_gt:
            #         agg_err += 1
            #         good = False

            if pred_sel:
                sel_pred = pred_qry['sel']
                sel_gt = gt_qry['sel']
                if [sel_pred] != sel_gt:
                    sel_err += 1
                    good = False

            if pred_cond:
                print 'pred_qry', json.dumps(pred_qry, indent=4)
                print 'gt_qry', json.dumps(gt_qry, indent=4)
                cond_pred = pred_qry['conds']
                cond_gt = gt_qry['cond']
                flag = True
                if len(cond_pred) != len(cond_gt):
                    flag = False
                    cond_num_err += 1

                if flag and set(
                        x[0] for x in cond_pred) != set(x[0] for x in cond_gt):
                    flag = False
                    cond_col_err += 1

                for idx in range(len(cond_pred)):
                    if not flag:
                        break
                    gt_idx = tuple(x[0] for x in cond_gt).index(cond_pred[idx][0])
                    if flag and cond_gt[gt_idx][1] != cond_pred[idx][1]:
                        flag = False
                        cond_op_err += 1

                for idx in range(len(cond_pred)):
                    if not flag:
                        break
                    gt_idx = tuple(x[0] for x in cond_gt).index(cond_pred[idx][0])
                    if flag and unicode(cond_gt[gt_idx][2]).lower() != \
                       unicode(cond_pred[idx][2]).lower():
                        flag = False
                        cond_val_err += 1

                if not flag:
                    cond_err += 1
                    good = False

            if not good:
                tot_err += 1

        return np.array((sel_err, cond_err)), tot_err

    

           

    def gen_query(self, score, q, col, raw_q, raw_col, pred_entry,
                  reinforce=False, verbose=False):
        pred_sel, pred_cond = pred_entry
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
                cur_query['sel'] =  self.gen_sel_query(col, cond_score, b, q, raw_col, raw_q)
            if pred_cond:
                cur_query['conds'] = self.gen_where_query(col, cond_score, b, q, raw_col, raw_q)
                # print ('cur_query_conds', cur_query['conds'])
            print('cur_query', cur_query)
            ret_queries.append(cur_query)

        return ret_queries
