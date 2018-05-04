import re
import io
import json
import numpy as np
import os
import logging
#from lib.dbengine import DBEngine

def lower_keys(x):
    if isinstance(x, list):
        return [lower_keys(v) for v in x]
    elif isinstance(x, dict):
        return dict((k.lower(), lower_keys(v)) for k, v in x.iteritems())
    else:
        return x

def get_main_table_name(file_path):
    prefix_pattern = re.compile('(processed/.*/)(.*)(_table\.json)')
    if prefix_pattern.search(file_path):
        return prefix_pattern.search(file_path).group(2)
    return None

def load_data_new(sql_paths, table_paths, use_small=False):
    if not isinstance(sql_paths, list):
        sql_paths = (sql_paths, )
    if not isinstance(table_paths, list):
        table_paths = (table_paths, )
    sql_data = []
    table_data = {}
    for i, SQL_PATH in enumerate(sql_paths):
        if use_small and i >= 2:
            break
        print "Loading data from %s"%SQL_PATH
        with open(SQL_PATH) as inf:
            data = lower_keys(json.load(inf))
            sql_data += data

    for i, TABLE_PATH in enumerate(table_paths):
        if use_small and i >= 2:
            break
        print "Loading data from %s"%TABLE_PATH
        with open(TABLE_PATH) as inf:
            table_data= json.load(inf)
    sql_data_new, table_data_new = process(sql_data, table_data)  # comment out if not on full dataset
    return sql_data_new, table_data_new

def epoch_reinforce_train(model, optimizer, batch_size, sql_data, table_data, db_path):
    engine = DBEngine(db_path)

    model.train()
    # perm = np.random.permutation(len(sql_data))
    perm = list(range(len(sql_data)))
    cum_reward = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data =\
            to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        gt_where_seq = model.generate_gt_where_seq(q_seq, col_seq, query_seq)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = model.generate_gt_sel_seq(q_seq, col_seq, query_seq)
        score = model.forward(q_seq, col_seq, col_num, (True, True, True, True),
                reinforce=True, gt_sel=gt_sel_seq)
        pred_queries = model.gen_query(score, q_seq, col_seq, raw_q_seq,
                raw_col_seq, (True, True, True, True), reinforce=True)

        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        rewards = []
        for idx, (sql_gt, sql_pred, tid) in enumerate(
                zip(query_gt, pred_queries, table_ids)):
            ret_gt = engine.execute(tid,
                    sql_gt['sel'], sql_gt['agg'], sql_gt['conds'])
            try:
                ret_pred = engine.execute(tid,
                        sql_pred['sel'], sql_pred['agg'], sql_pred['conds'])
            except:
                ret_pred = None

            if ret_pred is None:
                rewards.append(-2)
            elif ret_pred != ret_gt:
                rewards.append(-1)
            else:
                rewards.append(1)

        cum_reward += (sum(rewards))
        optimizer.zero_grad()
        model.reinforce_backward(score, rewards)
        optimizer.step()

        st = ed

    return cum_reward / len(sql_data)

def load_data(sql_paths, table_paths, use_small=False):
    if not isinstance(sql_paths, list):
        sql_paths = (sql_paths, )
    if not isinstance(table_paths, list):
        table_paths = (table_paths, )
    sql_data = []
    table_data = {}

    max_col_num = 0
    for SQL_PATH in sql_paths:
        print "Loading data from %s"%SQL_PATH
        with open(SQL_PATH) as inf:
            for idx, line in enumerate(inf):
                if use_small and idx >= 1000:
                    break
                print line.strip()
                sql = json.loads(line.strip())
                sql_data.append(sql)

    for TABLE_PATH in table_paths:
        print "Loading data from %s"%TABLE_PATH
        with open(TABLE_PATH) as inf:
            for line in inf:
                tab = json.loads(line.strip())
                table_data[tab[u'id']] = tab

    for sql in sql_data:
        assert sql[u'table_id'] in table_data

    return sql_data, table_data


def load_dataset(dataset_id, use_small=False):
    if dataset_id == 2:
        print "Loading from new dataset"
        # sql_data, table_data = load_data_new(['../alt/processed/train/art_1.json'],
        #          ['../alt/processed/tables/art_1_table.json'], use_small=use_small)
        # val_sql_data, val_table_data = load_data_new(['../alt/processed/train/art_1.json'],
        #          ['../alt/processed/tables/art_1_table.json'], use_small=use_small)

        # test_sql_data, test_table_data = load_data_new(['../alt/processed/train/art_1.json'],
        #          ['../alt/processed/tables/art_1_table.json'], use_small=use_small)

        sql_data, table_data = load_data_new(['New_Data/train.json'], 
                 ['New_Data/tables.json'], use_small=use_small)
        val_sql_data, val_table_data = load_data_new(['New_Data/dev.json'], 
                 ['New_Data/tables.json'], use_small=use_small)

        test_sql_data, test_table_data = load_data_new(['New_Data/train.json'], 
                 ['New_Data/tables.json'], use_small=use_small)

        TRAIN_DB = '../alt/data/train.db'
        DEV_DB = '../alt/data/dev.db'
        TEST_DB = '../alt/data/test.db'
    elif dataset_id == 0:
        print "Loading from original dataset"
        sql_data, table_data = load_data('../alt/data/train_tok.jsonl',
                 '../alt/data/train_tok.tables.jsonl', use_small=use_small)
        val_sql_data, val_table_data = load_data('../alt/data/dev_tok.jsonl',
                 '../alt/data/dev_tok.tables.jsonl', use_small=use_small)

        test_sql_data, test_table_data = load_data('../alt/data/test_tok.jsonl',
                '../alt/data/test_tok.tables.jsonl', use_small=use_small)
        TRAIN_DB = '../alt/data/train.db'
        DEV_DB = '../alt/data/dev.db'
        TEST_DB = '../alt/data/test.db'
    else:
        print "Loading from re-split dataset"
        sql_data, table_data = load_data('data_resplit/train.jsonl',
                'data_resplit/tables.jsonl', use_small=use_small)
        val_sql_data, val_table_data = load_data('data_resplit/dev.jsonl',
                'data_resplit/tables.jsonl', use_small=use_small)
        test_sql_data, test_table_data = load_data('data_resplit/test.jsonl',
                'data_resplit/tables.jsonl', use_small=use_small)
        TRAIN_DB = 'data_resplit/table.db'
        DEV_DB = 'data_resplit/table.db'
        TEST_DB = 'data_resplit/table.db'

    return sql_data, table_data, val_sql_data, val_table_data,\
            test_sql_data, test_table_data

def best_model_name(args, for_load=False):
    new_data = 'new' if args.dataset > 0 else 'old'
    mode = 'seq2sql' if args.baseline else 'sqlnet'
    if for_load:
        use_emb = use_rl = ''
    else:
        use_emb = '_train_emb' if args.train_emb else ''
        use_rl = 'rl_' if args.rl else ''
    use_ca = '_ca' if args.ca else ''

    agg_model_name = 'saved_model/%s_%s%s%s.agg_model'%(new_data,
            mode, use_emb, use_ca)
    sel_model_name = 'saved_model/%s_%s%s%s.sel_model'%(new_data,
            mode, use_emb, use_ca)
    cond_model_name = 'saved_model/%s_%s%s%s.cond_%smodel'%(new_data,
            mode, use_emb, use_ca, use_rl)
    groupby_model_name = 'saved_model/%s_%s%s%s.groupby_%smodel'%(new_data,
            mode, use_emb, use_ca, use_rl)

    if not for_load and args.train_emb:
        agg_embed_name = 'saved_model/%s_%s%s%s.agg_embed'%(new_data,
                mode, use_emb, use_ca)
        sel_embed_name = 'saved_model/%s_%s%s%s.sel_embed'%(new_data,
                mode, use_emb, use_ca)
        cond_embed_name = 'saved_model/%s_%s%s%s.cond_embed'%(new_data,
                mode, use_emb, use_ca)
        return agg_model_name, sel_model_name, cond_model_name,\
                agg_embed_name, sel_embed_name, cond_embed_name
    else:
        return agg_model_name, sel_model_name, cond_model_name, groupby_model_name


def to_batch_seq(sql_data, table_data, idxes, st, ed, ret_vis_data=False):
    q_seq = []
    col_seq = []
    col_num = []
    ans_seq = []
    query_seq = []
    gt_cond_seq = []
    vis_seq = []

    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        q_seq.append(sql['question_tok']) 
        table = table_data[sql['table_id']]
        col_num.append(len(table['col_map'])) 
        # tab_cols = [col[1] for col in table['col_map']]
        tab_cols = [col[1].split(' ') for col in table['col_map']]
        tab_col_indices = [(i, col[1].split(' ')) for i, col in enumerate(table['col_map'])]
        col_seq.append(tab_cols)

        ans_seq.append((sql['agg'], #index 0
            sql['sel'], # index 1
            len(sql['cond']), # index 2
            tuple(x[0] for x in sql['cond']),  # index 3
            tuple(x[1] for x in sql['cond']), # index 4
            len(set(sql['sel'])), # index 5       
            sql['group'][:-1], # index 6 = group         
            len(sql['group']) - 1,      
            sql['order'][0],            
            sql['order'][1],            
            len(sql['order'][1]),       
            sql['order'][2]             
            ))


        query_seq.append(sql['query_tok'])
        gt_cond_seq.append([x for x in sql['cond']])
        vis_seq.append((sql['question'], tab_col_indices, sql['query']))
    if ret_vis_data:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, vis_seq
    else:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq


def to_batch_query(sql_data, idxes, st, ed):
    query_gt = []
    table_ids = []
    for i in range(st, ed):
        # query_gt.append(sql_data[idxes[i]]['sql1'])
        # query_gt.append(sql_data[idxes[i]]['sql'])
        query_gt.append(sql_data[idxes[i]])
        table_ids.append(sql_data[idxes[i]]['table_id'])
    return query_gt, table_ids


def epoch_train_old(model, optimizer, batch_size, sql_data, table_data, pred_entry):
    model.train()
    perm=np.random.permutation(len(sql_data))
    cum_loss = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq = \
                to_batch_seq(sql_data, table_data, perm, st, ed)
        gt_where_seq = None#model.generate_gt_where_seq(q_seq, col_seq, query_seq)
        gt_sel_seq = [x[1] for x in ans_seq]
        gt_agg_seq = [x[0] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num, pred_entry,
                gt_where=gt_where_seq, gt_cond=gt_cond_seq, gt_sel=gt_sel_seq)
        
        # print 'score', score
        # print 'ans_seq', ans_seq
        # print 'pred_entry', pred_entry
        # print 'gt_where_seq', gt_where_seq
        # print 'loss about to called'
        loss = model.loss(score, ans_seq, pred_entry, gt_where_seq)
        cum_loss += loss.data.cpu().numpy()[0]*(ed - st)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        st = ed

    return cum_loss / len(sql_data)

def epoch_train(model, optimizer, batch_size, sql_data, table_data, pred_entry):
    logging.info('method epoch_train')
    model.train()
    # perm=np.random.permutation(len(sql_data))
    perm = list(range(len(sql_data)))
    cum_loss = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data = \
                to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        if len(q_seq) == 0:
            break
        # [(col_index, [col_tok_1, col_tok_2]), (col_index, [col_tok_1, col_tok_2])]
        # raw_col_seq = [x[1] for x in raw_data] 

        logging.warning('gt_cond_seq: {0}'.format(gt_cond_seq))
        gt_where_seq = model.generate_gt_where_seq(q_seq, col_seq, gt_cond_seq)

        # gt_where_seq = model.generate_gt_where_seq(q_seq, col_seq, query_seq)
        gt_sel_seq = None
        # gt_sel_seq = model.generate_gt_sel_seq(q_seq, col_seq, query_seq, ans_seq)
        gt_groupby_seq = None
        # gt_groupby_seq = model.generate_gt_group_seq(q_seq, col_seq, query_seq, ans_seq)
        score = model.forward(q_seq, col_seq, col_num, pred_entry,
                gt_where=gt_where_seq, gt_cond=gt_cond_seq, gt_sel=gt_sel_seq, gt_groupby=gt_groupby_seq)
        loss = model.loss(score, ans_seq, pred_entry, gt_where_seq, gt_sel_seq, gt_groupby_seq)
        cum_loss += loss.data.cpu().numpy()[0]*(ed - st)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        st = ed

    return cum_loss / len(sql_data)


def epoch_acc_new(model, batch_size, sql_data, table_data, pred_entry):
    logging.info('epoch_acc_new')
    model.eval()
    # print ('perm', perm)\
    # perm=np.random.permutation(len(sql_data))
    perm = list(range(len(sql_data)))
    # loggig.warning('len sql: %d', len(sql_data))
    st = 0
    one_acc = np.zeros((2,4)) # row 1 = acc_count, row 2 = tot_count
    tot_acc= np.zeros(2) #  1 = acc_count, 2 = tot_count
    tot_acc_count = 0.0
    acc_num_breakdown = 0.0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data = to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        # raw_data = (_, (col_index, [col_tok_1, col_tok_2]), _)
        if len(q_seq) == 0:
            break
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data] # [(col_index, [col_tok_1, col_tok_2]), (col_index, [col_tok_1, col_tok_2])]
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        logging.warning('query_gt: {0}'.format(json.dumps(query_gt)))
        score = model.forward(q_seq, col_seq, col_num,
                pred_entry)

        pred_queries = model.gen_query(score, q_seq, col_seq,
                raw_q_seq, raw_col_seq, pred_entry) # is this the decoder portion??
        
        logging.warning('pred_queries: {0}'.format(pred_queries))
        # pred_queries = model.gen_query(score, q_seq, col_seq,
        #         raw_q_seq, raw_col_seq, pred_entry, gt_cond = gt_cond_seq)
        # one_err, tot_err, err_breakdown = model.check_acc(raw_data,
        #         pred_queries, query_gt, pred_entry)
        ind_cor, tot_cor, _ = model.check_acc(raw_data, pred_queries, query_gt, pred_entry)


        # change to 
        '''
        (one_num_cor, total counted_arr),(tot_num_corr, total_counted), (corr_breakdown, that array repeated) = just keep track of the number correct
        '''

        # one_acc_num += (ed-st-one_err) # 5 - 0 - 2
        # tot_acc_num += (ed-st-tot_err)
        one_acc[0,:] += ind_cor[0,:] # total number correct
        one_acc[1,:] += ind_cor[1, :] # total number counted
        tot_acc[0] += tot_cor[0]
        tot_acc[1] += tot_cor[1]
        # # acc_num_breakdown += (ed-st-err_breakdown)
        # logging.debug('one_acc_num: {0}'.format(one_acc_num)) # should be 3
        # logging.debug('tot_acc_num: {0}'.format(tot_acc_num))
        # logging.warning('acc_num_breakdown: {0}'.format(acc_num_breakdown))
        # exit(1)

        st = ed

    total_acc_percent = np.divide(tot_acc[0], tot_acc[1], out=np.zeros_like(tot_acc[0]), where=tot_acc[1]!=0)
    one_acc_percent = np.divide(one_acc[0, :], one_acc[1,:], out=np.zeros_like(one_acc[0, :]), where=one_acc[1,:]!=0)
    logging.error('acc_percent: {0}, acc_percent_sp: {1}'.format(total_acc_percent, one_acc_percent))
    return total_acc_percent,  one_acc_percent, 0
    return tot_acc_num / len(sql_data), one_acc_num / len(sql_data), acc_num_breakdown / len(sql_data)


def epoch_acc(model, batch_size, sql_data, table_data, pred_entry, error_print=False, train_flag = False):
    model.eval()
    perm = list(range(len(sql_data)))
    st = 0
    one_acc_num = 0.0
    tot_acc_num = 0.0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq,\
         raw_data = to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        gt_ody_seq = [x[9] for x in ans_seq]
        if train_flag:
            score = model.forward(q_seq, col_seq, col_num, pred_entry, gt_sel=gt_sel_seq) #tmep for testing
        else:
            score = model.forward(q_seq, col_seq, col_num, pred_entry)
        pred_queries = model.gen_query(score, q_seq, col_seq,
                raw_q_seq, raw_col_seq, pred_entry, gt_sel = gt_sel_seq, gt_ody = gt_ody_seq) 
        one_err, tot_err = model.check_acc(raw_data, pred_queries, query_gt, pred_entry, error_print)

        one_acc_num += (ed-st-one_err)
        tot_acc_num += (ed-st-tot_err)

        st = ed
    return tot_acc_num / len(sql_data), one_acc_num / len(sql_data)

def epoch_exec_acc(model, batch_size, sql_data, table_data, db_path):
    engine = DBEngine(db_path)
    print 'exec acc'

    model.eval()
    perm = list(range(len(sql_data)))
    tot_acc_num = 0.0
    acc_of_log = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data = \
            to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        model.generate_gt_sel_seq(q_seq, col_seq, query_seq)
        gt_where_seq = model.generate_gt_where_seq(q_seq, col_seq, query_seq)
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        # print 'gt_sel_seq', gt_sel_seq
        score = model.forward(q_seq, col_seq, col_num,
                (True, True, True), gt_sel=gt_sel_seq)
        pred_queries = model.gen_query(score, q_seq, col_seq,
                raw_q_seq, raw_col_seq, (True, True, True))

        for idx, (sql_gt, sql_pred, tid) in enumerate(
                zip(query_gt, pred_queries, table_ids)):
            ret_gt = engine.execute(tid,
                    sql_gt['sel'], sql_gt['agg'], sql_gt['conds'])
            try:
                ret_pred = engine.execute(tid,
                        sql_pred['sel'], sql_pred['agg'], sql_pred['conds'])
            except:
                ret_pred = None
            tot_acc_num += (ret_gt == ret_pred)
        
        st = ed

    return tot_acc_num / len(sql_data)

def load_para_wemb(file_name):
    f = io.open(file_name, 'r', encoding='utf-8')
    lines = f.readlines()
    ret = {}
    if len(lines[0].split()) == 2:
        lines.pop(0)
    for (n,line) in enumerate(lines):
        info = line.strip().split(' ')
        if info[0].lower() not in ret:
            ret[info[0]] = np.array(map(lambda x:float(x), info[1:]))

    return ret


def load_comb_wemb(fn1, fn2):
    wemb1 = load_word_emb(fn1)
    wemb2 = load_para_wemb(fn2)
    comb_emb = {k: wemb1.get(k, 0) + wemb2.get(k, 0) for k in set(wemb1) | set(wemb2)}

    return comb_emb


def load_concat_wemb(fn1, fn2):
    wemb1 = load_word_emb(fn1)
    wemb2 = load_para_wemb(fn2)
    backup = np.zeros(300, dtype=np.float32)
    comb_emb = {k: np.concatenate((wemb1.get(k, backup), wemb2.get(k, backup)), axis=0) for k in set(wemb1) | set(wemb2)}

    return comb_emb


def load_word_emb(file_name, load_used=False, use_small=False):
    if not load_used:
        print ('Loading word embedding from %s'%file_name)
        ret = {}
        with open(file_name) as inf:
            for idx, line in enumerate(inf):
                if (use_small and idx >= 5000):
                    break
                info = line.strip().split(' ')
                if info[0].lower() not in ret:
                    ret[info[0]] = np.array(map(lambda x:float(x), info[1:]))
        return ret
    else:
        print ('Load used word embedding')
        with open('glove/word2idx.json') as inf:
            w2i = json.load(inf)
        with open('glove/usedwordemb.npy') as inf:
            word_emb_val = np.load(inf)
        return w2i, word_emb_val

def process(sql_data, table_data):
    output_tab = {}
    for i in range(len(table_data)):
        table = table_data[i]
        temp = {}
        temp['col_map'] = table['column_names']

        db_name = table['db_id']
        # print table
        output_tab[db_name] = temp

    output_sql = []
    for i in range(len(sql_data)):
        sql = sql_data[i]
        sql_temp = {}

        # add query metadata
        sql_temp['question'] = sql['question']
        sql_temp['question_tok'] = sql['question_toks']
        sql_temp['query'] = sql['query']
        sql_temp['query_tok'] = sql['query_toks']
        sql_temp['table_id'] = sql['db_id']

        # process agg/sel
        sql_temp['agg'] = []
        sql_temp['sel'] = []
        gt_sel = sql['sql']['select'][1]
        if len(gt_sel) > 4:
            gt_sel = gt_sel[:4]
        for tup in gt_sel:
            sql_temp['agg'].append(tup[0])
            sql_temp['sel'].append(tup[1][1][1])
        
        # process where conditions and conjuctions
        sql_temp['cond'] = []
        gt_cond = sql['sql']['where']
        if len(gt_cond) > 0:
            conds = [gt_cond[x] for x in range(len(gt_cond)) if x % 2 == 0]
            for cond in conds:
                curr_cond = []
                curr_cond.append(cond[2][1][1])
                curr_cond.append(cond[1])

                if cond[4] is not None:
                    curr_cond.append([cond[3], cond[4]])
                else:
                    curr_cond.append(cond[3])
                sql_temp['cond'].append(curr_cond)

        sql_temp['conj'] = [gt_cond[x] for x in range(len(gt_cond)) if x % 2 == 1]
        # logging.warning('process')
        # logging.warning('sql: {0}'.format(json.dumps(gt_cond, indent=4)))
        # logging.warning('sql_temp: {0}'.format(json.dumps(sql_temp, indent=4)))

        # process group by / having
        sql_temp['group'] = [x[1] for x in sql['sql']['groupby']]
        having_cond = []
        if len(sql['sql']['having']) > 0:
            
            gt_having = sql['sql']['having'][0] 
            having_cond.append(gt_having[2][1][0]) # aggregator
            having_cond.append(gt_having[2][1][1]) # column
            having_cond.append(gt_having[1]) # operator
            if gt_having[4] is not None:
                having_cond.append([gt_having[3], gt_having[4]])
            else:
                having_cond.append(gt_having[3])
        sql_temp['group'].append(having_cond)

        # process order by / limit
        order_aggs = []
        order_cols = []
        order_par = -1
        gt_order = sql['sql']['orderby']
        if len(gt_order) > 0:
            order_aggs = [x[1][0] for x in gt_order[1]]
            order_cols = [x[1][1] for x in gt_order[1]]
            if gt_order[0] == 'asc':
                order_par = 1 
            else:
                order_par = 0
        sql_temp['order'] = [order_aggs, order_cols, order_par]

        # process intersect/except/union
        sql_temp['special'] = 0
        if sql['sql']['intersect']:
            sql_temp['special'] = 1
        elif sql['sql']['except'] is not None:
            sql_temp['special'] = 2
        elif sql['sql']['union'] is not None:
            sql_temp['special'] = 3

        output_sql.append(sql_temp)
    return output_sql, output_tab
