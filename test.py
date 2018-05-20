import json
import torch
from sqlnet.utils_new import *
from sqlnet.model.seq2sql import Seq2SQL
from sqlnet.model.sqlnet import SQLNet
import numpy as np
import datetime

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', action='store_true', 
            help='If set, use small data; used for fast debugging.')
    parser.add_argument('--ca', action='store_true',
            help='Use conditional attention.')
    parser.add_argument('--dataset', type=int, default=0,
            help='0: original dataset, 1: re-split dataset')
    parser.add_argument('--rl', action='store_true',
            help='Use RL for Seq2SQL.')
    parser.add_argument('--baseline', action='store_true', 
            help='If set, then test Seq2SQL model; default is SQLNet model.')
    parser.add_argument('--train_emb', action='store_true',
            help='Use trained word embedding for SQLNet.')
    args = parser.parse_args()

    N_word=300
    B_word=42
    if args.toy:
        USE_SMALL=True
        GPU=True
        BATCH_SIZE=15
    else:
        USE_SMALL=False
        GPU=True
        BATCH_SIZE=64
    TRAIN_ENTRY=(True, True, True, True, True)
    logging.error('about to load dataset')
    sql_data, table_data, val_sql_data, val_table_data, test_sql_data, test_table_data = load_dataset(args.dataset, use_small=USE_SMALL)
    logging.error('loadeded dataset')
    if args.toy:       
        sql_data = sql_data[0:300]
        val_sql_data = val_sql_data[0:300]

    word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), \
        load_used=False, use_small=USE_SMALL) # load_used can speed up loading

    if args.baseline:
        model = Seq2SQL(word_emb, N_word=N_word, gpu=GPU, trainable_emb = False)
    else:
        model = SQLNet(word_emb, N_word=N_word, use_ca=args.ca, gpu=GPU,
                trainable_emb = True)
    logging.error('initialized the model')
    if args.train_emb:
        agg_m, sel_m, cond_m, agg_e, sel_e, cond_e = best_model_name(args)
        print "Loading from %s"%agg_m
        model.agg_pred.load_state_dict(torch.load(agg_m))
        print "Loading from %s"%sel_m
        model.sel_pred.load_state_dict(torch.load(sel_m))
        print "Loading from %s"%cond_m
        model.cond_pred.load_state_dict(torch.load(cond_m))
        print "Loading from %s"%agg_e
        model.agg_embed_layer.load_state_dict(torch.load(agg_e))
        print "Loading from %s"%sel_e
        model.sel_embed_layer.load_state_dict(torch.load(sel_e))
        print "Loading from %s"%cond_e
        model.cond_embed_layer.load_state_dict(torch.load(cond_e))
    else:
        agg_m, sel_m, cond_m, groupby_m, orderby_m = best_model_name(args)
        print "Loading from %s"%agg_m
        model.agg_pred.load_state_dict(torch.load(agg_m))
        print "Loading from %s"%sel_m
        model.sel_pred.load_state_dict(torch.load(sel_m))
        print "Loading from %s"%cond_m
        model.cond_pred.load_state_dict(torch.load(cond_m))
        print "Loading from %s"%groupby_m
        model.groupby_pred.load_state_dict(torch.load(groupby_m))
        print "Loading from %s"%orderby_m
        model.orderby_pred.load_state_dict(torch.load(orderby_m))

    (val_acc_tot, val_acc_indiv),  sql_queries = epoch_acc_new(model, BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY, train=False, generate_SQL_query = True, test_file=True)
    logging.error(' Dev acc_qm: %s\n   breakdown result: %s\n'%(val_acc_tot, val_acc_indiv))

    with open('predicted_test.sql', 'wb') as file:
        for sql_query in sql_queries:
            file.write(sql_query + '\n')
    print('\a')
