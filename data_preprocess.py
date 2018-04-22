# -*- coding: utf-8 -*-
import json
from sqlnet.lib.dbengine import DBEngine
import re
import numpy as np
from os import listdir
import re
import sys
from nltk import word_tokenize
from enum import Enum

PATH_NL2SQL = 'New_Data/Initial'
TRAIN_EXT = '/train'
DEV_EXT = '/dev'
TABLE_EXT = '/tables'
extra_sql_keywords = ['where', 'group', 'order', 'limit', 'intersect', 'union', 'except']
where_ops = ['not', 'between', '=', '>', '<', '>=', '<=', 'in', 'like', 'is']

class agg(Enum):
    none = 0
    mx = 1
    mn = 2
    count = 3
    sm = 4
    avg = 5

def lower_keys(x):
    if isinstance(x, list):
        return [lower_keys(v) for v in x]
    elif isinstance(x, dict):
        return dict((k.lower(), lower_keys(v)) for k, v in x.iteritems())
    else:
        return x

def get_main_file_name(file_path):
    prefix_pattern = re.compile('(New_Data/Initial/.*/)(.*)(\.json)')
    if prefix_pattern.search(file_path):
        return prefix_pattern.search(file_path).group(2)
    return None

def get_main_table_name(file_path):
    prefix_pattern = re.compile('(New_Data/Initial/.*/)(.*)(_table\.json)')
    if prefix_pattern.search(file_path):
        return prefix_pattern.search(file_path).group(2)
    return None

def get_tables_for_sql(orig_path, train=0):
    TABLE_PATH = orig_path + TABLE_EXT
    tables = [TABLE_PATH + '/' + file for file in listdir(TABLE_PATH)]
    table_names  = [get_main_table_name(file) for file in tables]

    if train == 0:
        SQL_PATH = PATH_NL2SQL + TRAIN_EXT
    else:
        SQL_PATH = PATH_NL2SQL + DEV_EXT
    sql_data = [get_main_file_name(SQL_PATH + '/' + file) for file in listdir(SQL_PATH)]
    table_data = [TABLE_PATH + '/' + file  + '_table.json' for file in table_names if file in sql_data]
    sql_data = [SQL_PATH + '/' + file + '.json'for file in sql_data]

    return sorted(sql_data), sorted(table_data) 

def safe_list_get(lst, index, default=None):
    try:
        return lst[index]
    except IndexError:
        return default

def get_table_names_from_query(query_tok, select=False):
    table_names = []
    if select:
        indices = [i + 1 for i, x in enumerate(query_tok) if x.lower() == "from" ]
    else:
        indices = [i + 1 for i, x in enumerate(query_tok) if x.lower() == "from" or x.lower() == 'join']
    table_names = [query_tok[index] for index in indices]
    return list(set(table_names))

def clean_table_data(json_data, use_small=False):
    '''
    schema:
    database_name {
    header: [list of the column names for every table]
    header_tok: [[each column name tokenized]]
    table_name: [for each column name, what is the name of the table?]
    table_name_tok: [tokenize each table_name]
    foreign_key: [(column_1, column_2), (,), ...]
    }
    '''
    table_data = {}
    for database_name in json_data.keys():
        database = {}
        # database['rows'] = []
        # database['page_title'] = ''
        # database['section_title'] = ''
        database['table_name'] = []
        database['table_name_tok'] = []
        database['foreign_key'] = [] # info not available yet
        
        header = []
        types = []
        table_names = []
        for table in json_data[database_name]:
            for col in table['col_data']:
                header.append(col['column_name'].lower())
                types.append(col['data_type'].lower())
                table_names.append(table['table'].lower())
        database['header'] = header
        database['types'] = types 
        database['header_tok'] = [word_tokenize(col) for col in database['header']]
        database['table_name'] = table_names
        database['table_name_tok'] = [word_tokenize(table) for table in database['table_name']]
        table_data[database_name] = database
    # print 'table data: \n', json.dumps(table_data, indent=4)
    return table_data

def convert_colnames_colnum(cleaned_data_item, table_data, col_names, table_name, select=False):
    # names of columns --> column_numbers!
    database_name = cleaned_data_item['table_id']

    column_numbers = []
    if '*' in col_names: # get all the column indices
        return range(len(table_data[database_name]['header'])) # number of columns in the database
    for col in col_names:
        # get indices that match the name
        indices = [i for i, x in enumerate(table_data[database_name]['header']) if x == col]
        for index in indices:
            # get correct table
            if table_data[database_name]['table_name'][index] == table_name:
                column_numbers.append(index)
                break
    return column_numbers

def get_select_indices(cleaned_data_item, table_data): #TODO: do for all table names
    query_tok = cleaned_data_item['query_tok']
    print query_tok
    orig_col_names = query_tok[query_tok.index('select') + 1:query_tok.index('from')]
    sql_keywords = ['where', 'group', 'order', 'limit', 'intersect', 'union', 'except']
    key_ind = None
    for keyword in sql_keywords:
        try:
            key_ind = query_tok.index(keyword)
            break
        except ValueError:
            pass

    if key_ind is not None:
        table_name =  query_tok[query_tok.index('from') + 1:key_ind]

    table_name =  query_tok[query_tok.index('from') + 1:] # needs to take all table names, otherwise fails on certain joins
    to_delete = ['max', 'min', 'count', 'sum', 'avg', 'distinct', ',', '(', ')']
    for item in to_delete:
        try:
            orig_col_names.remove(item)
        except ValueError:
            pass
    pattern = re.compile('(t\d+\.)(.*)')
    col_names = orig_col_names
    for i, item in enumerate(col_names):
        if pattern.search(item) is not None:
            col_names[i] = pattern.search(item).group(2)
    # names of columns --> column_numbers!
    return orig_col_names, convert_colnames_colnum(cleaned_data_item, table_data, col_names, table_name, select=True)

def get_agg_codes(query, col_names, star=True):
    agg_ops = ['(max)\((.+?)\)', '(min)\((.+?)\)', '(count)\((.+?)\)', '(sum)\((.+?)\)', '(avg)\((.+?)\)']

    new_col = [s.lower() for s in query.split(' ')]
    new_col = new_col[new_col.index('select') + 1: new_col.index('from')]
    to_delete = ['distinct', ',', '(', ')', '', ' ']
    for item in to_delete:
        try:
            new_col.remove(item)
        except ValueError:
            pass
    agg_codes = [0] * len(new_col)    
    for x, agg in enumerate(agg_ops):
        pattern = re.compile(agg)
        for i, col in enumerate(new_col):
            if pattern.search(query) is not None:
                agg_codes[i] = x + 1
    return agg_codes


#NOTE: should be only the table_data that is relevant to this question
def add_sql_item_to_data(cleaned_data_item, table_data):
    '''sql_format
    sql : {
        agg: [0, 1, 0, ... , codes of each select column]
        sel: [indices of select columns]
        cond: [[column_index, op_code, value!! will be recursive]]
        from: [...? what is here now]
    }
    '''
    # print cleaned_data_item
    query = cleaned_data_item['query']
    col_names, select_indices = get_select_indices(cleaned_data_item, table_data)
    star = False
    if '*' in col_names:
        col_names = table_data[cleaned_data_item['table_id']]['header']
        star = True
    agg_codes = get_agg_codes(query, col_names)

    #get conds
    conds = []
    query_tok = cleaned_data_item['query_tok'] + ['<TEMP>']
    if 'where' in query_tok and 'not' not in query_tok: #temporarily disregard not 
        where_clause = query_tok[query_tok.index('where') + 1:query_tok.index('<TEMP>')]
        and_inds = [ind for ind, tok in enumerate(where_clause) if tok == 'and' or tok == 'or']
        st_inds = [0] + [ind + 1 for ind in and_inds]
        end_inds = [ind for ind in and_inds] + [len(where_clause)]
        where_ops = ['=', '>', '<', '>=', '<=', 'in'] #worry about not in later

        pos = 0
        for (st, ed) in zip(st_inds, end_inds):
            curr = [0, 0, 0]
            curr_cond = where_clause[st:ed]
            # print curr_cond

            op_ind = None
            for i, op in enumerate(where_ops):
                try:
                    temp_ind = curr_cond.index(op)
                except ValueError:
                    temp_ind = None
                if temp_ind is not None:
                    op_ind = curr_cond.index(op)
                    curr[1] = i
                    # pos += st + op_ind 

            col = where_clause[st:op_ind] #hopefully just 1 token
            for c in col:
                if '.' in c:
                    per_ind = c.find('.')####
                    # print c
                    c = c[per_ind + 1:]
                    # print c

                if c in table_data[cleaned_data_item['table_id']]['header']:
                    curr[0] = table_data[cleaned_data_item['table_id']]['header'].index(c) + 1
                    break

            val = curr_cond[op_ind + 1:] #hopefully just 1 token
            # print 'val', val
            curr[2] = ''.join(val)

            conds += [curr]
            # print conds

    # need group by/having, order by, limit, from/join, etc


    # print agg_codes
    # print col_names, select_indices
    cleaned_data_item['sql'] = {'agg': agg_codes[0], 'sel': select_indices[0], 'conds': conds}


def clean_sql_data(json_data, table_data, use_small=False):
    #NOTE: should only be called after clean_table_data
    '''
    sql_data_json_schema: 
    [{ 
        'question' : ["question_1", "question_2", "question_3"]
        'query' : ['sql translation 1', ...]
        'query_tok' : [['tokenize each query']
        'table_id' : 'database_name'
        'question_tok':[['tokenize each question']]
            
    }, {list of sql queries}]
    '''
    sql_data = []
    count = 0
    exit = False
    for database in json_data:
        for item in database['data']:
            if count > 50 and use_small:
                exit = True
                break
            cleaned_data = {}
            cleaned_data['table_id'] = database['database_name']
            cleaned_data['question']= item['sqa']['question'][0] # get first question
            cleaned_data['query']= item['sqa']['sql'][0] # get first query
            cleaned_data['query_tok'] = word_tokenize(cleaned_data['query'].lower())
            #cleaned_data['table_ids'] = [get_table_names_from_query(query_tok,select=False) for query_tok in cleaned_data['query_tok']]
            #cleaned_data['selected_table'] = [get_table_names_from_query(query_tok, select=True) for query_tok in cleaned_data['query_tok']]
            cleaned_data['question_tok'] = word_tokenize(cleaned_data['question'].lower())
            
            sql_data.append(cleaned_data)
            print 'hello', item['sqa']['sql'][0]
            add_sql_item_to_data(cleaned_data, table_data)
            count += 1
        if exit:
            break  
    # for item in sql_data:
    #     add_sql_item_to_data(item, table_data) 
    #     specific_table_data = {} 
    #     for table in item['table_ids'][0]:
    #         specific_table_data[table] = table_data[item['database_name']][table]
    
    # print json.dumps(sql_data, indent=4)
    return sql_data
        #_ed_data['query_2'] = database['data']['sqa']['sql'][1]
        #eaned_data['query_3'] = safe_list_get(database['data']['sqa']['sql'], 2)


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
            file_name = get_main_file_name(SQL_PATH)
            if file_name:
                sql_data.append(lower_keys(json.load(inf)))
                sql_data[i]['database_name'] = file_name
                
    for i, TABLE_PATH in enumerate(table_paths):
        if use_small and i >= 2:
            break
        print "Loading data from %s"%TABLE_PATH
        with open(TABLE_PATH) as inf:
            file_name = get_main_table_name(TABLE_PATH)
            if file_name:
                table_data[file_name] = lower_keys(json.load(inf))
    # print sql_data, table_data
    return sql_data, table_data

def load_dataset_new(dataset_id, use_small=False):
    print "Loading from nl2sql Yale dataset"
    # Load training data  
    sql_paths_train, table_paths_train  = get_tables_for_sql(PATH_NL2SQL, train=0)
    train_sql_data, train_table_data = load_data_new(sql_paths_train, table_paths_train, \
        use_small=use_small)
    print train_table_data
    train_table_data = clean_table_data(train_table_data, use_small=use_small)
    train_sql_data = clean_sql_data(train_sql_data, train_table_data, use_small=use_small)
    # Load Dev Data
    sql_paths_val, table_paths_val  = get_tables_for_sql(PATH_NL2SQL, train=1)
    val_sql_data, val_table_data = load_data_new(sql_paths_val, table_paths_val, \
        use_small=use_small)
    val_table_data = clean_table_data(val_table_data, use_small=use_small)
    val_sql_data = clean_sql_data(val_sql_data, val_table_data, use_small=use_small)
    
    ##TODO
    test_sql_data, test_table_data = {}, {}
    return train_sql_data, train_table_data, val_sql_data, val_table_data,\
            test_sql_data, test_table_data

def load_processed_dataset(use_small=False):
    sql_paths_train, table_paths_train  = get_tables_for_sql(PATH_NL2SQL, train=0)
    print 'sql_paths_train', sql_paths_train
    train_sql_data, train_table_data = load_data_new(sql_paths_train, table_paths_train, \
        use_small=False)
    schema = {}

def table_create():
    print "Loading table data from nl2sql Yale dataset"
    # Load training data  
    sql_paths_train, table_paths_train  = get_tables_for_sql(PATH_NL2SQL, train=0)
    print 'sql_paths_train', sql_paths_train
    train_sql_data, train_table_data = load_data_new(sql_paths_train, table_paths_train, \
        use_small=False)
    schema = {}

    for db_name in train_table_data.keys():
        db = {}
        index_to_name = [(0, '*')]
        col_counter = 1 #reserve the 0 index for *
        table_inds = []

        for i, table_dict in enumerate(train_table_data[db_name]):
            table = {}
            table_name = table_dict['table'].lower()
            table_inds.append(table_name)

            for ind, column in enumerate(table_dict['col_data'], col_counter):
                col_dict = {}
                col_dict['name'] = column['column_name'].lower()
                col_dict['name_tok'] = word_tokenize(column['column_name'].lower())
                col_dict['foreign_key'] = None
                table[ind] = col_dict

                col_counter += 1
                index_to_name.append((ind, column['column_name'].lower(), table_name))

            db[table_name] = table

        db['tab_map'] = table_inds # list with table names corresponding to each index
        db['col_map'] = index_to_name # list with index, column name, and corresponding table for all columns in db
        schema[db_name] = db
    for db in schema.keys():
        # print db
        # print schema[db]
        with open('New_Data/Initial/processed/tables/' + db + '_processed.json', 'w') as fp:
            json.dump(schema[db], fp, indent=4)
    # sys.sleep()
    
    ##### sql processing
    for database in train_sql_data:
        curr_db = schema[database['database_name']]

        sql_data = []
        for item in database['data']:
            cleaned_data = {}

            dbn = database['database_name']
            cleaned_data['table_id'] = dbn
            cleaned_data['question'] = item['question']# item['sqa']['question'][0] # get first question
            cleaned_data['query'] = item['sql'][0] #item['sqa']['sql'][0] # get first query
            cleaned_data['query_tok'] = word_tokenize(cleaned_data['query'].lower())
            cleaned_data['question_tok'] = word_tokenize(cleaned_data['question'].lower())

            # currently not handling case of expressions in columns (e.g. endyear - startyear)
            if '-' in cleaned_data['query_tok']:
                print 'minus', cleaned_data['query_tok']
                continue

            # currently not handling case of t1.*
            if 'T1.*' in cleaned_data['query']:
                print 'hello', cleaned_data['query']
                continue

            # print cleaned_data['query']

            sql_counter = 1
            print 'query_tok', cleaned_data['query_tok']
            parse_sql(cleaned_data, cleaned_data['query_tok'], schema, dbn, sql_counter, 'sql1') #recursive parsing
            print cleaned_data
            sql_data.append(cleaned_data)
            
        with open('New_Data/Initial/processed/train/' + dbn + '_processed.json', 'w') as fp:
            json.dump(sql_data, fp, indent=4)



### Need fix:
# 1) >= or <= operators in where/having condition (these get tokenized)
def parse_sql(cleaned_data, query_tok, schema, dbn, sql_counter, pointer):
    # print sql_counter
    cleaned_data[pointer] = {}
    ### find tables for from
    # query_tok = cleaned_data['query_tok']
    from_ind = query_tok.index('from')
    key_ind = None
    for word in extra_sql_keywords:
        if word in query_tok:
            key_ind = query_tok.index(word)
            break

    if key_ind is not None:
        from_clause = query_tok[from_ind + 1:key_ind]
    else:
        from_clause = query_tok[from_ind + 1:]
    
    join_ind = False
    col_map = schema[dbn]['col_map']
    temp_map = [tup[1] for tup in col_map]

    if 'as' in from_clause:
        join_ind = True
        as_inds = [ind for ind, tok in enumerate(from_clause) if tok == 'as']
        tables = [from_clause[ind - 1] for ind in as_inds]
        as_map = [from_clause[ind + 1] for ind in as_inds]

        on_inds = [ind for ind, tok in enumerate(from_clause) if tok == 'on']
        print('on_inds', on_inds)
        prejoins = [[from_clause[ind + 1].split('.'), from_clause[ind + 3].split('.')] for ind in on_inds] 
        print ('prejoins', prejoins)
        joins_tables = [[schema[dbn]['tab_map'].index(tables[as_map.index(l[0][0])]), schema[dbn]['tab_map'].index(tables[as_map.index(l[1][0])])] for l in prejoins]
        print ('joins_tables', joins_tables)
        joins_cols = []#= [[col_map[temp_map.index(l[0][1])][0], col_map[temp_map.index(l[1][1])][0]] for l in prejoins]
        for l in prejoins:
            curr_tab_1 = tables[as_map.index(l[0][0])]
            rel_cols = [entry for entry in col_map[1:] if entry[2] == curr_tab_1] #(ind, col_name, table_name_match)
            rel_cols_stripped = [entry[1] for entry in rel_cols] #just the col_names
            join_col_1 = rel_cols[rel_cols_stripped.index(l[0][1])][0]

            curr_tab_2 = tables[as_map.index(l[1][0])]
            rel_cols = [entry for entry in col_map[1:] if entry[2] == curr_tab_2] #(ind, col_name, table_name_match)
            rel_cols_stripped = [entry[1] for entry in rel_cols] #just the col_names
            join_col_2 = rel_cols[rel_cols_stripped.index(l[1][1])][0]

            joins_cols.append([join_col_1, join_col_2])

        joins = [[(tabs[0], cols[0]), (tabs[1], cols[1])] for tabs, cols in zip(joins_tables, joins_cols)]

        if len(joins) == 0: 
            tab_ind = schema[dbn]['tab_map'].index(tables[0])
            joins.append([(tab_ind, None), (None, None)])
    else:
        tables = from_clause
        tab_ind = schema[dbn]['tab_map'].index(tables[0])
        joins = [[(tab_ind, None), (None, None)]] #table 1, no column joined with no table and no column 

    # print cleaned_data['question']
    # print 'clause', from_clause
    # print 'tables', tables
    # print 'joins', joins

    cleaned_data[pointer]['from'] = joins
    tables_used = list(set([tup1[0] for [tup1, tup2] in joins if tup1[0] is not None] + [tup2[0] for [tup1, tup2] in joins if tup2[0] is not None]))
    
    ### get select and agg lists
    sel_clause = [tok for tok in query_tok[query_tok.index('select') + 1:query_tok.index('from')] if tok != 'distinct'] + ['<TEMP>']
    sel_inds = [0] + [ind + 1 for ind, tok in enumerate(sel_clause) if tok == ","] 
    sel_end_inds = [ind - 1 for ind in sel_inds[1:]] + [len(sel_clause)]

    agg_toks = ['max', 'min', 'avg', 'count', 'sum']
    agg_list = []
    temp_sel_cols = []

    for start_ind, end_ind in zip(sel_inds, sel_end_inds):
        curr_sel = sel_clause[start_ind:end_ind]
        found_agg = False
        for i, tok in enumerate(agg_toks):
            if tok in curr_sel:
                agg_list.append(i + 1)
                agg_ind = curr_sel.index(tok)
                temp_sel_cols.append(curr_sel[agg_ind + 2]) # get select column from within parentheses
                found_agg = True
                break
            
        if not found_agg:
            temp_sel_cols.append(curr_sel[0]) # in the case of no agg operator
            agg_list.append(0)

    if join_ind:
        sel_cols = [col.split('.') for col in temp_sel_cols]
        sel_col_inds = []

        for col_split in sel_cols:
            if len(col_split) == 1:
                if col_split[0] == '*':
                    sel_col_inds.append(0)
                else:                            
                    sel_col_inds.append(col_map[temp_map.index(col_split[0])][0]) # this happens only when a column is unique across tables 
            else:
                curr_tab = tables[as_map.index(col_split[0])]
                for col in col_map:
                    if (col[1] == col_split[1]) and (col[2] == curr_tab): # check for column and table match
                        sel_col_inds.append(col[0])
                        break
        
        #map tables back to tables, map columns back to columns 
        # sel_cols_tabs = [schema[dbn]['tab_map'].index()]
        # print sel_cols, joins
    else:
        # need to fix gettign the correct column from the correct table
        sel_cols = [col.split('.')[0] if len(col.split('.')) == 1 else col.split('.')[1] for col in temp_sel_cols]  
        curr_tab = schema[dbn]['tab_map'][tables_used[0]]
        rel_cols = [entry for entry in col_map[1:] if entry[2] == curr_tab] #(ind, col_name, table_name_match)
        rel_cols_stripped = [entry[1] for entry in rel_cols] #just the col_names
        sel_col_inds = [rel_cols[rel_cols_stripped.index(col)][0] if col != '*' else 0 for col in sel_cols]
        # sel_col_inds = [col_map[temp_map.index(col)][0] for col in sel_cols]

    # print query_tok
    # print sel_col_inds

    # print 'sel_agg', query_tok
    cleaned_data[pointer]['sel'] = sel_col_inds
    cleaned_data[pointer]['agg'] = agg_list
    # print cleaned_data


    ### parse where condition
    conds = []
    
    if 'where' in query_tok: 
        key_ind = None
        for word in extra_sql_keywords[1:]:
            if word in query_tok:
                key_ind = query_tok.index(word)
                break

        where_ind = query_tok.index('where')

        if key_ind is not None:
            where_clause = query_tok[where_ind + 1:key_ind]
        else:
            where_clause = query_tok[where_ind + 1:]

        # print where_clause
        if len(where_clause) > 0:
            if 'not' in where_clause:
                not_ind = where_clause.index('not')
                del where_clause[not_ind + 1] # delete the 'in' token

            if 'between' in where_clause: 
                between_ind = where_clause.index('between')
                after_between = where_clause[between_ind:]
                between_and_ind = after_between.index('and')
                del where_clause[between_ind + between_and_ind] # delete the 'and' token


            and_inds = [ind for ind, tok in enumerate(where_clause) if tok == 'and' or tok == 'or']
            st_inds = [0] + [ind + 1 for ind in and_inds]
            end_inds = [ind for ind in and_inds] + [len(where_clause)]

            # pos = 0
            # print st_inds, end_inds
            for (st, ed) in zip(st_inds, end_inds):
                curr = [0, 0, 0, 0]
                curr_cond = where_clause[st:ed]
                # print curr_cond

                # check for agg ops
                for i, tok in enumerate(agg_toks):
                    if tok in curr_cond:
                        curr[0] = i + 1
                        break

                # get cond operator
                op_ind = st
                between_flag = False
                subquery_flag = False

                for i, op in enumerate(where_ops):
                    if op in curr_cond:
                        op_ind += curr_cond.index(op)                           
                        curr[2] = i

                        if op == 'between':
                            between_flag = True
                        # if op == 'not' or op == 'in':
                        #     subquery_flag = True

                # get where col
                col = where_clause[st:op_ind] #hopefully just 1 token
                # print where_clause
                # print curr_cond
                # print where_clause
                # print sql_counter, col, curr_cond
                col_split = col[0].split('.')

                if len(col_split) == 1:
                    try:
                        curr[1] = col_map[temp_map.index(col_split[0])][0] # doesn't handle operators (e.g. end_date-start_date)
                    except: 
                        curr[1] = 0
                else:
                    curr_tab = tables[as_map.index(col_split[0])]
                    rel_cols = [entry for entry in col_map[1:] if entry[2] == curr_tab] 
                    rel_cols_stripped = [entry[1] for entry in rel_cols] 
                    curr[1] = rel_cols[rel_cols_stripped.index(col_split[1])][0]

                # get where val
                # print op_ind, curr_cond
                # val_ind = op_ind
                # if val_ind > 1:
                #     val_ind -= 1 # account for the and in between conditions
                val = curr_cond[op_ind - st + 1:] #hopefully just 1 token
                # print val
                if 'select' in val or subquery_flag: # if there's a subquery or aggregation
                    # print subquery_flag, val
                    # break
                    sql_counter += 1
                    new_pointer = 'sql' + str(sql_counter)
                    curr[3] = new_pointer
                    subquery = val
                    if subquery[0] == '(':
                        subquery = subquery[1:-1]
                    parse_sql(cleaned_data, subquery, schema, dbn, sql_counter, new_pointer)
                elif between_flag:
                    curr[3] = val
                else:
                    curr[3] = ''.join(val)

                conds += [curr]
    # print conds
    cleaned_data[pointer]['cond'] = conds

    ### parse group by/having
    group_list = []
    if 'group' in query_tok:
        key_ind = None
        for word in extra_sql_keywords[2:]:
            if word in query_tok:
                key_ind = query_tok.index(word)
                break

        group_ind = query_tok.index('group')

        if key_ind is not None:
            group_clause = query_tok[group_ind + 2:key_ind]
        else:
            group_clause = query_tok[group_ind + 2:]

        having_flag = False
        # print 'group_clause', group_clause
        for col in group_clause:
            if col == 'having':
                having_flag = True
                break
            if col == ',':
                continue
            col_split = col.split('.')
            if len(col_split) == 1:
                group_list.append(col_map[temp_map.index(col_split[0])][0])
            else:
                curr_tab = tables[as_map.index(col_split[0])]
                rel_cols = [entry for entry in col_map[1:] if entry[2] == curr_tab] 
                rel_cols_stripped = [entry[1] for entry in rel_cols] 
                group_list.append(rel_cols[rel_cols_stripped.index(col_split[1])][0])

        if having_flag: 
            having_ind = group_clause.index('having')
            print 'having', group_clause[having_ind + 1:]

            curr = [0, 0, 0, 0]
            curr_cond = group_clause[having_ind + 1:]
            # print curr_cond

            # check for agg ops
            for i, tok in enumerate(agg_toks):
                if tok in curr_cond:
                    curr[0] = i + 1
                    break

            # get cond operator
            st = 0
            op_ind = 0
            between_flag = False

            for i, op in enumerate(where_ops):
                if op in curr_cond:
                    op_ind += curr_cond.index(op)                           
                    curr[2] = i

                    if op == 'between':
                        between_flag = True

            # get where col
            col = curr_cond[st:op_ind] #hopefully just 1 token
            col_split = col[0].split('.')

            if len(col_split) == 1:
                try:
                    curr[1] = col_map[temp_map.index(col_split[0])][0] # doesn't handle operators (e.g. end_date-start_date)
                except: 
                    curr[1] = 0
            else:
                curr_tab = tables[as_map.index(col_split[0])]
                rel_cols = [entry for entry in col_map[1:] if entry[2] == curr_tab] 
                rel_cols_stripped = [entry[1] for entry in rel_cols] 
                curr[1] = rel_cols[rel_cols_stripped.index(col_split[1])][0]

            # get where val
            val = curr_cond[op_ind - st + 1:]
            curr[3] = ''.join(val)
            if between_flag:
                curr[3] = val

            group_list.append(curr)#group_clause[having_ind + 1:])

    cleaned_data[pointer]['group'] = group_list

    ### parse order by
    parity = 1
    order_list = []
    order_agg_list = []
    if 'order' in query_tok:
        key_ind = None
        for word in extra_sql_keywords[3:]:
            if word in query_tok:
                key_ind = query_tok.index(word)
                break

        order_ind = query_tok.index('order')

        if key_ind is not None:
            order_clause = query_tok[order_ind + 2:key_ind]
        else:
            order_clause = query_tok[order_ind + 2:]

        print 'order_clause', order_clause
        for col in order_clause:
            if col == ',':
                continue

            agg_found = False
            for i, tok in enumerate(agg_toks):
                if tok in order_clause:
                    order_agg_list.append(i + 1)
                    agg_found = True
                    break
            if agg_found:
                break
            order_agg_list.append(0)

            if col == 'asc' or col == 'desc':
                parity = 1 if col == 'asc' else 0
                break

            col_split = col.split('.')
            if len(col_split) == 1:
                order_list.append(col_map[temp_map.index(col_split[0])][0])
            else:
                curr_tab = tables[as_map.index(col_split[0])]
                rel_cols = [entry for entry in col_map[1:] if entry[2] == curr_tab] 
                rel_cols_stripped = [entry[1] for entry in rel_cols] 
                order_list.append(rel_cols[rel_cols_stripped.index(col_split[1])][0])

    if len(order_list) == 0:
        parity = -1

    cleaned_data[pointer]['order'] = [order_agg_list, order_list, parity]

    ### parse limit
    val = 0
    if 'limit' in query_tok:
        lim_val_ind = query_tok.index('limit') + 1
        val = query_tok[lim_val_ind]

    cleaned_data[pointer]['limit'] = val

    #parse union/intersect/except
    special_tok = None
    for i, word in enumerate(extra_sql_keywords[4:]):
        if word in query_tok:
            special_tok = word
            sql_counter += 1
            new_pointer = 'sql' + str(sql_counter)
            cleaned_data['special'] = [i + 1, new_pointer]
            
            special_ind = query_tok.index(word)
            subquery = query_tok[special_ind + 1:]
            if subquery[0] == '(':
                subquery = subquery[1:-1]
            parse_sql(cleaned_data, subquery, schema, dbn, sql_counter, new_pointer)


if __name__ == '__main__':
    table_create()