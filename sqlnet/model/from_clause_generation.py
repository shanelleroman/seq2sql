import traceback
from collections import *
import json

def find_shortest_path(start, end, graph):
    stack = [[start, []]]
    visited = set()
    while len(stack) > 0:
        ele, history = stack.pop()
        if ele == end:
            return history
        for node in graph[ele]:
            if node[0] not in visited:
                stack.append((node[0], history + [(node[0], node[1])]))
                visited.add(node[0])
    print("Could not find a path between table {} and table {}".format(start, end))

def gen_from(candidate_tables, schema):
    if len(candidate_tables) <= 1:
        if len(candidate_tables) == 1:
            ret = "from {}".format(schema["table_names"][list(candidate_tables)[0]])
        else:
            ret = "from {}".format(schema["table_names"][0])
        # TODO: temporarily settings for select count(*)
        return {}, ret
    # print("candidate:{}".format(candidate_tables))
    table_alias_dict = {}
    uf_dict = {}
    for t in candidate_tables:
        uf_dict[t] = -1
    idx = 1
    graph = defaultdict(list)
    for acol, bcol in schema["foreign_keys"]:
        t1 = schema["col_map"][acol][0]
        t2 = schema["col_map"][bcol][0]
        graph[t1].append((t2, (acol, bcol)))
        graph[t2].append((t1, (bcol, acol)))
    candidate_tables = list(candidate_tables)
    start = candidate_tables[0]
    table_alias_dict[start] = idx
    idx += 1
    ret = "from {} as T1".format(schema["table_names"][start])
    try:
        for end in candidate_tables[1:]:
            if end in table_alias_dict:
                continue
            path = find_shortest_path(start, end, graph)
            prev_table = start
            if not path:
                table_alias_dict[end] = idx
                idx += 1
                ret = "{} join {} as T{}".format(ret, schema["table_names"][end],
                                                 table_alias_dict[end],
                                                 )
                continue
            for node, (acol, bcol) in path:
                if node in table_alias_dict:
                    prev_table = node
                    continue
                table_alias_dict[node] = idx
                idx += 1
                ret = "{} join {} as T{} on T{}.{} = T{}.{}".format(ret, schema["table_names"][node],
                                                                    table_alias_dict[node],
                                                                    table_alias_dict[prev_table],
                                                                    schema["column_names_original"][acol][1],
                                                                    table_alias_dict[node],
                                                                    schema["column_names_original"][bcol][1])
                prev_table = node
    except:
        traceback.print_exc()
        # print("db:{}".format(schema["db_id"]))
        # print(table["db_id"])
        return table_alias_dict, ret
    # if len(candidate_tables) != len(table_alias_dict):
    #     print("error in generate from clause!!!!!")
    return table_alias_dict, ret