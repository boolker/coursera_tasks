from __future__ import print_function
import sys, os, math, random
import copy
__author__ = 'drummer'

def build_graph_by_pure_pairs(pairs, weights):
    #graph = []
    lefts = []
    rights = []
    max_node = 0
    for p in pairs:
        start = int(p[0])
        end = int(p[1])

        lefts.append(start)
        rights.append(end)
        if start > max_node:
            max_node = start
        if end > max_node:
            max_node = end

    l_len = len(lefts)
    #graph = [max_node+1] + [0]*((max_node+2)*(max_node+1))
    graph = []
    rev_graph = []
    N = max_node+1
    graph_header = [0]*(N+1)
    graph.append([] + graph_header)
    graph.append([] + graph_header)
    graph_row = []
    for i in xrange(2*N):
        graph.append([] + graph_row)
    #graph = zeros((max_node+1,max_node+1),int)
    graph[0][0] = N
    for l in xrange(l_len):
        cur_i = lefts[l]
        cur_j = rights[l]
        graph[cur_i+2].append([cur_j,weights[l]])
        graph[N+cur_j+2].append([cur_i])
        graph[0][1+ cur_i] += 1
        graph[1][1+ cur_j] += 1

    return graph

def get_longest_path_val(_start, _end, _graph, _path_graph):

    if _start == _end:
        return 0
    if _end == 55:
        nn = 0

    if _path_graph[_end] > (-sys.maxint - 1):
        return _path_graph[_end]

    max_val = (-sys.maxint - 1)
    max_i = 0
    N = _graph[0][0]
    pred_p = -1
    for p in _graph[2+N+_end]:
        if p[0] == 47:
            mm = 0
        w_list = _graph[2+p[0]]
        cur_weight = -sys.maxint - 1
        for w in w_list:
            #print(p[0], len(w_list))
            if w[0] == _end:
                cur_weight = w[1]
                if cur_weight > (-sys.maxint - 1):
                    cur_val = get_longest_path_val(_start,p[0],_graph, _path_graph) + cur_weight
                    if cur_val >= max_val:
                        max_val = cur_val
                        max_i = p[0]

    _path_graph[_end] = max_val
    for i in xrange(len(_graph[2+N+_end])):
        v = _graph[2+N+_end][i]
        if v[0] == max_i:
            _graph[2+N+_end][i].append(max_val)

    #_path_graph[_end] = 0
    #print(_i,_j,_graph[_i][_j],_backtrack[_i][_j])
    return max_val

def get_longest_path(_start, _end, _graph, _path_graph):

    if _start == _end:
        return []

    result = [_end]
    cur_start = _start
    cur_end = _end
    N = _graph[0][0]

    while cur_start != cur_end:
        max_val = (-sys.maxint - 1)
        p_s = _graph[2+N+cur_end]
        for p in _graph[2+N+cur_end]:
            if len(p) > 1:
                if p[1] > max_val:
                    max_val = p[1]
                    cur_end = p[0]
        if result[-1] == cur_end:
            print(cur_end)
            print(_graph[2+N+cur_end])
        #print(cur_end)
        result.append(cur_end)
    result.reverse()
    return result

def print_distances(_distances):
    d_len = len(_distances)
    for i in xrange(d_len):
        dist_str = ""
        for j in xrange(d_len):
            if i>j:
                dist_str += str(_distances[j][i-d_len]) + " "
            else:
                dist_str += str(_distances[i][j-d_len]) + " "
        print(dist_str)


def calc_distances(_graph, _leaves, _distances):    
    graph_size = _graph[0][0]

    for i in xrange(graph_size):
        if _graph[0][i+1] == 1:
            start_leave = _leaves.index(i)
            #find all distanses from i to other nodes
            print("find all distanses from " + str(i) + " to other nodes")                 
            path_nodes = _graph[i+2]    
            cur_node = -1
            prev_nodes = [i]
            while len(path_nodes)>=1:
                print("cur nodes: ",path_nodes,", prev nodes: ",prev_nodes)
                tmp_path_nodes = []
                for path_node in path_nodes:                                
                    cur_node = path_node[0]      
                    cur_val = path_node[1]
                    if cur_node in _leaves:
                        end_leave = _leaves.index(cur_node)
                        if end_leave != -1:
                            #set distance
                            if start_leave>end_leave:
                                _distances[end_leave][start_leave-end_leave]= cur_val
                            else:
                                _distances[start_leave][end_leave-start_leave]= cur_val
                        continue
                    end_nodes = _graph[cur_node+2]
                    print("start node: ",i, ", cur node: ",cur_node)
                    print("end nodes: ",end_nodes)
                    for end_node in end_nodes:                        
                        if end_node[0] not in prev_nodes:
                            print(end_node)
                            tmp_path_nodes.append([end_node[0],cur_val+end_node[1]])
                    prev_nodes.append(cur_node)
                print(tmp_path_nodes)
                path_nodes = tmp_path_nodes




def task41():
    #Distances Between Leaves Problem: Compute the distances between leaves in a weighted tree.
    #Input:  A weighted tree with n leaves.
    #Output: An n x n matrix (di,j), where di,j is the length of the path between leaves i and j.
    
    input_file_name = os.getcwd() + "/bio02/data/04/input43.txt"
    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()
    
    print(data)
    pairs = []
    weights = []
    for d in data[1:]:
        arch_info = d.replace('\n','').split(':')
        arch_str = arch_info[0]
        pair = [int(i) for i in arch_str.split('->')]
        arch_weight = int(arch_info[1])
        pairs.append(pair)
        weights.append(arch_weight)

    print(pairs)
    print(weights)

    graph = build_graph_by_pure_pairs(pairs,weights)
    print(graph)

    leaves = []
    graph_size = graph[0][0]
    for i in xrange(graph_size):
        if graph[0][i+1] >1:
            continue
        leaves.append(i)

    #print(leaves)        
    distances = []
    for i in xrange(len(leaves)):
        distances.append([-1 for j in xrange(len(leaves)-i)])
    for i in xrange(len(distances)):
        distances[i][0] = 0

    calc_distances(graph,leaves,distances)
    print_distances(distances)

def get_limb_length(_distances,_limb_num):
    dim = len(_distances)
    limb_val = sys.maxint
    for i in xrange(dim):
        for k in xrange(dim):
            if i!=k and i!=_limb_num and k!=_limb_num:
                val = (_distances[i][_limb_num]+_distances[_limb_num][k]-_distances[i][k])/2.0
                if val < limb_val:
                    limb_val = val

    return limb_val


def task42():
    input_file_name = os.getcwd() + "/bio02/data/04/input023.txt"
    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()
    
    dim = int(data[0])
    limb_num = int(data[1])
    distances=[]
    for d in data[2:]:
        row_str = d.replace('\n','').replace('\t',' ')
        row_data = [int(i) for i in row_str.split(' ')]
        distances.append(row_data)
        
    print(distances)
    
    print(get_limb_length(distances,limb_num))

def additive_phylogeny(_tree, _distances, _n):
    print(_distances,_n)
    distances_bald=[]
    distances_trimmed=[]

    if _n == 2:
        add_pair_to_graph(_tree, 0,1, _distances[0][1])
        #print(_tree)
        return

    limb_num = _n-1
    limb_len = get_limb_length(_distances,limb_num)
    #print(limb_len)

    for i in xrange(_n):
        distances_bald.append([0 for j in xrange(_n)])
        for j in xrange(_n):
            if i!=limb_num and j!= limb_num:
                distances_bald[i][j] = _distances[i][j]
            else:
                if i!=j:
                    distances_bald[i][j] = _distances[i][j]  - limb_len
    
    print(distances_bald) 
    _i = -1
    _k = -1
    _val = 0
    for i in xrange(_n-1):
        for k in xrange(_n-1):
            if i<k:
                if distances_bald[i][k] == (distances_bald[i][_n-1]+distances_bald[_n-1][k]):
                    #print(i,k,distances_bald[i][k],distances_bald[i][_n-1]+distances_bald[_n-1][k])
                    _i = i
                    _k = k
                    _val = distances_bald[i][_n-1]
                    break                    
    print(_i,_k,_val,limb_len)
    trimmed_len = _n -1
    for i in xrange(trimmed_len):
        distances_trimmed.append([0 for j in xrange(trimmed_len)])
        for j in xrange(trimmed_len):
            distances_trimmed[i][j] = distances_bald[i][j]

    #print(distances_trimmed)
    additive_phylogeny(_tree, distances_trimmed,trimmed_len)

    #add vertex
    path = find_path(_tree, _i, _k, _val)
    print(path)
    vertex_num = find_vertex_in_path(_tree,path,_val)
    add_edge(_tree,trimmed_len,vertex_num,limb_len)
    add_edge(_tree,vertex_num,trimmed_len,limb_len)
    print(_tree)



def build_empty_graph(_graph, _vnum):
    graph_header = [0]*(_vnum+1)
    _graph.append([] + graph_header)
    _graph.append([] + graph_header)
    graph_row = []
    for i in xrange(2*_vnum):
        _graph.append([] + graph_row)
    #graph = zeros((max_node+1,max_node+1),int)
    _graph[0][0] = _vnum


def add_pair_to_graph(_graph, _start, _end, _len):
    cur_i = _start
    cur_j = _end
    max_num = _graph[0][0]
    _graph[cur_i+2].append([cur_j,_len])
    _graph[max_num+cur_j+2].append([cur_i])
    _graph[0][1+ cur_i] += 1
    _graph[1][1+ cur_j] += 1

    cur_i = _end
    cur_j = _start
    _graph[cur_i+2].append([cur_j,_len])
    _graph[max_num+cur_j+2].append([cur_i])
    _graph[0][1+ cur_i] += 1
    _graph[1][1+ cur_j] += 1

def find_path(_tree,_start,_end,_len):
    print("find path from ", _start," to ",_end,"with len = ",_len)
    
    if _start >= _tree[0][0]:
        return []
    paths = []
    paths.append([_start])
    while len(paths) >0:
        tmp_paths = []
        for _path in paths:
            start_leave = _path[-1]
            #print(start_leave)
            for end_v in _tree[2+start_leave]:
                tmp_path = _path[:]
                if end_v[0] not in tmp_path:
                    tmp_path.append(end_v[0])
                    #print("tmp path is ",tmp_path," new end point is ",end_v[0])
                    if end_v[0] == _end:
                        #print(tmp_path)
                        return tmp_path
                    tmp_paths.append(tmp_path)
        paths = tmp_paths
        #print("paths are ", paths)

    return []

def find_vertex_in_path(_tree,_path,_len):
    print("find vertex in path ",_path," with len ",_len)
    cur_len = _len
    old_len = 0
    prev_p = -1
    p = -1
    res_vertex = -1
    for i in xrange(len(_path)):
        p = _path[i]
        if prev_p>=0:
            for end_p in _tree[2+prev_p]:
                if end_p[0] == p:
                    old_len = cur_len
                    cur_len -= end_p[1]
                    break
        if cur_len <0:
            #add vertex between prev_p and p
            res_vertex = add_vertex(_tree,prev_p,p,old_len)
            return res_vertex
        elif cur_len == 0:
            res_vertex = p
            break
        prev_p = p
    return res_vertex

def add_vertex(_tree,_start,_end,_len):
    v_num = -1
    print("add vertex from ", _start," to ",_end,"at len = ",_len)
    v_num = _tree[0][0]
    _tree[0][0] += 1
    _tree.insert(2+v_num,[])
    _tree.append([])

    _tree[0].append(0)
    _tree[1].append(0)


    for v in _tree[2+_start]:
        if v[0] == _end:
            v[0] = v_num
            v[1] = _len
            _tree[2+v_num].append([_start,_len])
            _tree[0][1+ v_num] += 1
            
    for v in _tree[2+_end]:
        if v[0] == _start:
            v[0] = v_num
            v[1] = v[1] - _len
            _tree[2+v_num].append([_end,v[1]])
            _tree[0][1+ v_num] += 1

    print(_tree)
    return v_num

def add_vertex_alone(_tree):    
    v_num = _tree[0][0]
    _tree[0][0] += 1
    _tree.insert(2+v_num,[])
    _tree.append([])

    _tree[0].append(0)
    _tree[1].append(0)

    return v_num

def add_edge(_tree,_start,_end,_len):
    print("add edge from ", _start," to ",_end,"with len = ",_len)
    
    N = _tree[0][0]
    _tree[_start+2].append([_end,_len])
    _tree[N+_end+2].append([_start])
    _tree[0][1+ _start] += 1
    _tree[1][1+ _end] += 1

def del_edge(_tree,_start,_end):
    print("del edge from ", _start," to ",_end)     

    for i in xrange(len(_tree[_start+2])):
        end = _tree[_start+2][i]
        if end[0] == _end:
            del _tree[_start+2][i]
            break
    
    _tree[0][1+ _start] -= 1
    _tree[1][1+ _end] -= 1

def print_tree(_tree):
    N = _tree[0][0]
    for i in xrange(N):
        ends = _tree[2+i]
        for end in ends:
            print(i,"->",end[0],":","{0:.3f}".format(end[1]))

def to_int(s):
    try:
        return int(s)
    except ValueError:
        return -1

def construct_parsimony_graph(_size,_data):
    _graph = []
    print(_data)
    pairs = []
    cur_len = _size
    print(cur_len)
    build_empty_graph(_graph,_size)

    for d in _data:
        arch_info = d.replace('\n','')        
        pair = [leaf for leaf in arch_info.split('->')]        
        pairs.append(pair)
        
    cur_empty = 0;
    strings = {}
    print(pairs)
    for p in pairs:
        start_int = to_int(p[0])
        if start_int <0:
            #string
            res = strings.get(p[0],cur_empty);
            if res == cur_empty:
                strings[cur_empty] = p[0]
                #strings[cur_empty] = p[0][2]
                start_int = cur_empty
                cur_empty += 1
        else:
            strings[start_int] = ""
            while start_int >= _graph[0][0]:
                add_vertex_alone(_graph)

        end_int = to_int(p[1])
        if end_int <0:
            #string
            res = strings.get(p[1],cur_empty);
            if res == cur_empty:
                strings[cur_empty] = p[1]
                #strings[cur_empty] = p[1][2]
                end_int = cur_empty
                cur_empty += 1
        else:
            strings[end_int] = ""
            while end_int >= _graph[0][0]:
                add_vertex_alone(_graph)

        add_edge(_graph,start_int,end_int,0)
        #add_edge(_graph,end_int,start_int,0)
            

    #print(strings)
    #print(_graph)

    return [_graph,strings]

def construct_rooted_parsimony_graph(_size,_data):
    _graph = []
    print(_data)
    pairs = []
    cur_len = _size
    print(cur_len)
    build_empty_graph(_graph,_size)

    for d in _data:
        arch_info = d.replace('\n','')        
        pair = [leaf for leaf in arch_info.split('->')]        
        pairs.append(pair)
        
    cur_empty = 0;
    strings = {}
    strings_rev = {}
    print(pairs)
    start_int = -1
    end_int = -1
    for p in pairs:
        start_int = to_int(p[0])
        if start_int <0:
            #string
            res = strings_rev.get(p[0],cur_empty);
            if res == cur_empty:
                strings[cur_empty] = p[0]
                strings_rev[p[0]] = cur_empty                
                start_int = cur_empty
                cur_empty += 1
            else:
                start_int = res
        else:
            strings[start_int] = ""
            while start_int >= _graph[0][0]:
                add_vertex_alone(_graph)

        end_int = to_int(p[1])
        if end_int <0:
            #string
            res = strings_rev.get(p[1],cur_empty);
            if res == cur_empty:
                strings[cur_empty] = p[1]
                strings_rev[p[1]] = cur_empty                 
                end_int = cur_empty
                cur_empty += 1
            else:
                end_int = res
        else:
            strings[end_int] = ""
            while end_int >= _graph[0][0]:
                add_vertex_alone(_graph)

        add_edge(_graph,start_int,end_int,0)
        #add_edge(_graph,end_int,start_int,0)
    #take last pair
    del_edge(_graph,start_int,end_int)
    del_edge(_graph,end_int,start_int)

    root_id = add_vertex_alone(_graph)
    strings[root_id] = ''
    add_edge(_graph,root_id,start_int,0)
    add_edge(_graph,root_id,end_int,0)

    print(_graph)

    tagged = [root_id]
    cur_nodes = _graph[2+root_id]
    while len(cur_nodes)>0:
        new_nodes = []
        for node in cur_nodes:
            node_id = node[0]
            tagged.append(node_id)
            children = _graph[2+node_id]
            for child in children:
                if child[0] in tagged:
                    #del 
                    del_edge(_graph,node_id,child[0])
                    continue
                else:
                    new_nodes.append(child)

        cur_nodes = new_nodes[:]

    print(strings)
    print(_graph)

    return [_graph,strings]

def calc_parsimony_values(_tree,_v_id,_values,_tags,_alphabet,_def_val):
    if _tags[_v_id] == 1:
        return
    print("calc ",_v_id)
    children = _tree[2+_v_id]
    cur_def_val = []
    for d in _def_val:
        cur_def_val.append(d[:])    
    self_val = _values.get(_v_id,cur_def_val)
    #for l in xrange(len(_def_val)):
    #    for k in xrange(len(_alphabet)):
    
    for child in children:
        if _tags[child[0]] == 0:
            calc_parsimony_values(_tree,child[0],_values,_tags,_alphabet,_def_val)
        child_val = _values[child[0]]
        child_vals = []
        print(child)
        for i in xrange(len(self_val)):
            #for each letter
            #print("letter # ",i)
            #print(child_val[i])
            for j in xrange(len(_alphabet)):    
                #print(_alphabet[j])     
                min_val = sys.maxint       
                for k in xrange(len(_alphabet)):
                    delta = 0
                    if j!=k:
                        delta = 1                    
                                    
                    if child_val[i][k] < sys.maxint:
                        child_len = child_val[i][k]
                    else:
                        child_len = 1

                    #print(j,k,delta,child_len)
                    if (child_len + delta) < min_val:
                        min_val = child_len + delta

                #print(i,j,min_val)
                self_val[i][j] += min_val                            
        print(self_val)
    _values[_v_id] = self_val[:]

    _tags[_v_id] = 1

def generate_strings(_tree,_root_id,_values,_strings,_alphabet):    
    
    if _root_id == -1:
        tree_size = _tree[0][0]
        for i in xrange(tree_size):
            if _tree[1][i+1] == 0:
                value = _values[i]
                new_str = ''
                #print(value)
                for v_el in value:
                    min_i = -1
                    min_val = sys.maxint
                    for j in xrange(len(v_el)):
                        if v_el[j] < min_val:
                            min_val = v_el[j]
                            min_i = j
                    new_str += _alphabet[min_i]
                #print(new_str)
                _strings[i] = new_str[:]        
                generate_strings(_tree,i,_values,_strings,_alphabet)
                break
    else:
        root_str = _strings[_root_id]
        #print(_root_id,root_str)
        children = _tree[2+_root_id]
        for child in children:
            child_id = child[0]
            #print("child ",child_id)
            if len(_strings[child_id]) >0:
                continue
            value = _values[child_id]            
            new_str = ''
            #print(value)
            v_ind = 0
            for v_el in value:
                min_i = -1
                min_val = sys.maxint
                min_letters = []
                for j in xrange(len(v_el)):
                    if v_el[j] < min_val:
                        min_val = v_el[j]
                        #min_i = j
                        min_letters = [_alphabet[j]]
                    elif v_el[j] == min_val:
                        min_letters.append(_alphabet[j])
                #new_str += _alphabet[min_i]
                #print("parent letter ",root_str[v_ind],", letters are ", min_letters)
                if root_str[v_ind] in min_letters:
                    new_str += root_str[v_ind]
                else:
                    new_str += min_letters[0]
                v_ind += 1
            #print(new_str)
            _strings[child_id] = new_str[:]        
            generate_strings(_tree,child_id,_values,_strings,_alphabet)
    
    pass

def calc_parsimony_score(_tree,_values,_strings,_alphabet):
    root_id = -1
    tree_size = _tree[0][0]
    score = 0
    for i in xrange(tree_size):
        if _tree[1][i+1] == 0:
            root_id = i
            break
    value = _values[root_id]
    for val in value:
        min_v = sys.maxint
        for v in val:
            if v<min_v:
                min_v = v
        score += min_v
    print(score)    

def calc_parsimony_score_ex(_tree,_strings):
    tree_size = _tree[0][0]
    score = 0
    for i in xrange(tree_size):
        edges = _tree[2+i]
        for edge in edges:
            score += hammings_dist(_strings[i],_strings[edge[0]])
        
    #print(score/2)    
    return score/2

def hammings_dist(_str_a,_str_b):
    dist = 0
    _len = len(_str_a)
    for i in xrange(_len):
        if _str_a[i] != _str_b[i]:
            dist +=1
    return dist


def print_parsinomy_tree(_tree,_strings,_rooted=0):
    tree_size = _tree[0][0]
    for i in xrange(tree_size):
        for ends in _tree[2+i]:
            end = ends[0]
            str_a = ''
            str_a = "("+str(i)+"->"+str(end)+") "
            str_a += _strings[i]+"->"+_strings[end]+":"+str(hammings_dist(_strings[i],_strings[end]))
            #print("%1>%2:%3" % _strings[i],_strings[end],hammings_dist(_strings[i],_strings[end]))
            print(str_a)
            if _rooted == 1:
                str_b = _strings[end]+"->"+_strings[i]+":"+str(hammings_dist(_strings[i],_strings[end]))
                print(str_b)
            #print(_strings[end],"->",_strings[i],":",hammings_dist(_strings[i],_strings[end]))

def print_tree(_tree):
    tree_size = _tree[0][0]
    for i in xrange(tree_size):
        for ends in _tree[2+i]:
            end = ends[0]
            str_a = str(i)+"->"+str(end)
            print(str_a)            

def small_parsimony(_tree,_strings):
    alphabet = ['A','C','G','T']

    print(_tree)
    print(_strings)
    max_len = 0
    for s in _strings:
        len_s = len(_strings[s])
        if  len_s > max_len:
            max_len = len_s

    print(max_len) 
    tree_size = _tree[0][0]
    s_values = {}
    def_self_val = []
    for i in xrange(max_len):
         def_self_val.append([0 for j in xrange(len(alphabet))])

    vertex_tags = [0 for j in xrange(tree_size)]
    for i in xrange(tree_size):
        if len(_tree[2+i]) == 0:
            s_value = []
            for s in _strings[i]:                
                letter_s = [sys.maxint for j in xrange(len(alphabet))]
                letter_ind = alphabet.index(s)
                letter_s[letter_ind] = 0
                s_value.append(letter_s)               
            s_values[i] = s_value
            vertex_tags[i] = 1
    print(s_values)
    print(vertex_tags)

    ripes = [x for x in vertex_tags if x == 0]
    #while len(ripes) > 0:
    print(_tree)
    for i in xrange(len(vertex_tags)):
        if vertex_tags[i] != 0:
            continue
        calc_parsimony_values(_tree,i,s_values,vertex_tags,alphabet,def_self_val)
        #break
    print(s_values)
    calc_parsimony_score(_tree,s_values,_strings,alphabet)    
    generate_strings(_tree,-1,s_values,_strings,alphabet)
    #print(_strings)
    return _tree

def get_distance_to_centers(_point,_points,_centers,_dim):
    res = sys.maxint
    for c in _centers:
        dist = 0
        for i in xrange(_dim):
            dist += (_point[i]-_points[c][i])*(_point[i]-_points[c][i])
        dist = math.sqrt(dist)
        if dist < res:
            res = dist
    return res

def get_distance(_a,_b,_dim):
    res = 0
    for i in xrange(_dim):
        res += (_a[i]-_b[i])*(_a[i]-_b[i])
        #dist = math.sqrt(dist)
    res = math.pow(res,1.0/2)
    return res

def get_distance_to_centers_ex(_point,_points,_centers,_dim):
    res = sys.maxint
    for c in _centers:
        dist = 0        
        for i in xrange(_dim):
            dist += (_point[i]-c[i])*(_point[i]-c[i])
        #dist = math.sqrt(dist)
        dist = math.pow(dist,1.0/2)
        if dist < res:
            res = dist    
    return res

def distortion(_points,_centers,_dim):
    res = 0
    n = len(_points)
    print(n)
    for i in xrange(n):
        dist = get_distance_to_centers_ex(_points[i],_points,_centers,_dim)
        res += dist*dist
    print(res)
    res = res/n
    return res

def max_distance(_points,_centers,_dim):
    max_dist = 0    
    for i in xrange(len(_points)):
        pt_dist = get_distance_to_centers_ex(_points[i],_points,_centers,_dim)
        if pt_dist > max_dist:
            max_dist = pt_dist            
    return max_dist

def task71():
    # Implement the FarthestFirstTraversal clustering heuristic.
    # Input: Integers k and m followed by a set of points Data in m-dimensional space.
    # Output: A set Centers consisting of k points (centers) resulting from applying
    # FarthestFirstTraversal(Data, k), where the first point from Data is chosen as the
    # first center to initialize the algorithm.
    input_file_name = os.getcwd() + "/part2/data/07/input12.txt"
    #input_file_name = "/Users/boolker/Desktop/tasks/bio02/data/04/input034.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()
    
    k = int((data[0].replace('\n','').split(' '))[0])
    dim = int((data[0].replace('\n','').split(' '))[1])
    print(k)
    points = []
    for d in data[1:]:
        point_strings = d.replace('\n','').split(' ')
        points.append([float(x) for x in point_strings])
    #print(points)

    centers = [0]
    while len(centers) < k:
        max_dist = 0
        max_i = -1
        for i in xrange(len(points)):
            if i in centers:
                continue
            pt_dist = get_distance_to_centers(points[i],points,centers,dim)
            if pt_dist > max_dist:
                max_dist = pt_dist
                max_i = i
        centers.append(max_i)        
    #print(centers)
    for c in centers:
        str_c = ""
        for i in xrange(dim):
            str_c += str(points[c][i]) + " "
        print(str_c)

def task720():  
    points = [[1.0,6.0],[1.0,3.0],[3.0,4.0],[5.0,2.0],[5.0,6.0],[7.0,1.0],[8.0,7.0],[10.0,3.0]]
    centers = [[3.0,4.5],[6.0,1.5],[9.0,5.0]]
    centers1 = [[5/3.0,13.0/3],[6.5,6.5],[22.0/3,2.0]]

    print(max_distance(points,centers,2),distortion(points,centers,2))
    print(max_distance(points,centers1,2),distortion(points,centers1,2))

def task72():
    # Solve the Squared Error Distortion Problem.
    # Input: Integers k and m, followed by a set of centers Centers and a set of points Data.
    # Output: The squared error distortion Distortion(Data, Centers).
    input_file_name = os.getcwd() + "/part2/data/07/input22.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    k = int((data[0].replace('\n','').split(' '))[0])
    dim = int((data[0].replace('\n','').split(' '))[1])

    centers = []
    points = []
    for i in xrange(k):
        cnt_string = data[i+1].replace('\n','').split(' ')
        centers.append([float(x) for x in cnt_string])

    for d in data[k+2:]:        
        pt_string = d.replace('\n','').split(' ')
        points.append([float(x) for x in pt_string])

    #print(centers)
    #print(points)
    res = distortion(points,centers,dim)
    print(res)

def calc_centers(_clusters,_dim):
    _centers = []
    for cluster in _clusters:
        c_len = len(cluster)
        cnt = [0 for i in xrange(_dim)]
        for i in xrange(_dim):
            for c in cluster:            
                cnt[i] += c[i]
            cnt[i] = cnt[i]/c_len
        _centers.append(cnt)

    return _centers

def calc_center_of_mass(_points,_dim):
    _center = [0 for i in xrange(_dim)]
    for i in xrange(_dim):
        for c in _points:            
            _center[i] += c[i]
        _center[i] = _center[i]/len(_points)       

    return _center

def calc_centers_delta(_old,_cur,_dim):
    delta = 0
    _k = len(_old)
    for i in xrange(_k):
        delta += get_distance(_cur[i],_old[i],_dim)
    delta = delta/_k
    return delta

def init_centers(_points,_k,_dim):
    centers = []
    n = len(_points)
    centers.append(random.randrange(0,n))    
    while len(centers) < _k:
        rand_pt = 0
        distances = []
        sum_dist = 0
        for i in xrange(len(_points)):
            if i in centers:
                distances.append(0)
                continue
            dist = get_distance_to_centers(_points[i],_points,centers,_dim)
            distances.append(dist)
            sum_dist += dist
            
        #print("sum = ",sum_dist)
        #for i in xrange(len(_points)):
        #    print(i,distances[i]/sum_dist)
        choice = random.random()
        choice_num = -1
        cur_val = 0
        for d in distances:        
            cur_val += d/sum_dist
            choice_num +=1
            if cur_val > choice:
                break
        #print(choice, choice_num)
        centers.append(choice_num)
    #print(centers)
    return centers

def lloyd_algorythm(_points,_k,_dim):
    old_centers = []
    cur_centers = []
    rand_centers = init_centers(_points,_k,_dim)
    for i in xrange(_k):        
        #cur_centers.append(_points[rand_centers[i]])
        cur_centers.append(_points[i])
        old_centers.append([0 for i in xrange(_dim)])

    delta = calc_centers_delta(old_centers,cur_centers,_dim)
    #print("delta is ", delta)
    while delta > 0.0000001:
        clusters = []
        for i in xrange(_k):        
            clusters.append([])

        for i in xrange(len(_points)):
            min_dist = sys.maxint
            min_j = -1
            for j in xrange(_k):
                dist = get_distance(_points[i],cur_centers[j],_dim)
                #print(i,j,dist)
                if dist < min_dist:
                    min_dist = dist
                    min_j = j
            clusters[min_j].append(_points[i])
        #print(clusters)
        old_centers = cur_centers[:]
        cur_centers = calc_centers(clusters,_dim)
        #print(cur_centers)
        delta = calc_centers_delta(old_centers,cur_centers,_dim)
        #print("delta is ", delta)

    return cur_centers

def task73():
    # Implement the Lloyd algorithm for k-means clustering.
    # Input: Integers k and m followed by a set of points Data in m-dimensional space.
    # Output: A set Centers consisting of k points (centers) resulting from applying the
    # Lloyd algorithm to Data and Centers, where the first k points from Data are selected
    # as the first k centers.

    input_file_name = os.getcwd() + "/part2/data/07/input3.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    k = int((data[0].replace('\n','').split(' '))[0])
    dim = int((data[0].replace('\n','').split(' '))[1])

    
    points = []
    for d in data[1:]:        
        pt_string = d.replace('\n','').split(' ')
        points.append([float(x) for x in pt_string])

    print(points)
    centers = lloyd_algorythm(points,k,dim)
    
    print(centers)
    for c in centers:
        str_c = ""
        for i in xrange(dim):
            str_c += "{0:.3f}".format(c[i])
            str_c += " "
        print(str_c)

def task731():
    # Implement the Lloyd algorithm for k-means clustering.
    # Input: Integers k and m followed by a set of points Data in m-dimensional space.
    # Output: A set Centers consisting of k points (centers) resulting from applying the
    # Lloyd algorithm to Data and Centers, where the first k points from Data are selected
    # as the first k centers.

    input_file_name = os.getcwd() + "/part2/data/07/input32.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    k = 6
    dim = 7

    
    points = []
    for d in data[1:]:        
        pt_string = d.replace('\n','').split('\t')
        points.append([float(x) for x in pt_string[3:]])

    #print(points)
    centers = lloyd_algorythm(points,k,dim)
    
    print(centers)
    for c in centers:
        str_c = ""
        for i in xrange(dim):
            str_c += "{0:.3f}".format(c[i])
            str_c += " "
        print(str_c)

def task_quizz():

    points = [[2,6],[4,9],[5,7],[6,5],[8,3]]
    centers = [[4,5],[7,4]]
    print(distortion(points,centers,2))
    print(max_distance(points,centers,2))

    centers_2 = [[1,3,-1],[9,8,14],[6,2,10],[4,3,1]]
    print(calc_center_of_mass(centers_2,3))

if __name__ == "__main__":   
    #task731() 
    task_quizz()
    
