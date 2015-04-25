from __future__ import print_function
import sys, os
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

def print_tree(_tree):
    N = _tree[0][0]
    for i in xrange(N):
        ends = _tree[2+i]
        for end in ends:
            print(i,"->",end[0],":","{0:.3f}".format(end[1]))


def construct_parsimony_graph(_size,_data):
    _graph = []
    print(_data)
    pairs = []
    for d in _data:
        arch_info = d.replace('\n','')        
        pair = [leaf for leaf in arch_info.split('->')]        
        pairs.append(pair)
        
    print(pairs)
    return _graph


def small_parsimony(_tree):
    pass

def task61():
    # Implement SmallParsimony to solve the Small Parsimony Problem.
    # Input: An integer n followed by an adjacency list for a rooted binary tree with n leaves
    # labeled by DNA strings.
    # Output: The minimum parsimony score of this tree, followed by the adjacency list of the
    # tree corresponding to labeling internal nodes by DNA strings in order to minimize the
    # parsimony score of the tree.
    input_file_name = os.getcwd() + "/bio02/data/06/input01.txt"
    #input_file_name = "/Users/boolker/Desktop/tasks/bio02/data/04/input034.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()
    
    dim = int(data[0])    
    p_graph = construct_parsimony_graph(dim,data[1:])
    res_tree = small_parsimony(p_graph)
    print(res_tree)

    

if __name__ == "__main__":   
    task61() 
    
