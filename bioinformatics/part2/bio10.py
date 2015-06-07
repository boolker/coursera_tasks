from __future__ import print_function
import sys, os, math, random
import copy
__author__ = 'drummer'

import matplotlib.pyplot as plt



def to_int(s):
    try:
        return int(s)
    except ValueError:
        return -1

def build_empty_graph(_graph, _vnum):
    graph_header = [0]*(_vnum+1)
    _graph.append([] + graph_header)
    _graph.append([] + graph_header)
    graph_row = []
    for i in xrange(2*_vnum):
        _graph.append([] + graph_row)
    #graph = zeros((max_node+1,max_node+1),int)
    _graph[0][0] = _vnum

def add_vertex(_tree,_start,_end,_len):
    v_num = -1
    #print("add vertex from ", _start," to ",_end,"at len = ",_len)
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
    #print("add edge from ", _start," to ",_end,"with len = ",_len)    
    N = _tree[0][0]
    _tree[_start+2].append([_end,_len])
    _tree[N+_end+2].append([_start])
    _tree[0][1+ _start] += 1
    _tree[1][1+ _end] += 1


def cluster_distance_avg(_distances, cluster_a, cluster_b):
    res = 0
    for a in cluster_a:
        for b in cluster_b:
            res += _distances[a][b]
    res = res / (len(cluster_a)*len(cluster_b))
    return res

def cluster_distance_min(_distances, cluster_a, cluster_b):
    res = sys.maxint
    for a in cluster_a:
        for b in cluster_b:
            if _distances[a][b] < res:
                res = _distances[a][b]
    
    return res

def hierarchical_clustering(_distances,_size):
    clusters = []    
    cluster_ids = []
    ages = []
    
    result_tree = []    
    for i in xrange(_size):
        clusters.append([i])    
        cluster_ids.append(i)
        ages.append(0)

    build_empty_graph(result_tree,_size)    

    cluster_distances = []
    for d in _distances:
        cluster_distances.append(d[:])
    
    while len(cluster_ids)>1:
        # 1. find min distance between clusters
        min_dist = sys.maxint
        min_i = -1
        min_j = -1
        clusters_len = len(cluster_ids)
        for i in xrange(clusters_len):
            for j in xrange(clusters_len):
                if i<j:
                    c_i = cluster_ids[i]
                    c_j = cluster_ids[j]                    
                    c_distance = cluster_distance(_distances,clusters[c_i],clusters[c_j])
                    if c_distance< min_dist:
                        min_dist = c_distance
                        min_i = c_i
                        min_j = c_j
        #print(min_i,min_j,min_dist)
        if min_i <0 or min_j <0:
            print("error searching minimum distance")
            return []
        # 2. create cluster
        new_cluster = clusters[min_i][:] + clusters[min_j][:]    
        #print(new_cluster)            
        str_c = ""
        for c in new_cluster:
            str_c += str(c+1) + " "
        print(str_c)
        clusters.append(new_cluster)
        ages.append(min_dist/2.0)
        
        v_num = add_vertex_alone(result_tree)
        cluster_ids.append(v_num)
                
        cluster_ids.remove(min_j)
        cluster_ids.remove(min_i)
        
        add_edge(result_tree,min_i,v_num,0)
        add_edge(result_tree,min_j,v_num,0)
        add_edge(result_tree,v_num,min_i,0)
        add_edge(result_tree,v_num,min_j,0)

    #print(clusters)           
    
    return result_tree

def task91():
    # Solve the Probability of a Hidden Path Problem.
    # Given: A hidden path pi followed by the states States and transition matrix Transition of an HMM
    # ( States, Transition, Emission).
    # Return: The probability of this path, Pr().

    input_file_name = os.getcwd() + "/part2/data/09/input11.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    path = data[0].replace('\n','')
    print(path)
    elems_str = data[2].replace('\n','').replace('\t',' ')        
    elems = [e for e in elems_str.split(' ')]
    print(elems)
    elem_size = len(elems)
    mtx = []
    for i in xrange(elem_size):
        mtx_str = data[5+i].replace('\n','').replace('\t',' ')        
        mtx_els = [float(i) for i in (mtx_str.split())[1:]]
        mtx.append(mtx_els)

    res = 0.5
    for i in xrange(len(path)):
        if i == 0:
            continue
        
        c_from = path[i-1]
        c_to = path[i]
        from_ind = elems.index(c_from)
        to_ind = elems.index(c_to)        
        res = res * mtx[from_ind][to_ind]
    print(res)

def task92():
    # Solve the Probability of an Outcome Given a Hidden Path Problem.
    # Input: A string x, followed by the alphabet from which x was constructed, followed by
    # a hidden path pi, followed by the states States and emission matrix Emission of an HMM
    # (sigma, States, Transition, Emission).
    # Output: The conditional probability Pr(x|pi) that x will be emitted given that the HMM
    # follows the hidden path pi.

    input_file_name = os.getcwd() + "/part2/data/09/input21.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    dest_path = data[0].replace('\n','')    
    elems_str = data[2].replace('\n','').replace('\t',' ')        
    dest_elems = [e for e in elems_str.split(' ')]

    src_path = data[4].replace('\n','')
    elems_str = data[6].replace('\n','').replace('\t',' ')        
    src_elems = [e for e in elems_str.split(' ')]
    
    print(dest_path)
    print(src_path)
    print(dest_elems)
    print(src_elems)
    
    src_elem_size = len(src_elems)
    mtx = []
    for i in xrange(src_elem_size):
        mtx_str = data[9+i].replace('\n','').replace('\t',' ')        
        mtx_els = [float(i) for i in (mtx_str.split())[1:]]
        mtx.append(mtx_els)

    print(mtx)

    res = 1
    for i in xrange(len(src_path)):        
        c_from = src_path[i]
        c_to = dest_path[i]
        from_ind = src_elems.index(c_from)
        to_ind = dest_elems.index(c_to)        
        res = res * mtx[from_ind][to_ind]
    print(res)


def get_optimal_path(_transition,_emission,_path,_src,_dest):    

    res_mtx = []
    path_points = []

    src_elem_size = len(_src)
    for i in xrange(src_elem_size):
        res_mtx.append([0 for j in xrange(len(_path))])
        path_points.append([0 for j in xrange(len(_path))])

    for _iter in xrange(len(_path)):        
        path_ind = _dest.index(_path[_iter])
        if _iter == 0:
            res = 0.5            
            for i in xrange(len(_src)):
                res_mtx[i][_iter] = 1*0.5*_emission[i][path_ind]
        else:
            for i in xrange(len(_src)):
                max_val = 0
                max_ind = -1
                for j in xrange(len(_src)):
                    cur_val = res_mtx[j][_iter-1]*_transition[j][i]*_emission[i][path_ind]
                    #if _iter == 4:
                    #    print(_iter,i,j,res_mtx[j][_iter-1],_transition[j][i],_emission[i][path_ind],cur_val)
                    if cur_val > max_val:
                        max_val = cur_val
                        max_ind = j
                res_mtx[i][_iter] = max_val
                path_points[i][_iter] = max_ind

    for s in res_mtx:
        str_s = ""
        for _s in s:
            if _s>0:
                #str_s += str(math.log(_s,2)) + " "
                str_s += str(_s) + " "
            else:
                str_s += "0"
        print(str_s)

    for s in path_points:
        str_s = ""
        for _s in s:            
            str_s += str(_s) + " "
            
        print(str_s)

    res_str = ""
    cur_max_ind = -1
    max_val = 0        
    next_point = -1
    for i in xrange(len(_src)):                
        if res_mtx[i][-1] > max_val:
            max_val = res_mtx[i][-1]
            cur_max_ind = i    
    res_str += _src[cur_max_ind]

    for i in xrange(len(_path)-2,-1,-1):
        next_point = path_points[cur_max_ind][i+1]
        res_str = _src[next_point] + res_str
        cur_max_ind = next_point

    print(res_str)

def task93():
    # Implement the Viterbi algorithm solving the Decoding Problem.
    # Input: A string x, followed by the alphabet from which x was constructed,
    # followed by the states States, transition matrix Transition, and emission matrix
    # Emission of an HMM (sigma, States, Transition, Emission).
    # Output: A path that maximizes the (unconditional) probability Pr(x, pi) over all possible paths pi.

    input_file_name = os.getcwd() + "/part2/data/09/input32.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    path = data[0].replace('\n','')    
    elems_str = data[2].replace('\n','').replace('\t',' ')        
    dest_elems = [e for e in elems_str.split(' ')]

    elems_str = data[4].replace('\n','').replace('\t',' ')        
    src_elems = [e for e in elems_str.split(' ')]
    
    print(path)    
    print(dest_elems)
    print(src_elems)
    
    src_elem_size = len(src_elems)
    transition_mtx = []
    for i in xrange(src_elem_size):
        mtx_str = data[7+i].replace('\n','').replace('\t',' ')        
        mtx_els = [float(i) for i in (mtx_str.split())[1:]]
        transition_mtx.append(mtx_els)

    print(transition_mtx)

    emission_mtx = []
    dest_elem_size = len(dest_elems)
    for i in xrange(src_elem_size):
        ind = 9 + src_elem_size + i        
        mtx_str = data[ind].replace('\n','').replace('\t',' ')        
        mtx_els = [float(i) for i in (mtx_str.split())[1:]]
        emission_mtx.append(mtx_els)


    print(emission_mtx)

    get_optimal_path(transition_mtx,emission_mtx,path,src_elems,dest_elems)    


def likehood(_transition,_emission,_path,_src,_dest):    

    res_mtx = []
    
    src_elem_size = len(_src)
    for i in xrange(src_elem_size):
        res_mtx.append([0 for j in xrange(len(_path))])
    
    for _iter in xrange(len(_path)):        
        path_ind = _dest.index(_path[_iter])
        if _iter == 0:
            res = 0.5            
            for i in xrange(len(_src)):
                res_mtx[i][_iter] = 1*0.5*_emission[i][path_ind]
        else:
            for i in xrange(len(_src)):
                max_val = 0
                max_ind = -1
                for j in xrange(len(_src)):
                    cur_val = res_mtx[j][_iter-1]*_transition[j][i]*_emission[i][path_ind]
                    #if _iter == 4:
                    #    print(_iter,i,j,res_mtx[j][_iter-1],_transition[j][i],_emission[i][path_ind],cur_val)
                    res_mtx[i][_iter] += cur_val
                   
                
    print(res_mtx)
    prob = 0
    for i in xrange(len(_src)):
        prob += res_mtx[i][-1]
    print(prob)
                

def task94():
    # Solve the Outcome Likelihood Problem.
    # Input: A string x, followed by the alphabet from which x was constructed,
    # followed by the states States, transition matrix Transition, and emission matrix
    # Emission of an HMM (sum, States, Transition, Emission).
    # Output: The probability Pr(x) that the HMM emits x.

    input_file_name = os.getcwd() + "/part2/data/09/input41.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    path = data[0].replace('\n','')    
    elems_str = data[2].replace('\n','').replace('\t',' ')        
    dest_elems = [e for e in elems_str.split(' ')]

    elems_str = data[4].replace('\n','').replace('\t',' ')        
    src_elems = [e for e in elems_str.split(' ')]
    
    print(path)    
    print(dest_elems)
    print(src_elems)
    
    src_elem_size = len(src_elems)
    transition_mtx = []
    for i in xrange(src_elem_size):
        mtx_str = data[7+i].replace('\n','').replace('\t',' ')        
        mtx_els = [float(i) for i in (mtx_str.split())[1:]]
        transition_mtx.append(mtx_els)

    print(transition_mtx)

    emission_mtx = []
    dest_elem_size = len(dest_elems)
    for i in xrange(src_elem_size):
        ind = 9 + src_elem_size + i        
        mtx_str = data[ind].replace('\n','').replace('\t',' ')        
        mtx_els = [float(i) for i in (mtx_str.split())[1:]]
        emission_mtx.append(mtx_els)


    print(emission_mtx)
    likehood(transition_mtx,emission_mtx,path,src_elems,dest_elems)    
    
def get_next_vertex_and_increment(_graph,_N,_cur,_type):
    res = -1 
    N = _graph[0][0]
    next_vertices = _graph[2+_cur]
    # 0 - start, 1 - end
    # 2...2+(N-1) - M1...MN
    # 2 + N... 2+2*N - I0...IN
    # 2+ 2*N + 1... 3 + 3*N - 1 - D1...DN
    print('get_next_vertex_and_increment ',_cur,_type)
    for v in next_vertices:
        print(v)
        if _type == 'm':
            if v[0]>=2 and v[0]<=(_N+1):
                v[1] += 1
                return v[0]
        elif _type == 'i':
            if v[0]>=(2+_N) and v[0]<=(2*_N+2):
                v[1] += 1
                return v[0]
        elif _type == 'd':
            if v[0]>=(3+2*_N) and v[0]<=(3*_N+2):
                v[1] += 1
                return v[0]
        elif _type == 'e':
            if v[0]==1:
                v[1] += 1
                return v[0]
    print('err')
    return -1

def get_vertex_ind(_type,_num):
    if _type == 'm':
        if v[0]>=2 and v[0]<=(_N+1):
            v[1] += 1
            return v[0]
    elif _type == 'i':
        if v[0]>=(2+_N) and v[0]<=(2*_N+2):
            v[1] += 1
            return v[0]
    elif _type == 'd':
        if v[0]>=(3+2*_N) and v[0]<=(3*_N+2):
            v[1] += 1
            return v[0]
    elif _type == 'e':
        if v[0]==1:
            v[1] += 1
            return v[0]
    elif _type == 's':
        return 0
    return -1

def task101():
    # Solve the Profile HMM Problem.
    # Input: A threshold theta, followed by an alphabet sum, followed by a multiple alignment
    # Alignment whose strings are formed from sum.
    # Output: The transition matrix followed by the emission matrix of HMM(Alignment, theta).

    input_file_name = os.getcwd() + "/part2/data/10/input1.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    threshold = float(data[0].replace('\n',''))        
    elems_str = data[2].replace('\n','').replace('\t',' ')        
    alphabet = [e for e in elems_str.split(' ')]
    
    strings = []    
    
    transition_mtx = []
    emission_mtx = []
    for d in data[4:]:
        strings.append(d.replace('\n',''))

    print(threshold)    
    print(alphabet)
    print(strings)

    str_len = len(strings[0])
    thresh_indices = []
    for i in xrange(str_len):
        num_in = 0.0
        num_out = 0.0
        for s in strings:
            if s[i] in alphabet:
                num_in += 1
            else:
                num_out += 1
        res = num_out/len(strings)                
        if res > threshold:
            thresh_indices.append(i)
    print(thresh_indices)

    N = str_len - len(thresh_indices)
    v_graph = []
    # 0 - start, 1 - end
    # 2...2+(N-1) - M1...MN
    # 2 + N... 2+2*N - I0...IN
    # 2+ 2*N + 1... 3 + 3*N - 1 - D1...DN
    vertices = {}
    cur_v = 0
    vertices[cur_v] = 'S'
    cur_v += 1
    vertices[cur_v] = 'I0'
    cur_v += 1
    for i in xrange(N):
        vertices[cur_v] = 'M'+str(i+1)
        cur_v += 1
        vertices[cur_v] = 'D'+str(i+1)
        cur_v += 1
        vertices[cur_v] = 'I'+str(i+1)
        cur_v += 1
    vertices[cur_v] = 'E'

    print(vertices)
    
    build_empty_graph(v_graph,3*(N+1))
    add_edge(v_graph,0,2,0)
    add_edge(v_graph,0,2+N,0)    
    add_edge(v_graph,0,3+2*N,0)

    # Mi to...
    for i in xrange(N-1):
        add_edge(v_graph,2+i,3+N+i,0) # to Ii
        add_edge(v_graph,2+i,2+i+1,0) # to M(i+1)
        add_edge(v_graph,2+i,3+2*N + i+1,0) # to D(i+1)

    #Mn to
    add_edge(v_graph,2+N-1,2+2*N,0) # to In    
    add_edge(v_graph,2+N-1,1,0) # to E

    # Ii to
    for i in xrange(N):
        add_edge(v_graph,2+N+i,2+N+i,0) # to I(i)        
        add_edge(v_graph,2+N+i,2+i,0) # to M(i+1)
        add_edge(v_graph,2+N+i,3+2*N+i,0) # to D(i+1)

    #In to
    add_edge(v_graph,2+2*N,2+2*N,0) # to In    
    add_edge(v_graph,2+2*N,1,0) # to E

    #Di to
    for i in xrange(N-1):
        add_edge(v_graph,3+2*N+i,3+N+i,0) # to Ii
        add_edge(v_graph,3+2*N+i,2+i+1,0) # to M(i+1)
        add_edge(v_graph,3+2*N+i,3+2*N+i+1,0) # to D(i+1)

    #Dn to
    add_edge(v_graph,3+3*N-1,2+2*N,0) # to In    
    add_edge(v_graph,3+3*N-1,1,0) # to E

    print('initial state of graph')
    print(v_graph)

    v_stats = [{} for i in xrange(len(vertices))]

    for s in strings:
        cur_vertex = 0
        print(s)
        for i in xrange(len(s)):
            if i in thresh_indices:
                if s[i] in alphabet:
                    cur_vertex = get_next_vertex_and_increment(v_graph,N,cur_vertex,'i')
                    v_stat = v_stats[cur_vertex]
                    cur_val = v_stat.get(s[i],0)
                    v_stat[s[i]] = cur_val + 1
                #else:
                #    cur_vertex = get_next_vertex_and_increment(v_graph,N,cur_vertex,'m')
            else:
                if s[i] in alphabet:
                    cur_vertex = get_next_vertex_and_increment(v_graph,N,cur_vertex,'m')
                    v_stat = v_stats[cur_vertex]
                    cur_val = v_stat.get(s[i],0)
                    v_stat[s[i]] = cur_val + 1
                else:
                    cur_vertex = get_next_vertex_and_increment(v_graph,N,cur_vertex,'d')
        cur_vertex = get_next_vertex_and_increment(v_graph,N,cur_vertex,'e')      


    transtion_row = [0 for i in xrange(len(vertices))]
    for i in xrange(len(vertices)):
        transition_mtx.append(transtion_row[:])

    v_num = v_graph[0][0]
    for i in xrange(v_num):
        vs = v_graph[2+i]
        print(vs)
        _sum = 0
        for v in vs:
            _sum += v[1]
        for v in vs:
            if _sum > 0:
                transition_mtx[i][v[0]] = float(v[1])/_sum

    print(v_stats)

    emission_row = [0 for i in xrange(len(alphabet))]
    for i in xrange(len(vertices)):
        emission_mtx.append(emission_row[:])

    for i in xrange(len(vertices)):
        v_stat = v_stats[i]
        _sum = 0
        for v in v_stat:
            _sum += v_stat[v]

        for v in v_stat:
            ind = alphabet.index(v)
            if _sum > 0:
                emission_mtx[i][ind] = float(v_stat[v])/_sum
    #print transition matrix
    # print transition header
    mtx_str = ' '
    for i in xrange(len(vertices)):
        mtx_str += '\t'+vertices[i]
    print(mtx_str)
    # print transition matrix
    for i in xrange(len(vertices)):
        mtx_str = vertices[i]
        for j in xrange(len(vertices)):
            mtx_str += '\t'+"{0:.3f}".format(transition_mtx[i][j])
        print(mtx_str)    

    print('--------')
    #print emission matrix
    #print header
    mtx_str = ' '
    for i in xrange(len(alphabet)):
        mtx_str += '\t'+alphabet[i]
    print(mtx_str)
    # print transition matrix
    for i in xrange(len(vertices)):
        mtx_str = vertices[i]
        for j in xrange(len(alphabet)):
            mtx_str += '\t'+"{0:.3f}".format(emission_mtx[i][j])
        print(mtx_str)

if __name__ == "__main__":   
    task101() 
    
