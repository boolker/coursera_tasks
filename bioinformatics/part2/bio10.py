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
    
def get_next_vertex_and_increment(_graph,_vertices,_cur,_type):
    res = -1 
    N = _graph[0][0]
    next_vertices = _graph[2+_cur]
    # 0 - start, 1 - end
    # 2...2+(N-1) - M1...MN
    # 2 + N... 2+2*N - I0...IN
    # 2+ 2*N + 1... 3 + 3*N - 1 - D1...DN
    #print('get_next_vertex_and_increment ',_cur,_type)
    for v in next_vertices:
        #print(v)
        cur_v = _vertices[v[0]]
        if cur_v[0] == _type:
            v[1] +=1
            return v[0]
        
    print('err')
    return -1

def get_vertex_ind(_type,_num,_N):
    # 0 - start, 1 - end
    # 1 + _num*3 - Inum
    # 2 + (_num-1)*3 - Mnum
    # 3 + (_num-1)*3 - Dnum
    # 1+1+3*_N - end
    if _type == 'M':
        return 2+(_num-1)*3
    elif _type == 'I':
        return 1+(_num)*3
    elif _type == 'D':
        return 3+(_num-1)*3
    elif _type == 'E':
        return 2+_N*3
    elif _type == 'S':
        return 0
    return -1

def task101():
    # Solve the Profile HMM Problem.
    # Input: A threshold theta, followed by an alphabet sum, followed by a multiple alignment
    # Alignment whose strings are formed from sum.
    # Output: The transition matrix followed by the emission matrix of HMM(Alignment, theta).

    input_file_name = os.getcwd() + "/part2/data/10/input11.txt"

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
    print(N)
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
    add_edge(v_graph,0,get_vertex_ind('I',0,N),0)
    add_edge(v_graph,0,get_vertex_ind('M',1,N),0)
    add_edge(v_graph,0,get_vertex_ind('D',1,N),0)

    # Mi to...
    for i in xrange(N-1):
        add_edge(v_graph,get_vertex_ind('M',i+1,N),get_vertex_ind('I',i+1,N),0) # to Ii
        add_edge(v_graph,get_vertex_ind('M',i+1,N),get_vertex_ind('M',i+2,N),0) # to M(i+1)
        add_edge(v_graph,get_vertex_ind('M',i+1,N),get_vertex_ind('D',i+2,N),0) # to D(i+1)

    #Mn to
    add_edge(v_graph,get_vertex_ind('M',N,N),get_vertex_ind('I',N,N),0) # to In    
    add_edge(v_graph,get_vertex_ind('M',N,N),get_vertex_ind('E',0,N),0) # to E

    # Ii to
    for i in xrange(N):
        add_edge(v_graph,get_vertex_ind('I',i,N),get_vertex_ind('I',i,N),0) # to I(i)        
        add_edge(v_graph,get_vertex_ind('I',i,N),get_vertex_ind('M',i+1,N),0) # to M(i+1)
        add_edge(v_graph,get_vertex_ind('I',i,N),get_vertex_ind('D',i+1,N),0) # to D(i+1)

    #In to
    add_edge(v_graph,get_vertex_ind('I',N,N),get_vertex_ind('I',N,N),0) # to In    
    add_edge(v_graph,get_vertex_ind('I',N,N),get_vertex_ind('E',0,N),0) # to E

    #Di to
    for i in xrange(N-1):
        add_edge(v_graph,get_vertex_ind('D',i+1,N),get_vertex_ind('I',i+1,N),0) # to Ii
        add_edge(v_graph,get_vertex_ind('D',i+1,N),get_vertex_ind('M',i+2,N),0) # to M(i+1)
        add_edge(v_graph,get_vertex_ind('D',i+1,N),get_vertex_ind('D',i+2,N),0) # to D(i+1)

    #Dn to
    add_edge(v_graph,get_vertex_ind('D',N,N),get_vertex_ind('I',N,N),0) # to In    
    add_edge(v_graph,get_vertex_ind('D',N,N),get_vertex_ind('E',0,N),0) # to E

    print('initial state of graph')
    print(v_graph)
    #return

    v_stats = [{} for i in xrange(len(vertices))]

    for s in strings:
        cur_vertex = 0
        print(s)
        for i in xrange(len(s)):
            if i in thresh_indices:
                if s[i] in alphabet:
                    cur_vertex = get_next_vertex_and_increment(v_graph,vertices,cur_vertex,'I')
                    v_stat = v_stats[cur_vertex]
                    cur_val = v_stat.get(s[i],0)
                    v_stat[s[i]] = cur_val + 1
                #else:
                #    cur_vertex = get_next_vertex_and_increment(v_graph,N,cur_vertex,'m')
            else:
                if s[i] in alphabet:
                    cur_vertex = get_next_vertex_and_increment(v_graph,vertices,cur_vertex,'M')
                    print(cur_vertex)
                    v_stat = v_stats[cur_vertex]
                    cur_val = v_stat.get(s[i],0)
                    v_stat[s[i]] = cur_val + 1
                else:
                    cur_vertex = get_next_vertex_and_increment(v_graph,vertices,cur_vertex,'D')
        cur_vertex = get_next_vertex_and_increment(v_graph,vertices,cur_vertex,'E')      


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
        
def task102():
    # Solve the Profile HMM Problem.
    # Input: A threshold theta, followed by an alphabet sum, followed by a multiple alignment
    # Alignment whose strings are formed from sum.
    # Output: The transition matrix followed by the emission matrix of HMM(Alignment, theta).

    input_file_name = os.getcwd() + "/part2/data/10/input21.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    params = data[0].replace('\n','').split(' ')
    threshold = float(params[0])
    pseudocount = float(params[1])    
    print(threshold,pseudocount)    
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
    print(N)
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
    add_edge(v_graph,0,get_vertex_ind('I',0,N),0)
    add_edge(v_graph,0,get_vertex_ind('M',1,N),0)
    add_edge(v_graph,0,get_vertex_ind('D',1,N),0)

    # Mi to...
    for i in xrange(N-1):
        add_edge(v_graph,get_vertex_ind('M',i+1,N),get_vertex_ind('I',i+1,N),0) # to Ii
        add_edge(v_graph,get_vertex_ind('M',i+1,N),get_vertex_ind('M',i+2,N),0) # to M(i+1)
        add_edge(v_graph,get_vertex_ind('M',i+1,N),get_vertex_ind('D',i+2,N),0) # to D(i+1)

    #Mn to
    add_edge(v_graph,get_vertex_ind('M',N,N),get_vertex_ind('I',N,N),0) # to In    
    add_edge(v_graph,get_vertex_ind('M',N,N),get_vertex_ind('E',0,N),0) # to E

    # Ii to
    for i in xrange(N):
        add_edge(v_graph,get_vertex_ind('I',i,N),get_vertex_ind('I',i,N),0) # to I(i)        
        add_edge(v_graph,get_vertex_ind('I',i,N),get_vertex_ind('M',i+1,N),0) # to M(i+1)
        add_edge(v_graph,get_vertex_ind('I',i,N),get_vertex_ind('D',i+1,N),0) # to D(i+1)

    #In to
    add_edge(v_graph,get_vertex_ind('I',N,N),get_vertex_ind('I',N,N),0) # to In    
    add_edge(v_graph,get_vertex_ind('I',N,N),get_vertex_ind('E',0,N),0) # to E

    #Di to
    for i in xrange(N-1):
        add_edge(v_graph,get_vertex_ind('D',i+1,N),get_vertex_ind('I',i+1,N),0) # to Ii
        add_edge(v_graph,get_vertex_ind('D',i+1,N),get_vertex_ind('M',i+2,N),0) # to M(i+1)
        add_edge(v_graph,get_vertex_ind('D',i+1,N),get_vertex_ind('D',i+2,N),0) # to D(i+1)

    #Dn to
    add_edge(v_graph,get_vertex_ind('D',N,N),get_vertex_ind('I',N,N),0) # to In    
    add_edge(v_graph,get_vertex_ind('D',N,N),get_vertex_ind('E',0,N),0) # to E

    print('initial state of graph')
    print(v_graph)
    #return

    v_stats = [{} for i in xrange(len(vertices))]

    for s in strings:
        cur_vertex = 0
        print(s)
        for i in xrange(len(s)):
            if i in thresh_indices:
                if s[i] in alphabet:
                    cur_vertex = get_next_vertex_and_increment(v_graph,vertices,cur_vertex,'I')
                    v_stat = v_stats[cur_vertex]
                    cur_val = v_stat.get(s[i],0)
                    v_stat[s[i]] = cur_val + 1
                #else:
                #    cur_vertex = get_next_vertex_and_increment(v_graph,N,cur_vertex,'m')
            else:
                if s[i] in alphabet:
                    cur_vertex = get_next_vertex_and_increment(v_graph,vertices,cur_vertex,'M')
                    print(cur_vertex)
                    v_stat = v_stats[cur_vertex]
                    cur_val = v_stat.get(s[i],0)
                    v_stat[s[i]] = cur_val + 1
                else:
                    cur_vertex = get_next_vertex_and_increment(v_graph,vertices,cur_vertex,'D')
        cur_vertex = get_next_vertex_and_increment(v_graph,vertices,cur_vertex,'E')      


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
        _sum = 0
        for v in vs:
            _sum += transition_mtx[i][v[0]] + pseudocount
        for v in vs:
            transition_mtx[i][v[0]] = float(transition_mtx[i][v[0]]+pseudocount)/_sum

    print(v_stats)

    emission_row = [0 for i in xrange(len(alphabet))]
    for i in xrange(len(vertices)):
        emission_mtx.append(emission_row[:])

    for i in xrange(len(vertices)):
        v_stat = v_stats[i]
        print(i, v_stat)
        _sum = 0
        for v in v_stat:
            _sum += v_stat[v]

        for v in v_stat:
            ind = alphabet.index(v)
            if _sum > 0:
                emission_mtx[i][ind] = float(v_stat[v])/_sum
        vertex = vertices[i]
        if vertex[0] == 'M' or vertex[0] == 'I':
            _sum = 0
            for a in alphabet:
                ind = alphabet.index(a)
                _sum += emission_mtx[i][ind] + pseudocount
            for a in alphabet:
                ind = alphabet.index(a)
                emission_mtx[i][ind] = float(emission_mtx[i][ind]+pseudocount)/_sum
            
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


def get_optimal_path_ex(_transition,_emission,_path,_src,_dest):    

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

def get_vertex_ind_ex(_type,_num,_N,_iter,_start,_cols):
    # 0 - start, 1 - end
    # 1 + _num*3 - Inum
    # 2 + (_num-1)*3 - Mnum
    # 3 + (_num-1)*3 - Dnum
    # 1+1+3*_N - end
    res = 0
    if _type == 'M':
        res = 1+(_num-1)*3
    elif _type == 'I':
        res =  (_num)*3
    elif _type == 'D':
        if _iter == 0:
            return _num
        res =  2+(_num-1)*3
    elif _type == 'E':
        return _start + _cols*_N 
    elif _type == 'S':
        return 0 
    res += (_iter-1)*_N + _start
    return res

def task103():
    # Solve the Sequence Alignment with Profile HMM Problem.
    # Input: A string x followed by a threshold  and a pseudocount , followed by an
    # alphabet sigma, followed by a multiple alignment Alignment whose strings are formed from sigma. 
    # Output: An optimal hidden path emitting x in HMM(Alignment, thet, sig).

    input_file_name = os.getcwd() + "/part2/data/10/input3.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    path = data[0].replace('\n','')
    params = data[2].replace('\n','').split(' ')
    threshold = float(params[0])
    pseudocount = float(params[1])    
    print(threshold,pseudocount)    
    elems_str = data[4].replace('\n','').replace('\t',' ')        
    alphabet = [e for e in elems_str.split(' ')]
    
    strings = []    
    
    transition_mtx = []
    emission_mtx = []
    for d in data[6:]:
        strings.append(d.replace('\n',''))

    print(path)
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
        if res >= threshold:
            thresh_indices.append(i)
    print(thresh_indices)

    N = str_len - len(thresh_indices)
    print(N)
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
    add_edge(v_graph,0,get_vertex_ind('I',0,N),0)
    add_edge(v_graph,0,get_vertex_ind('M',1,N),0)
    add_edge(v_graph,0,get_vertex_ind('D',1,N),0)

    # Mi to...
    for i in xrange(N-1):
        add_edge(v_graph,get_vertex_ind('M',i+1,N),get_vertex_ind('I',i+1,N),0) # to Ii
        add_edge(v_graph,get_vertex_ind('M',i+1,N),get_vertex_ind('M',i+2,N),0) # to M(i+1)
        add_edge(v_graph,get_vertex_ind('M',i+1,N),get_vertex_ind('D',i+2,N),0) # to D(i+1)

    #Mn to
    add_edge(v_graph,get_vertex_ind('M',N,N),get_vertex_ind('I',N,N),0) # to In    
    add_edge(v_graph,get_vertex_ind('M',N,N),get_vertex_ind('E',0,N),0) # to E

    # Ii to
    for i in xrange(N):
        add_edge(v_graph,get_vertex_ind('I',i,N),get_vertex_ind('I',i,N),0) # to I(i)        
        add_edge(v_graph,get_vertex_ind('I',i,N),get_vertex_ind('M',i+1,N),0) # to M(i+1)
        add_edge(v_graph,get_vertex_ind('I',i,N),get_vertex_ind('D',i+1,N),0) # to D(i+1)

    #In to
    add_edge(v_graph,get_vertex_ind('I',N,N),get_vertex_ind('I',N,N),0) # to In    
    add_edge(v_graph,get_vertex_ind('I',N,N),get_vertex_ind('E',0,N),0) # to E

    #Di to
    for i in xrange(N-1):
        add_edge(v_graph,get_vertex_ind('D',i+1,N),get_vertex_ind('I',i+1,N),0) # to Ii
        add_edge(v_graph,get_vertex_ind('D',i+1,N),get_vertex_ind('M',i+2,N),0) # to M(i+1)
        add_edge(v_graph,get_vertex_ind('D',i+1,N),get_vertex_ind('D',i+2,N),0) # to D(i+1)

    #Dn to
    add_edge(v_graph,get_vertex_ind('D',N,N),get_vertex_ind('I',N,N),0) # to In    
    add_edge(v_graph,get_vertex_ind('D',N,N),get_vertex_ind('E',0,N),0) # to E

    v_stats = [{} for i in xrange(len(vertices))]

    for s in strings:
        cur_vertex = 0
        #print(s)
        for i in xrange(len(s)):
            if i in thresh_indices:
                if s[i] in alphabet:
                    cur_vertex = get_next_vertex_and_increment(v_graph,vertices,cur_vertex,'I')
                    v_stat = v_stats[cur_vertex]
                    cur_val = v_stat.get(s[i],0)
                    v_stat[s[i]] = cur_val + 1
                #else:
                #    cur_vertex = get_next_vertex_and_increment(v_graph,N,cur_vertex,'m')
            else:
                if s[i] in alphabet:
                    cur_vertex = get_next_vertex_and_increment(v_graph,vertices,cur_vertex,'M')
                    v_stat = v_stats[cur_vertex]
                    cur_val = v_stat.get(s[i],0)
                    v_stat[s[i]] = cur_val + 1
                else:
                    cur_vertex = get_next_vertex_and_increment(v_graph,vertices,cur_vertex,'D')
        cur_vertex = get_next_vertex_and_increment(v_graph,vertices,cur_vertex,'E')      


    transtion_row = [0 for i in xrange(len(vertices))]
    for i in xrange(len(vertices)):
        transition_mtx.append(transtion_row[:])

    v_num = v_graph[0][0]
    for i in xrange(v_num):
        vs = v_graph[2+i]
        _sum = 0
        for v in vs:
            _sum += v[1]

        for v in vs:
            if _sum > 0:
                transition_mtx[i][v[0]] = float(v[1])/_sum
        _sum = 0
        for v in vs:
            _sum += transition_mtx[i][v[0]] + pseudocount
        for v in vs:
            transition_mtx[i][v[0]] = float(transition_mtx[i][v[0]]+pseudocount)/_sum

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
        vertex = vertices[i]
        if vertex[0] == 'M' or vertex[0] == 'I':
            _sum = 0
            for a in alphabet:
                ind = alphabet.index(a)
                _sum += emission_mtx[i][ind] + pseudocount
            for a in alphabet:
                ind = alphabet.index(a)
                emission_mtx[i][ind] = float(emission_mtx[i][ind]+pseudocount)/_sum
            
    #print(v_graph)
    vertices_elems = []
    for v in vertices:
        vertices_elems.append(vertices[v])

    '''N = v_graph[0][0]
    for i in xrange(N):
        v_s = v_graph[2+i]
        for v in v_s:
            v[1] = transition_mtx[i][v[0]]
    print(v_graph)'''    

    vit_graph = []
    v_count = len(path)*(v_num-2) + 2 + N #len(path) columns * N-2 lines + start + end + silent column
    print(len(path),v_num,N,v_count)
    build_empty_graph(vit_graph,v_count) # 
    #fill graph

    map_vertices = [0]
    for i in xrange(N):
        map_vertices.append(get_vertex_ind('D',i+1,N))
    for i in xrange(len(path)):
        for j in xrange(1,v_num-1):
            map_vertices.append(j)
    map_vertices.append(get_vertex_ind('E',0,N))
    
    print(map_vertices)
    print(len(map_vertices))
    _iter = 1
    for i in xrange(N+1):
        if i < N:
            #to Di+1
            j = get_vertex_ind_ex('D',i+1,v_num,0,N+1,N)
            real_i = map_vertices[i]
            real_j = map_vertices[j]
            print('D',i,j,real_i,real_j)        
            add_edge(vit_graph,i,j,transition_mtx[real_i][real_j])
        
            #to Mi+1
            j = get_vertex_ind_ex('M',i+1,v_num,_iter,N+1,N)
            real_i = map_vertices[i]
            real_j = map_vertices[j]
            print('M',i,j,real_i,real_j)        
            add_edge(vit_graph,i,j,transition_mtx[real_i][real_j])

            #to Ii
            j = get_vertex_ind_ex('I',i,v_num,_iter,N+1,N)
            real_i = map_vertices[i]
            real_j = map_vertices[j]
            print('I',i,j,real_i,real_j)
            add_edge(vit_graph,i,j,transition_mtx[real_i][real_j])
        
        if i == N:
            #to Ii
            j = get_vertex_ind_ex('I',i,v_num,_iter,N+1,N)
            real_i = map_vertices[i]
            real_j = map_vertices[j]
            print('I',i,j,real_i,real_j)
            add_edge(vit_graph,i,j,transition_mtx[real_i][real_j])

    states_num = v_num - 2
    for j in xrange(len(path)-1):
        _iter = j+1
         # Mi to...
        for i in xrange(N-1):
            _i = get_vertex_ind_ex('M',i+1,states_num,_iter,N+1,N)
            _j = get_vertex_ind_ex('I',i+1,states_num,_iter+1,N+1,N)
            real_i = map_vertices[_i]
            real_j = map_vertices[_j]
            print('iter ',_iter,', M',i+1,' to I',i+1, _i,_j,real_i,real_j)
            add_edge(vit_graph,i,j,transition_mtx[real_i][real_j]) # to Ii next col

            _i = get_vertex_ind_ex('M',i+1,states_num,_iter,N+1,N)
            _j = get_vertex_ind_ex('M',i+2,states_num,_iter+1,N+1,N)
            real_i = map_vertices[_i]
            real_j = map_vertices[_j]
            print('iter ',_iter,', M',i+1,' to M',i+2, _i,_j,real_i,real_j)
            add_edge(vit_graph,i,j,transition_mtx[real_i][real_j]) # to Mi+1 next col

            _i = get_vertex_ind_ex('M',i+1,states_num,_iter,N+1,N)
            _j = get_vertex_ind_ex('D',i+2,states_num,_iter,N+1,N)
            real_i = map_vertices[_i]
            real_j = map_vertices[_j]
            print('iter ',_iter,', M',i+1,' to D',i+1, _i,_j,real_i,real_j)
            add_edge(vit_graph,i,j,transition_mtx[real_i][real_j]) # to Di+1 cur col

        break

        #Mn to
        add_edge(v_graph,get_vertex_ind('M',N,N),get_vertex_ind('I',N,N),0) # to In    
        add_edge(v_graph,get_vertex_ind('M',N,N),get_vertex_ind('E',0,N),0) # to E

        # Ii to
        for i in xrange(N):
            add_edge(v_graph,get_vertex_ind('I',i,N),get_vertex_ind('I',i,N),0) # to I(i)        
            add_edge(v_graph,get_vertex_ind('I',i,N),get_vertex_ind('M',i+1,N),0) # to M(i+1)
            add_edge(v_graph,get_vertex_ind('I',i,N),get_vertex_ind('D',i+1,N),0) # to D(i+1)

        #In to
        add_edge(v_graph,get_vertex_ind('I',N,N),get_vertex_ind('I',N,N),0) # to In    
        add_edge(v_graph,get_vertex_ind('I',N,N),get_vertex_ind('E',0,N),0) # to E

        #Di to
        for i in xrange(N-1):
            add_edge(v_graph,get_vertex_ind('D',i+1,N),get_vertex_ind('I',i+1,N),0) # to Ii
            add_edge(v_graph,get_vertex_ind('D',i+1,N),get_vertex_ind('M',i+2,N),0) # to M(i+1)
            add_edge(v_graph,get_vertex_ind('D',i+1,N),get_vertex_ind('D',i+2,N),0) # to D(i+1)

        #Dn to
        add_edge(v_graph,get_vertex_ind('D',N,N),get_vertex_ind('I',N,N),0) # to In    
        add_edge(v_graph,get_vertex_ind('D',N,N),get_vertex_ind('E',0,N),0) # to E

    #for cur_col in xrange(len(path)):
    #print(vit_graph)    
    #add_edge(v_graph,0,,0)
    #add_edge(v_graph,0,get_vertex_ind('M',1,N),0)
    #add_edge(v_graph,0,get_vertex_ind('D',1,N),0)
    #get_optimal_path_ex(transition_mtx,emission_mtx,path,vertices_elems,alphabet)

if __name__ == "__main__":   
    task103() 
    
