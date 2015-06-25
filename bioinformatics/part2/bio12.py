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

def build_empty_symbol_graph(_graph, _vnum):
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

def add_symbol_edge(_tree,_start,_end,_symbol):
    #print("add edge from ", _start," to ",_end,"with len = ",_len)    
    N = _tree[0][0]
    _tree[_start+2].append([_end,_symbol])
    _tree[N+_end+2].append([_start])
    _tree[0][1+ _start] += 1
    _tree[1][1+ _end] += 1

def add_edge(_tree,_start,_end,_len):
    #print("add edge from ", _start," to ",_end,"with len = ",_len)    
    N = _tree[0][0]
    _tree[_start+2].append([_end,_len])
    _tree[N+_end+2].append([_start])
    _tree[0][1+ _start] += 1
    _tree[1][1+ _end] += 1
       
def task111():

    input_file_name = os.getcwd() + "/part2/data/11/input11.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    params = [int(i) for i in data[0].replace('\n','').split(' ')]
    params = [0] + params

    masses = {'A':71,'C':103,'D':115,'E':129,'F':147,'G':57,'H':137,'I':113,'K':128,'L':113,'M':131,'N':114,'P':97,'Q':128,'R':156,'S':87,'T':101,'V':99,'W':186,'Y':163}
    rev_masses = {71:'A',103:'C',115:'D',129:'E',147:'F',57:'G',137:'H',113:'I',128:'K',113:'L',131:'M',114:'N',97:'P',128:'Q',156:'R',87:'S',101:'T',99:'V',186:'W',163:'Y'}

    print(rev_masses)
    print(params)

    for i in xrange(len(params)):
        for j in xrange(len(params)):
            if i < j:
                delta = params[j] - params[i]
                symb = rev_masses.get(delta,'')
                if symb != '' :
                    print(params[i],'->',params[j],':',symb)

def get_mass(_masses,_peptide):
    res = 0
    #print(_peptide)
    for i in xrange(len(_peptide)):        
        res += _masses.get(_peptide[i])
    #print(_peptide,res)
    return res

def ideal_spectrum(_masses,_peptide):
    res = [0]
    for i in xrange(len(_peptide)):
        left = get_mass(_masses,_peptide[:i])
        right = get_mass(_masses,_peptide[i:])
        if left not in res:
            res.append(left)
        if right not in res:
            res.append(right)
        #print(_peptide[:i],_peptide[i:],,))
    return sorted(res)

def get_path_list(_graph,_start,_end):    
    _starts = [_start]
    _path_dict = {}    
    while len(_starts) > 0:
        new_starts = []           
        for _st in _starts:            
            prefixes = _path_dict.get(_st,[''])            
            #print(_st, 'prefixes: ',prefixes,len(prefixes))
            ways = _graph[2+_st]        
            for way in ways:
                _peptides = _path_dict.get(way[0],[])
                #print(way[0],_peptides,prefixes,len(prefixes))
                for p in prefixes:
                    #print(_peptides,p,way[1])
                    _peptides.append(p+way[1])
                _path_dict[way[0]] = _peptides
                if way[0] not in new_starts:
                    new_starts.append(way[0])

            if _st != _end:
                _path_dict[_st] = []

        _starts = new_starts[:]
        #print(_path_dict)       
        
    return _path_dict[_end]

def task112():

    input_file_name = os.getcwd() + "/part2/data/11/input21.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    params = [int(i) for i in data[0].replace('\n','').split(' ')]
    params = [0] + params

    masses = {'A':71,'C':103,'D':115,'E':129,'F':147,'G':57,'H':137,'I':113,'K':128,'L':113,'M':131,'N':114,'P':97,'Q':128,'R':156,'S':87,'T':101,'V':99,'W':186,'Y':163}
    rev_masses = {71:'A',103:'C',115:'D',129:'E',147:'F',57:'G',137:'H',113:'I',128:'K',113:'L',131:'M',114:'N',97:'P',128:'Q',156:'R',87:'S',101:'T',99:'V',186:'W',163:'Y'}

    print(rev_masses)
    print(params)

    _graph = []
    build_empty_symbol_graph(_graph, len(params))
    for i in xrange(len(params)):
        for j in xrange(len(params)):
            if i < j:
                delta = params[j] - params[i]
                symb = rev_masses.get(delta,'')
                if symb != '' :
                    add_symbol_edge(_graph,i,j,symb)
                    print(params[i],'->',params[j],':',symb)

    print(params)
    print(_graph)

    path_list = get_path_list(_graph,0,len(params)-1)
    #print(path_list)
    for path in path_list:
        spectrum = ideal_spectrum(masses,path)
        #print(path,spectrum)
        if set(spectrum) == set(params):
            print(path)

def task113():
    # Solve the Converting a Peptide into a Peptide Vector Problem.
    # Given: An amino acid string P.
    # Return: The peptide vector of P (in the form of space-separated integers).

    input_file_name = os.getcwd() + "/part2/data/11/input31.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    acid = data[0].replace('\n','')
    

    masses = {'A':71,'C':103,'D':115,'E':129,'F':147,'G':57,'H':137,'I':113,'K':128,'L':113,'M':131,'N':114,'P':97,'Q':128,'R':156,'S':87,'T':101,'V':99,'W':186,'Y':163}
    #rev_masses = {71:'A',103:'C',115:'D',129:'E',147:'F',57:'G',137:'H',113:'I',128:'K',113:'L',131:'M',114:'N',97:'P',128:'Q',156:'R',87:'S',101:'T',99:'V',186:'W',163:'Y'}

    #print(rev_masses)    

    acid_mass = get_mass(masses,acid)
    print(acid_mass)    

    peptide_vector = [0 for i in xrange(acid_mass)]

    for i in xrange(1,len(acid)+1):
        prefix = acid[:i]
        prefix_mass = get_mass(masses,prefix)        
        #print(prefix,prefix_mass)
        peptide_vector[prefix_mass-1] = 1
 
    res_str = ''
    for i in xrange(len(peptide_vector)):
        res_str += str(peptide_vector[i]) + ' '

    print(res_str)

def task114():
    # Solve the Converting a Peptide Vector into a Peptide Problem.
    # Given: A space-delimited binary vector P
    # Return: An amino acid string whose binary peptide vector matches P. For masses
    # with more than one amino acid, any choice may be used

    input_file_name = os.getcwd() + "/part2/data/11/input41.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()
    
    peptide_vector = [int(i) for i in data[0].replace('\n','').split(' ')]

    masses = {'A':71,'C':103,'D':115,'E':129,'F':147,'G':57,'H':137,'I':113,'K':128,'L':113,'M':131,'N':114,'P':97,'Q':128,'R':156,'S':87,'T':101,'V':99,'W':186,'Y':163}
    rev_masses = {71:'A',103:'C',115:'D',129:'E',147:'F',57:'G',137:'H',113:'I',128:'K',113:'L',131:'M',114:'N',97:'P',128:'Q',156:'R',87:'S',101:'T',99:'V',186:'W',163:'Y'}

    params = [0]
    print(peptide_vector)
    for i in xrange(len(peptide_vector)):
        if peptide_vector[i] == 1:
            params.append(i+1)
    print(params)
    
    _graph = []
    build_empty_symbol_graph(_graph, len(params))
    for i in xrange(len(params)):
        for j in xrange(len(params)):
            if i < j:
                delta = params[j] - params[i]
                symb = rev_masses.get(delta,'')
                if symb != '' :
                    add_symbol_edge(_graph,i,j,symb)
                    print(params[i],'->',params[j],':',symb)

    print(params)
    print(_graph)

    path_list = get_path_list(_graph,0,len(params)-1)
    print(path_list)
    for path in path_list:
        spectrum = ideal_spectrum(masses,path)
        print(path,spectrum)
        if set(spectrum) == set(params):
            print(path)

def construct_spectrum_graph(_vector,_masses):
    _graph = []
    build_empty_symbol_graph(_graph,len(_vector))
    for i in xrange(len(_vector)):
        for j in xrange(len(_vector)):
            if i<j:
                delta = j - i
                if len(_masses.get(delta,''))>0:
                    add_edge(_graph,i,j,0)

    return _graph

def get_max_path(_graph,_weights,_masses):
    res = ''
    N = _graph[0][0]
    max_vals= [[-sys.maxint,-1] for i in xrange(N)]
    max_vals[0] = [0,-1]
    for i in xrange(N):
        next_nodes = _graph[2+i]
        for node in next_nodes:
            cur_val = (max_vals[i])[0] + _weights[node[0]]
            if cur_val > (max_vals[node[0]])[0]:
                max_vals[node[0]] = [cur_val,i]

    print(max_vals)
    cur_node = max_vals[-1]
    cur_pos = len(max_vals)-1
    while cur_node[0] > 0:
        new_pos = cur_node[1]
        delta = cur_pos - new_pos
        new_node = max_vals[cur_node[1]]
        #print(cur_pos, cur_node,new_node)
        
        #print(delta)
        res = _masses[delta] + res
        cur_pos = new_pos
        cur_node = new_node

    return res

def task115():
    # Solve the Peptide Sequencing Problem.
    # Given: A space-delimited spectral vector Spectrum'.
    # Return: An amino acid string with maximum score against Spectrum'. For masses
    # with more than one amino acid, any choice may be used.

    input_file_name = os.getcwd() + "/part2/data/11/input5.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()
    
    spectrum = [int(i) for i in data[0].replace('\n','').split(' ')]
    spectrum = [0] + spectrum

    masses = {'A':71,'C':103,'D':115,'E':129,'F':147,'G':57,'H':137,'I':113,'K':128,'L':113,'M':131,'N':114,'P':97,'Q':128,'R':156,'S':87,'T':101,'V':99,'W':186,'Y':163}
    rev_masses = {71:'A',103:'C',115:'D',129:'E',147:'F',57:'G',137:'H',113:'I',128:'K',113:'L',131:'M',114:'N',97:'P',128:'Q',156:'R',87:'S',101:'T',99:'V',186:'W',163:'Y'}

    print(spectrum)    
    s_graph = construct_spectrum_graph(spectrum,rev_masses)
    #print(s_graph)

    print(get_max_path(s_graph,spectrum,rev_masses))

def task121():
    # Solve the Peptide Identification Problem.
    # Given: A space-delimited spectral vector Spectrum' and an amino acid string Proteome.
    # Return: A substring of Proteome with maximum score against Spectrum'.

    input_file_name = os.getcwd() + "/part2/data/12/input1.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    spectrum = [int(i) for i in data[0].replace('\n','').split(' ')]
    spectrum = [0] + spectrum

    peptide = data[1].replace('\n','')

    masses = {'A':71,'C':103,'D':115,'E':129,'F':147,'G':57,'H':137,'I':113,'K':128,'L':113,'M':131,'N':114,'P':97,'Q':128,'R':156,'S':87,'T':101,'V':99,'W':186,'Y':163}
    rev_masses = {71:'A',103:'C',115:'D',129:'E',147:'F',57:'G',137:'H',113:'I',128:'K',113:'L',131:'M',114:'N',97:'P',128:'Q',156:'R',87:'S',101:'T',99:'V',186:'W',163:'Y'}

    masses = {'X':4,'Z':5}
    rev_masses = {4:'X',5:'Z'}

    print(spectrum)    
    print(peptide)
    

if __name__ == "__main__":   
    task121() 
    
