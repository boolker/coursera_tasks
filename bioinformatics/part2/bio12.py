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

def get_peptide(_str,_masses):
    acid_mass = get_mass(_masses,_str)
    #print(acid_mass)    

    peptide_vector = [0 for i in xrange(acid_mass)]

    for i in xrange(1,len(_str)+1):
        prefix = _str[:i]
        prefix_mass = get_mass(_masses,prefix)        
        #print(prefix,prefix_mass)
        peptide_vector[prefix_mass-1] = 1

        #postfix = _str[i+1:]
        #postfix_mass = get_mass(_masses,postfix)        
        #peptide_vector[postfix_mass-1] = 1
 
    res_str = ''
    for i in xrange(len(peptide_vector)):
        res_str += str(peptide_vector[i]) + ' '

    #print(res_str)
    return peptide_vector


def task113():
    # Solve the Converting a Peptide into a Peptide Vector Problem.
    # Given: An amino acid string P.
    # Return: The peptide vector of P (in the form of space-separated integers).

    input_file_name = os.getcwd() + "/part2/data/11/input3.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    acid = data[0].replace('\n','')
    print(acid)

    masses = {'A':71,'C':103,'D':115,'E':129,'F':147,'G':57,'H':137,'I':113,'K':128,'L':113,'M':131,'N':114,'P':97,'Q':128,'R':156,'S':87,'T':101,'V':99,'W':186,'Y':163}
    #rev_masses = {71:'A',103:'C',115:'D',129:'E',147:'F',57:'G',137:'H',113:'I',128:'K',113:'L',131:'M',114:'N',97:'P',128:'Q',156:'R',87:'S',101:'T',99:'V',186:'W',163:'Y'}

    #print(rev_masses)    
    masses = {'X':4,'Z':5}
    rev_masses = {4:'X',5:'Z'}

    get_peptide(acid,masses)
    

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
    #print('construct',_vector,_masses)
    build_empty_symbol_graph(_graph,len(_vector))
    for i in xrange(len(_vector)):
        for j in xrange(len(_vector)):
            if i<j:
                delta = j - i         
                letters = _masses.get(delta,[])
                for l in xrange(len(letters)):
                #if len(_masses.get(delta,''))>0:
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

def peptide_identification(_proteome,_spectrum,_masses):
    #print(_proteome,_spectrum)
    max_val = -sys.maxint
    max_str = ''
    max_len = len(_spectrum)/57 +1
    #print(max_len,len(_spectrum))

    for i in xrange(1,max_len):
        #get substring of len i 
        #print(i)              
        for j in xrange(len(_proteome)+1-i):
            #print(i,j,_proteome[j:j+i])
            cur_mass =  get_mass(_masses,_proteome[j:i+j])
            #print(i,j,cur_mass)            
            if cur_mass == len(_spectrum):
                peptide = get_peptide(_proteome[j:i+j],_masses)
                score = sum( [peptide[k]*_spectrum[k] for k in xrange(len(peptide))] )
                if score > max_val:
                    max_val = score
                    max_str = _proteome[j:i+j]

    #s_graph = construct_spectrum_graph(spectrum,rev_masses)
    #print(s_graph)

    #print(get_max_path(s_graph,spectrum,rev_masses))
    #print(max_str,max_val)
    return [max_val,max_str]


def task121():
    # Solve the Peptide Identification Problem.
    # Given: A space-delimited spectral vector Spectrum' and an amino acid string Proteome.
    # Return: A substring of Proteome with maximum score against Spectrum'.

    input_file_name = os.getcwd() + "/part2/data/12/input11.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    spectrum = [int(i) for i in data[0].replace('\n','').split(' ')]
    #spectrum = [0] + spectrum

    proteome = data[1].replace('\n','')

    masses = {'A':71,'C':103,'D':115,'E':129,'F':147,'G':57,'H':137,'I':113,'K':128,'L':113,'M':131,'N':114,'P':97,'Q':128,'R':156,'S':87,'T':101,'V':99,'W':186,'Y':163}
    rev_masses = {71:'A',103:'C',115:'D',129:'E',147:'F',57:'G',137:'H',113:'I',128:'K',113:'L',131:'M',114:'N',97:'P',128:'Q',156:'R',87:'S',101:'T',99:'V',186:'W',163:'Y'}

    #masses = {'X':4,'Z':5}
    #max_len = len(spectrum)/4 +1
    

    rev_masses = {4:'X',5:'Z'}
    
    print(max_len)    
    print(spectrum)    

    peptide_identification(proteome,spectrum,masses)

def task122():
    # Implement PSMSearch to solve the Peptide Search Problem.
    # Given: A set of space-delimited spectral vectors SpectralVectors, an amino acid string
    # Proteome, and an integer threshold.
    # Return: The set PSMthreshold(Proteome, SpectralVectors)

    input_file_name = os.getcwd() + "/part2/data/12/input21.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    
    masses = {'A':71,'C':103,'D':115,'E':129,'F':147,'G':57,'H':137,'I':113,'K':128,'L':113,'M':131,'N':114,'P':97,'Q':128,'R':156,'S':87,'T':101,'V':99,'W':186,'Y':163}
    rev_masses = {71:'A',103:'C',115:'D',129:'E',147:'F',57:'G',137:'H',113:'I',128:'K',113:'L',131:'M',114:'N',97:'P',128:'Q',156:'R',87:'S',101:'T',99:'V',186:'W',163:'Y'}
    #masses = {'X':4,'Z':5}
    spectrums = []
    proteome = ''
    threshold = 0
    for d in data:
        d_ = d.replace('\n','')
        if masses.get(d_[0],0) == 0:
            #not alphabet
            if len(proteome) > 0:
                threshold = int(d_)
            else:
                spectrum = [int(i) for i in d_.split(' ')]
                spectrums.append(spectrum)
        else:
            proteome = d_




    print(spectrums)
    print(proteome)
    print(threshold)

    for s in spectrums:
        res = peptide_identification(proteome,s,masses)
        if res[0] > threshold:
            print(res[1])
    

def spectral_dictionary(_masses,_rev_masses,_spectrum,_min,_max):
    #print(_masses)
    _graph = construct_spectrum_graph(_spectrum,_rev_masses)
    N = _graph[0][0]
    sizes = [{} for i in xrange(N)]
    sizes[0][0] = 1    
    #print(sizes)

    _vals= [[] for i in xrange(N)]
    _vals[0].append([0,-1])
    #print(_vals)

    for i in xrange(N):
        #print(_graph[2+i])
        next_vs = _graph[2+i]
        for _next in next_vs:
            for v in _vals[i]:
                _sum = v[0] + _spectrum[_next[0]]
                old_val = sizes[_next[0]].get(_sum,0)
                #print(v,_next,_sum)
                if _sum <0 :
                    #print('sum < 0')
                    (sizes[_next[0]])[_sum] = 0
                else:
                    (sizes[_next[0]])[_sum] = old_val + 1
                _vals[_next[0]].append([_sum,i])
                #print(v,_next)

    #print(_graph)
    #print(sizes)
    print(sizes[-1])
    last_sizes = sizes[-1]
    _sum = 0
    for s in last_sizes:
        if s >= _min and s< _max:
            _sum += last_sizes[s]
            print(s, last_sizes[s])
    
    print('sum is ',_sum)
    return

def dict_size(_masses,_rev_masses,_spectrum,_cache,_i,_t):
    if _i == 0:
        if _t == 0:
            return 1
        else:
            return 0
    if _i < 0:
        return 0
    if _t < 0:
        return 0
    
    s = 0
    sub_cache = _cache.get(_i,{})
    cache_val = sub_cache.get(_t,None)
    if cache_val != None:
        return cache_val

    for mass in _rev_masses:
        for acid in _rev_masses[mass]:
            #print(_i,_t,mass)
            s += dict_size(_masses,_rev_masses,_spectrum,_cache,_i - mass, _t - _spectrum[_i])
    sub_cache[_t] = s
    _cache[_i] = sub_cache
    return s;

def task123():
    # Solve the Size of Spectral Dictionary Problem.
    # Given: A spectral vector Spectrum', an integer threshold, and an integer max_score.
    # Return: The size of the dictionary Dictionarythreshold(Spectrum').

    input_file_name = os.getcwd() + "/part2/data/12/input31.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    
    masses = {'A':71,'C':103,'D':115,'E':129,'F':147,'G':57,'H':137,'I':113,'K':128,'L':113,'M':131,'N':114,'P':97,'Q':128,'R':156,'S':87,'T':101,'V':99,'W':186,'Y':163}
    #rev_masses = {71:'A',103:'C',115:'D',129:'E',147:'F',57:'G',137:'H',113:'I',128:'K',113:'L',131:'M',114:'N',97:'P',128:'Q',156:'R',87:'S',101:'T',99:'V',186:'W',163:'Y'}
    rev_masses = {71:['A'],103:['C'],115:['D'],129:['E'],147:['F'],57:['G'],137:['H'],113:['I','L'],128:['K','Q'],131:['M'],114:['N'],97:['P'],156:['R'],87:['S'],101:['T'],99:['V'],186:['W'],163:['Y']}
    
    #masses = {'X':4,'Z':5}
    #rev_masses = {4:'X',5:'Z'}
    spectrum = []
    print(masses.keys())
    
    spectrum = [int(i) for i in data[0].replace('\n','').split(' ')]
    spectrum = [0] + spectrum
    threshold = int(data[1].replace('\n',''))
    max_score = int(data[2].replace('\n',''))

    print(spectrum)
    print(threshold,max_score)
    print(len(masses))

    spectral_dictionary(masses,rev_masses,spectrum,threshold,max_score)

    print('new try')
    _sum = 0
    dict_cache = {}
    for i in xrange(threshold,max_score):
        res = dict_size(masses,rev_masses,spectrum,dict_cache,len(spectrum)-1,i)
        print(i,res)
        _sum += res

    print('sum is ',_sum)

def dict_prob(_masses,_rev_masses,_spectrum,_cache,_i,_t):
    if _i == 0:
        if _t == 0:
            return 1
        else:
            return 0
    if _i < 0:
        return 0
    if _t < 0:
        return 0
    
    s = 0
    sub_cache = _cache.get(_i,{})
    cache_val = sub_cache.get(_t,None)
    if cache_val != None:
        return cache_val

    for mass in _rev_masses:
        for acid in _rev_masses[mass]:
            #print(_i,_t,mass)
            s += dict_prob(_masses,_rev_masses,_spectrum,_cache,_i - mass, _t - _spectrum[_i])/20.0
    sub_cache[_t] = s
    _cache[_i] = sub_cache
    return s;

def task124():
    # Solve the Size of Spectral Dictionary Problem.
    # Given: A spectral vector Spectrum', an integer threshold, and an integer max_score.
    # Return: The size of the dictionary Dictionarythreshold(Spectrum').

    input_file_name = os.getcwd() + "/part2/data/12/input41.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    
    masses = {'A':71,'C':103,'D':115,'E':129,'F':147,'G':57,'H':137,'I':113,'K':128,'L':113,'M':131,'N':114,'P':97,'Q':128,'R':156,'S':87,'T':101,'V':99,'W':186,'Y':163}
    #rev_masses = {71:'A',103:'C',115:'D',129:'E',147:'F',57:'G',137:'H',113:'I',128:'K',113:'L',131:'M',114:'N',97:'P',128:'Q',156:'R',87:'S',101:'T',99:'V',186:'W',163:'Y'}
    rev_masses = {71:['A'],103:['C'],115:['D'],129:['E'],147:['F'],57:['G'],137:['H'],113:['I','L'],128:['K','Q'],131:['M'],114:['N'],97:['P'],156:['R'],87:['S'],101:['T'],99:['V'],186:['W'],163:['Y']}
    
    #masses = {'X':4,'Z':5}
    #rev_masses = {4:'X',5:'Z'}
    spectrum = []
    print(masses.keys())
    
    spectrum = [int(i) for i in data[0].replace('\n','').split(' ')]
    spectrum = [0] + spectrum
    threshold = int(data[1].replace('\n',''))
    max_score = int(data[2].replace('\n',''))

    print(spectrum)
    print(threshold,max_score)
    print(len(masses))

    _sum = 0
    dict_cache = {}
    for i in xrange(threshold,max_score):
        res = dict_prob(masses,rev_masses,spectrum,dict_cache,len(spectrum)-1,i)
        print(i,res)
        _sum += res

    print('prob is ',_sum)

def diff(pep_array,_i):
    if _i <=0:
        return 0
    return pep_array[_i]-pep_array[_i-1]

def score(_masses,_spectrum,_pep_array,_cache,_i,_j,_t):
    if _i < 0:
        return [0,0,0]
    if _t < 0:
        return [0,0,0]
    if _j < 0:
        return [0,0,0]
    cache_val = _cache[_t][_i][_j]
    if cache_val[0] > -1*sys.maxint:
        return _cache[_t][_i][_j]

    #print(_t,_i,_j)
    _diff = diff(_pep_array,_i)        
    max_score = [-sys.maxint,-1,-1]
    max_j = -1
    max_t = -1
    if _i > 0 and _j-_diff >=0:
        max_score = score(_masses,_spectrum,_pep_array,_cache,_i-1,_j-_diff,_t)
        max_j = _j-_diff
        max_t = _t
    for jj in xrange(_j):
        if _t > 0:
            cur_score = score(_masses,_spectrum,_pep_array,_cache,_i-1,_j-jj,_t-1)
            if cur_score > max_score:
                max_score = cur_score
                max_t = _t-1
                max_j = _j - jj

    _cache[_t][_i][_j] = [max_score[0] + _spectrum[_j],max_j,max_t]
    return _cache[_t][_i][_j]

def task125():
    # Solve the Spectral Alignment Problem.
    # Given: A peptide Peptide, a spectral vector Spectrum', and an integer k.
    # Return: A peptide Peptide' related to Peptide by up to k modifications with
    # maximal score against Spectrum' out of all possibilities.

    input_file_name = os.getcwd() + "/part2/data/12/input5.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    
    masses = {'A':71,'C':103,'D':115,'E':129,'F':147,'G':57,'H':137,'I':113,'K':128,'L':113,'M':131,'N':114,'P':97,'Q':128,'R':156,'S':87,'T':101,'V':99,'W':186,'Y':163}
    #rev_masses = {71:'A',103:'C',115:'D',129:'E',147:'F',57:'G',137:'H',113:'I',128:'K',113:'L',131:'M',114:'N',97:'P',128:'Q',156:'R',87:'S',101:'T',99:'V',186:'W',163:'Y'}
    rev_masses = {71:['A'],103:['C'],115:['D'],129:['E'],147:['F'],57:['G'],137:['H'],113:['I','L'],128:['K','Q'],131:['M'],114:['N'],97:['P'],156:['R'],87:['S'],101:['T'],99:['V'],186:['W'],163:['Y']}
    
    masses = {'X':4,'Z':5}
    #rev_masses = {4:'X',5:'Z'}
    spectrum = []    
    peptide = data[0].replace('\n','')    
    spectrum = [int(i) for i in data[1].replace('\n','').split(' ')]
    spectrum = [0] + spectrum
    k_modif = int(data[2].replace('\n',''))
    
    print(spectrum)
    print(peptide, k_modif)    

    peptide_v = get_peptide(peptide,masses)
    pep_array = [0]
    for i in xrange(len(peptide_v)):
        if peptide_v[i] != 0:
            pep_array.append(i+1)

    print(peptide_v)
    print(pep_array)
    print(len(peptide_v),len(spectrum))
    score_cache=[]    
    for i in xrange(k_modif+1):
        score_row = [[-sys.maxint,-1,-1] for i in xrange(len(spectrum))]
        #score_row = [0 for i in xrange(len(spectrum))]
        score_layer = []
        for i in xrange(len(peptide)+1):
            score_layer.append(score_row[:])
        score_cache.append(score_layer)
    #print(score_cache)
    score_cache[0][0][0] = [0,0,0]    

    max_path = []
    max_score = [-sys.maxint,-1,-1]
    max_t = -1
    for i in xrange(k_modif+1):
        #print('layer ',i)
        sc = score(masses,spectrum,pep_array,score_cache,len(peptide),len(spectrum)-1,i)
        if sc[0] > max_score[0]:
            max_score = sc
            max_t = i
        #print(sc)
    
    cur_point = [len(peptide),len(spectrum)-1,max_t]
    print('points')
    print(cur_point)
    max_path.append(cur_point)
    
    while cur_point[1]>-1 and cur_point[2]>-1 and cur_point[0] > -1:        
        cache = score_cache[cur_point[2]][cur_point[0]][cur_point[1]]
        print('cache ',cache)
        prev_point = [cur_point[0]-1,cache[1],cache[2]]
        print(prev_point)
        cur_point = prev_point
        max_path.insert(0,cur_point)
    print(max_path)
    '''k=0
    for layer in score_cache:
        print('layer ',k)
        for l in layer:
            print(l)
        k+=1'''

if __name__ == "__main__":   
    task125() 
    
