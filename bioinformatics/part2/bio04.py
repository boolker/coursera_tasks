from __future__ import print_function
import sys, os
__author__ = 'drummer'

def get_coins(_sum, _coins, _counted_coins):
    if _sum <=0:
        return -1
    coins_num = _sum
    cur_coins = 0
    if _counted_coins[_sum] >=0:
        return _counted_coins[_sum]
    for c in _coins:
        delta = _sum - c
        if delta >=0:
            cur_coins = get_coins(delta,_coins,_counted_coins) + 1
            if cur_coins < coins_num:
                coins_num = cur_coins
    _counted_coins[_sum] = coins_num
    return coins_num

def task61():
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\06\\input11.txt", "r") as myfile:
        data=myfile.readlines()

    sum = int(data[0].replace('\n',''))

    coins_str = data[1].replace('\n','').split(',')
    coins = [int(i) for i in coins_str]
    print(coins_str)

    counted_coins = [-1]*(sum+1)
    for i in xrange(sum+1):
        get_coins(i, coins, counted_coins)

    result = get_coins(sum, coins, counted_coins) + 1
    #print(counted_coins)
    print(result)

'''def build_graph_by_matrices(down_m, right_m):
    graph = []
    graph_row = [0]*len(down_m[0])
    graph_rows = len(right_m[0])
    graph_cols = len(down_m[0])
    for i in xrange(graph_rows):
        graph.append([]+graph_row)

    for i in xrange(graph_rows):
        for j in xrange(len(right_m[i])):
            graph[i][j+1] = right_m[i][j]

    for i in xrange(graph_cols):
        for j in xrange(len(down_m[i])):
            graph[i][j+1] = down_m[i][j]

    for j in xrange(graph_cols):
            graph[i][j]
    return graph'''

def south_or_east(_i,_j, _graph, _dirs, _down_matrix, _right_matrix):
    if _i == 0 and _j == 0:
        return 0

    if _graph[_i][_j] > (-sys.maxint - 1):
        return _graph[_i][_j]

    x = -sys.maxint - 1
    y = -sys.maxint - 1
    if _i > 0:
        x = south_or_east(_i -1,_j, _graph,_dirs,_down_matrix,_right_matrix) + _down_matrix[_i-1][_j]
    if _j > 0:
        y = south_or_east(_i ,_j-1, _graph,_dirs,_down_matrix,_right_matrix) + _right_matrix[_i][_j-1]
    max_val = x
    if y > x:
        max_val = y
    _graph[_i][_j] = max_val
    return max_val

def task62():
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\06\\input21.txt", "r") as myfile:
        data=myfile.readlines()

    n = int(data[0].replace('\n',''))
    m = int(data[1].replace('\n',''))

    down_matrix = []
    right_matrix = []
    is_right_matrix = 0
    for d in data[2:]:
        data_str = d.replace('\n','')
        if data_str == '-':
            is_right_matrix = 1
            continue
        if is_right_matrix == 0:
            down_matrix.append([int(i) for i in d.split(' ')])
        else:
            right_matrix.append([int(i) for i in d.split(' ')])
    print(right_matrix)
    print(down_matrix)

    graph = []
    graph_row = [-sys.maxint - 1]*len(down_matrix[0])
    graph_rows = len(right_matrix)
    graph_cols = len(down_matrix[0])
    for i in xrange(graph_rows):
        graph.append([]+graph_row)

    directions = []
    directions_row = [0]*len(down_matrix[0])
    directions_rows = len(right_matrix)
    directions_cols = len(down_matrix[0])
    for i in xrange(graph_rows):
        directions.append([]+directions_row)


    graph[0][0] = 0
    max_val = south_or_east(graph_rows-1,graph_cols-1,graph,directions,down_matrix,right_matrix)
    print(max_val)
    print(graph)

def south_or_east_or_diag(_i,_j, _graph, _down_matrix, _right_matrix, _diag_matrix, _backtrack):
    if _i == 0 and _j == 0:
        return 0

    if _graph[_i][_j] > (-sys.maxint - 1):
        return _graph[_i][_j]

    x = -sys.maxint - 1
    y = -sys.maxint - 1
    z = -sys.maxint - 1
    if _i > 0:
        val = _down_matrix[_i-1][_j]
        x = south_or_east_or_diag(_i -1,_j, _graph, _down_matrix, _right_matrix, _diag_matrix, _backtrack) + val
    if _j > 0:
        val = _right_matrix[_i][_j-1]
        y = south_or_east_or_diag(_i ,_j-1, _graph, _down_matrix, _right_matrix, _diag_matrix, _backtrack) + val
    if _i > 0 and _j > 0:
        z = south_or_east_or_diag(_i-1 ,_j-1, _graph, _down_matrix, _right_matrix, _diag_matrix, _backtrack) + _diag_matrix[_i-1][_j-1]
    max_val = x
    _backtrack[_i][_j] = 0
    if y > max_val:
        max_val = y
        _backtrack[_i][_j] = 1
    if z > max_val:
        max_val = z
        _backtrack[_i][_j] = 2
    _graph[_i][_j] = max_val
    #print(_i,_j,_graph[_i][_j],_backtrack[_i][_j])
    return max_val

def get_backtrack(_i,_j,_backtrack, word):
    res_str = ''

    cur_i = _i
    cur_j = _j
    while 1:
        if cur_i == 0 and cur_j == 0:
            break
        if _backtrack[cur_i][cur_j] == 0:
            cur_i = cur_i - 1
            cur_j = cur_j
            #res_str = get_backtrack(cur_i - 1, cur_j, _backtrack, word)
        elif _backtrack[cur_i][cur_j] == 1:
            cur_i = cur_i
            cur_j = cur_j - 1
            #res_str = get_backtrack(_i , _j -1, _backtrack, word)
        elif _backtrack[cur_i][cur_j] == 2:
            res_str = word[cur_i - 1] + res_str
            cur_i = cur_i - 1
            cur_j = cur_j - 1
            #res_str = get_backtrack(_i - 1, _j - 1, _backtrack, word) + word[_i-1]
    return res_str

def task63():
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\06\\input31.txt", "r") as myfile:
        data=myfile.readlines()

    string_a = data[0].replace('\n','')
    string_b = data[1].replace('\n','')

    letters_a = len(string_a)
    letters_b = len(string_b)

    down_matrix = []
    down_row = [0]*(letters_b+1)
    for i in xrange(letters_a):
        down_matrix.append([]+down_row)

    print(len(down_matrix))
    print(len(down_row))

    right_matrix = []
    right_row = [0]*(letters_b)
    for i in xrange(letters_a+1):
        right_matrix.append([]+right_row)

    print(len(right_matrix))
    print(len(right_row))


    diag_matrix = []
    diag_row = [0]*(letters_b)
    for i in xrange(letters_a):
        diag_matrix.append([]+diag_row)

    for i in xrange(letters_a):
        for j in xrange(letters_b):
            if string_a[i] == string_b[j]:
                diag_matrix[i][j] = 1

    graph = []
    graph_rows = letters_a + 1
    graph_cols = letters_b + 1
    graph_row = [-sys.maxint - 1]*graph_cols
    print(graph_rows,graph_cols)
    for i in xrange(graph_rows):
        graph.append([]+graph_row)

    backtrack = []
    backtrack_row = [-sys.maxint - 1]*graph_cols
    backtrack_rows = letters_a + 1
    backtrack_cols = len(right_row)
    for i in xrange(backtrack_rows):
        backtrack.append([]+backtrack_row)


    graph[0][0] = 0
    for i in xrange(graph_rows):
        for j in xrange(graph_cols):
            south_or_east_or_diag(i,j,graph,down_matrix,right_matrix,diag_matrix,backtrack)
    max_val = south_or_east_or_diag(graph_rows-1,graph_cols-1,graph,down_matrix,right_matrix,diag_matrix,backtrack)
    print(max_val)
    #print(graph)
    print(get_backtrack(graph_rows-1,graph_cols-1,backtrack,string_a))

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



def task64():
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\06\\input41.txt", "r") as myfile:
        data=myfile.readlines()

    start_node = int(data[0].replace('\n',''))
    end_node = int(data[1].replace('\n',''))

    pairs = []
    weights = []
    for d in data[2:]:
        arch_info = d.replace('\n','').split(':')
        arch_str = arch_info[0]
        pair = [int(i) for i in arch_str.split('->')]
        arch_weight = int(arch_info[1])
        pairs.append(pair)
        weights.append(arch_weight)

    print(start_node,end_node)
    print(pairs)
    print(weights)

    graph = build_graph_by_pure_pairs(pairs,weights)
    print(graph)

    graph_size = graph[0][0]
    path_graph = []
    path_row = [-sys.maxint - 1]*graph_size
    for i in xrange(graph_size):
        path_graph.append([] + path_row)
    backtrack = []
    max_val = get_longest_path_val(start_node,end_node,graph,path_graph)
    print(max_val)
    path = get_longest_path(start_node,end_node,graph,path_graph)
    print(path)
    res_str = ''
    for p in path:
        res_str += str(p)
        res_str += '->'
    print(res_str)

def get_weight(letter_a, letter_b, _vocab, _b_matrix):
    result = 0
    ind_a = _vocab.index(letter_a)
    ind_b = _vocab.index(letter_b)
    result = _b_matrix[ind_a][ind_b]
    return result

def build_letter_graph(str_a,str_b,_vocab,_b_matrix,_mu):
    letters_a = len(str_a)
    letters_b = len(str_b)

    M = letters_a + 1
    N = letters_b + 1

    max_node = M * N

    graph = []
    rev_graph = []

    graph_header = [0]*(max_node+1)
    graph.append([] + graph_header)
    graph.append([] + graph_header)
    graph_row = []
    for i in xrange(2*max_node):
        graph.append([] + graph_row)

    graph[0][0] = max_node
    for i in xrange(M):
        for j in xrange(N):
            cur_i = i + j*M

            #down
            if i < M-1:
                cur_w = _mu
                #if j < N - 1:
                #    cur_w  = 0 #get_weight(str_a[i],str_b[j],_vocab, _b_matrix)
                #else:
                #    cur_w = 0
                cur_j = cur_i + 1
                graph[2+cur_i].append([cur_j,cur_w])
                graph[max_node+cur_j+2].append([cur_i])
                graph[0][1+ cur_i] += 1
                graph[1][1+ cur_j] += 1

            #right
            if j < N-1:
                cur_w = _mu
                #if i < M - 1:
                #    cur_w  = 0 #get_weight(str_a[i],str_b[j],_vocab, _b_matrix)
                #else:
                #    cur_w = 0

                cur_j = cur_i + M
                graph[2+cur_i].append([cur_j,cur_w])
                graph[max_node+cur_j+2].append([cur_i])
                graph[0][1+ cur_i] += 1
                graph[1][1+ cur_j] += 1

            #diag
            if i < M - 1 and j < N - 1:
                cur_w  = get_weight(str_a[i],str_b[j],_vocab, _b_matrix)
                cur_j = cur_i + M + 1
                graph[2+cur_i].append([cur_j,cur_w])
                graph[max_node+cur_j+2].append([cur_i])
                graph[0][1+ cur_i] += 1
                graph[1][1+ cur_j] += 1

            #graph[cur_i+2].append([cur_j,weights[l]])
            #graph[N+cur_j+2].append([cur_i])
            #graph[0][1+ cur_i] += 1
            #graph[1][1+ cur_j] += 1
    return graph

def build_letter_graph_ex(str_a,str_b,_vocab,_b_matrix,_mu):
    letters_a = len(str_a)
    letters_b = len(str_b)

    M = letters_a + 1
    N = letters_b + 1

    max_node = M * N

    graph = []
    rev_graph = []

    graph_header = [0]*(max_node+1)
    graph.append([] + graph_header)
    graph.append([] + graph_header)
    graph_row = []
    for i in xrange(2*max_node):
        graph.append([] + graph_row)

    graph[0][0] = max_node
    for i in xrange(M):
        for j in xrange(N):
            cur_i = i + j*M

            #down
            if i < M-1:
                cur_w = _mu
                cur_j = cur_i + 1
                graph[2+cur_i].append([cur_j,cur_w])
                graph[max_node+cur_j+2].append([cur_i])
                graph[0][1+ cur_i] += 1
                graph[1][1+ cur_j] += 1

            #right
            if j < N-1:
                cur_w = _mu
                cur_j = cur_i + M
                graph[2+cur_i].append([cur_j,cur_w])
                graph[max_node+cur_j+2].append([cur_i])
                graph[0][1+ cur_i] += 1
                graph[1][1+ cur_j] += 1

            #diag
            if i < M - 1 and j < N - 1:
                cur_w  = get_weight(str_a[i],str_b[j],_vocab, _b_matrix)
                cur_j = cur_i + M + 1
                graph[2+cur_i].append([cur_j,cur_w])
                graph[max_node+cur_j+2].append([cur_i])
                graph[0][1+ cur_i] += 1
                graph[1][1+ cur_j] += 1

            #from start fo cur_i
            #if cur_i > 0 and cur_i != (M + 1) and cur_i != 1 and cur_i != :
            if cur_i > 0:
                #graph[2].append([cur_i,0])
                graph[max_node+cur_i+2].append([0])
                #graph[0][1] += 1
                graph[1][1+ cur_i] += 1


            #from cur_i to end
            #if cur_i < (max_node-1) and cur_i != (max_node - 1 - (M+1)):
            if cur_i < (max_node-1):
                graph[2+cur_i].append([(max_node - 1),0])
                graph[max_node+(max_node - 1)+2].append([cur_i])
                graph[0][1 + cur_i] += 1
                graph[1][1+ (max_node - 1)] += 1

    return graph

def get_longest_path_letters(_graph, _path, str_a, str_b):
    result = []
    letters_a = len(str_a)
    letters_b = len(str_b)
    M = letters_a + 1
    N = letters_b + 1

    max_node = M * N
    prev_p  = -1
    for p in _path:
        if prev_p >=0:
            delta = p - prev_p
            if delta == 1:
                cur_i = p % M
                if prev_p > 0:
                    result.append([str_a[cur_i - 1],'-'])
            elif delta == M:
                cur_j = p/M
                if prev_p > 0:
                    result.append(['-',str_b[cur_j - 1]])
            elif delta == (M + 1):
                cur_j = p/M
                cur_i = p % M
                if prev_p == 0:
                    cur_weight = -sys.maxint - 1
                    max_w = -sys.maxint - 1
                    for _p in _graph[2]:
                        if _p[0] == p:
                            cur_weight = _p[1]
                            if cur_weight > max_w:
                                max_w = cur_weight
                    if max_w < 0:
                        max_w = 0
                    if max_w != 0:
                        result.append([str_a[cur_i - 1],str_b[cur_j - 1]])
                else:
                    result.append([str_a[cur_i - 1],str_b[cur_j - 1]])
        prev_p = p

    return result
import  time

def task65():
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\06\\input5.txt", "r") as myfile:
        data=myfile.readlines()

    string_a = data[0].replace('\n','')
    string_b = data[1].replace('\n','')

    letters_a = len(string_a)
    letters_b = len(string_b)

    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\extra\\blossum.txt", "r") as myfile:
        blossum_data=myfile.readlines()

    #print(blossum_data)
    vocab = blossum_data[0].strip().replace('\n','').replace('  ',' ').split(' ')
    print(vocab)

    b_matrix = []

    for b in blossum_data[1:]:
        row = b.strip().replace('\n','').replace('  ',' ').split(' ')
        b_row = [int(i) for i in row[1:]]
        b_matrix.append([]+b_row)
    print(b_matrix)
    t0 = time.clock()
    graph = build_letter_graph(string_a,string_b,vocab,b_matrix)
    graph_time =  time.clock() - t0
    print(graph_time)
    size = graph[0][0]

    graph_size = graph[0][0]
    path_graph = [-sys.maxint - 1]*graph_size
    #path_row = [-sys.maxint - 1]*graph_size
    #for i in xrange(graph_size):
    #    path_graph.append([] + path_row)
    #backtrack = []

    for i in xrange(letters_a + 1):
        print(i)
        for j in xrange(letters_b + 1):
            cur_i = i + j*(letters_a + 1)
            get_longest_path_val(0,cur_i,graph,path_graph)

    max_val = get_longest_path_val(0,graph_size-1,graph,path_graph)
    print(max_val)

    path = get_longest_path(0,graph_size-1,graph,path_graph)
    #print(path)
    res = get_longest_path_letters(graph, path,string_a,string_b)
    '''res_str = ''
    for p in path:
        res_str += str(p)
        res_str += '->'
    print(res_str)'''
    res_a = ''
    res_b = ''
    for r in res:
        res_a += r[0]
        res_b += r[1]
    print(res_a)
    print(res_b)


def get_longest_path_val_ex(_start, _end, _graph, _path_graph):
    if _start == _end:
        return 0
    
    if _path_graph[_end] > (-sys.maxint - 1):
        return _path_graph[_end]

    max_val = (-sys.maxint - 1)
    max_i = 0
    N = _graph[0][0]
    pred_p = -1
    for p in _graph[2+N+_end]:
        w_list = _graph[2+p[0]]
        cur_weight = -sys.maxint - 1
        for w in w_list:
            #print(p[0], len(w_list))
            if w[0] == _end:
                cur_weight = w[1]
                if cur_weight > (-sys.maxint - 1):
                    cur_val = get_longest_path_val_ex(_start,p[0],_graph, _path_graph) + cur_weight
                    if cur_val >= max_val:
                        max_val = cur_val
                        max_i = p[0]

    if max_val <= 0:
        max_val = 0
        max_i = 0
    _path_graph[_end] = max_val
    for i in xrange(len(_graph[2+N+_end])):
        v = _graph[2+N+_end][i]
        if v[0] == max_i:
            _graph[2+N+_end][i].append(max_val)

    #_path_graph[_end] = 0
    #print(_i,_j,_graph[_i][_j],_backtrack[_i][_j])
    return max_val

def calc_score(str_a, str_b, _vocab, _b_matrix, _mu):
    result = 0
    for i in xrange(len(str_a)):
        l_a = str_a[i]
        l_b = str_b[i]
        if l_a == '-' or l_b=='-':
            result += _mu
        else:
            i = _vocab.index(l_a)
            j = _vocab.index(l_b)
            result += _b_matrix[i][j]
    return result

def task66():
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\06\\input61.txt", "r") as myfile:
        data=myfile.readlines()

    string_a = data[0].replace('\n','')
    string_b = data[1].replace('\n','')
    print(string_a)
    print(string_b)
    letters_a = len(string_a)
    letters_b = len(string_b)

    mu = -5

    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\extra\\pam250.txt", "r") as myfile:
        blossum_data=myfile.readlines()

    #print(blossum_data)
    vocab = blossum_data[0].strip().replace('\n','').replace('  ',' ').split(' ')
    #print(vocab)

    b_matrix = []

    for b in blossum_data[1:]:
        row = b.strip().replace('\n','').replace('  ',' ').split(' ')
        b_row = [int(i) for i in row[1:]]
        b_matrix.append([]+b_row)

    '''b_matrix = []
    vocab_size = len(vocab)
    row = [-1]*vocab_size
    for i in xrange(vocab_size):
        b_matrix.append([]+row)
        b_matrix[i][i] = 1'''
    #print(b_matrix)
    t0 = time.clock()
    graph = build_letter_graph_ex(string_a,string_b,vocab,b_matrix, mu)
    graph_time =  time.clock() - t0
    print(graph_time)
    size = graph[0][0]

    graph_size = graph[0][0]
    path_graph = [-sys.maxint - 1]*graph_size
    #path_row = [-sys.maxint - 1]*graph_size
    #for i in xrange(graph_size):
    #    path_graph.append([] + path_row)
    #backtrack = []
    path_graph[0] = 0

    for i in xrange(letters_a + 1):

        for j in xrange(letters_b + 1):
            cur_i = i + j*(letters_a + 1)
            t0 = time.clock()
            get_longest_path_val_ex(0,cur_i,graph,path_graph)
            t1 = time.clock() - t0
            #print(i,j,t1)

    max_val = get_longest_path_val_ex(0,graph_size-1,graph,path_graph)
    print(max_val)

    path = get_longest_path(0,graph_size-1,graph,path_graph)
    #print(path)
    res = get_longest_path_letters(graph, path,string_a,string_b)
    '''res_str = ''
    for p in path:
        res_str += str(p)
        res_str += '->'
    print(res_str)'''
    res_a = ''
    res_b = ''
    for r in res:
        res_a += r[0]
        res_b += r[1]
    print(res_a)
    print(res_b)

    fg = open('G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\06\\g6.txt','w')
    #print(graph,file=fg)
    fg.close()

    pg = open('G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\06\\pg6.txt','w')
    print(path_graph,file=pg)
    pg.close()

    pf = open('G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\06\\pf6.txt','w')
    print(path,file=pf)
    pf.close()

    print(calc_score(res_a,res_b,vocab,b_matrix,mu))

def task71():
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\07\\input11.txt", "r") as myfile:
        data=myfile.readlines()

    string_a = data[0].replace('\n','')
    string_b = data[1].replace('\n','')

    letters_a = len(string_a)
    letters_b = len(string_b)

    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\extra\\blossum.txt", "r") as myfile:
        blossum_data=myfile.readlines()

    #print(blossum_data)
    vocab = blossum_data[0].strip().replace('\n','').replace('  ',' ').split(' ')
    print(vocab)
    vocab_size = len(vocab)

    b_matrix = []

    row = [1]*vocab_size
    for i in xrange(vocab_size):
        b_matrix.append([]+row)
        b_matrix[i][i] = 2
    print(b_matrix)

    t0 = time.clock()
    graph = build_letter_graph(string_a,string_b,vocab,b_matrix,0)
    graph_time =  time.clock() - t0
    print(graph_time)
    size = graph[0][0]

    graph_size = graph[0][0]
    path_graph = [-sys.maxint - 1]*graph_size
    #path_row = [-sys.maxint - 1]*graph_size
    #for i in xrange(graph_size):
    #    path_graph.append([] + path_row)
    #backtrack = []

    for i in xrange(letters_a + 1):
        for j in xrange(letters_b + 1):
            cur_i = i + j*(letters_a + 1)
            get_longest_path_val(0,cur_i,graph,path_graph)

    max_val = get_longest_path_val(0,graph_size-1,graph,path_graph)
    print(max_val)

    path = get_longest_path(0,graph_size-1,graph,path_graph)
    #print(path)
    res = get_longest_path_letters(graph, path,string_a,string_b)
    '''res_str = ''
    for p in path:
        res_str += str(p)
        res_str += '->'
    print(res_str)'''
    res_a = ''
    res_b = ''
    for r in res:
        res_a += r[0]
        res_b += r[1]
    print(res_a)
    print(res_b)

    result = 0
    indels = 0
    for i in xrange(len(res_a)):
        l_a = res_a[i]
        l_b = res_b[i]
        if l_a == '-':
            indels += 1
        elif l_b=='-':
            indels += 1
        else:
            if l_a != l_b:
                result +=1
    result += indels
    print(result)

def build_letter_graph_fit(str_a,str_b,_vocab,_b_matrix,_mu):
    letters_a = len(str_a)
    letters_b = len(str_b)

    M = letters_a + 1
    N = letters_b + 1

    max_node = M * N

    graph = []
    rev_graph = []

    graph_header = [0]*(max_node+1)
    graph.append([] + graph_header)
    graph.append([] + graph_header)
    graph_row = []
    for i in xrange(2*max_node):
        graph.append([] + graph_row)

    graph[0][0] = max_node
    for i in xrange(M):
        for j in xrange(N):
            cur_i = i + j*M

            #down
            if i < M-1:
                cur_w = _mu
                cur_j = cur_i + 1
                graph[2+cur_i].append([cur_j,cur_w])
                graph[max_node+cur_j+2].append([cur_i])
                graph[0][1+ cur_i] += 1
                graph[1][1+ cur_j] += 1

            #right
            if j < N-1:
                cur_w = _mu
                cur_j = cur_i + M
                graph[2+cur_i].append([cur_j,cur_w])
                graph[max_node+cur_j+2].append([cur_i])
                graph[0][1+ cur_i] += 1
                graph[1][1+ cur_j] += 1

            #diag
            if i < M - 1 and j < N - 1:
                cur_w  = get_weight(str_a[i],str_b[j],_vocab, _b_matrix)
                cur_j = cur_i + M + 1
                graph[2+cur_i].append([cur_j,cur_w])
                graph[max_node+cur_j+2].append([cur_i])
                graph[0][1+ cur_i] += 1
                graph[1][1+ cur_j] += 1

            #from start fo cur_i
            #if cur_i > 0 and cur_i != (M + 1) and cur_i != 1 and cur_i != :
            if j == 0:
                if cur_i > 0:
                    #graph[2].append([cur_i,0])
                    graph[max_node+cur_i+2].append([0])
                    #graph[0][1] += 1
                    graph[1][1+ cur_i] += 1


            #from cur_i to end
            #if cur_i < (max_node-1) and cur_i != (max_node - 1 - (M+1)):
            if j == N - 1 and cur_i != (max_node - 1 - (M+1)) and cur_i < (max_node-1):
                graph[2+cur_i].append([(max_node - 1),0])
                graph[max_node+(max_node - 1)+2].append([cur_i])
                graph[0][1 + cur_i] += 1
                graph[1][1+ (max_node - 1)] += 1

    return graph

def get_longest_path_val_fit(_start, _end, _graph, _path_graph,_M):
    if _start == _end:
        return 0
    if _end == 55:
        nn = 0

    if _path_graph[_end] > (-sys.maxint - 1):
        if _end == 10945:
            print('for end = 10945 vallues is: ', _path_graph[_end])
        return _path_graph[_end]

    max_val = (-sys.maxint - 1)
    max_i = 0
    N = _graph[0][0]
    pred_p = -1
    for p in _graph[2+N+_end]:
        if p[0] == 47:
            mm = 0
        w_list = _graph[2+p[0]]
        if _end == 10945:
            print(p[0], w_list)
        cur_weight = -sys.maxint - 1
        for w in w_list:
            #print(p[0], len(w_list))
            if w[0] == _end:
                cur_weight = w[1]
                if cur_weight > (-sys.maxint - 1):
                    cur_val = get_longest_path_val_fit(_start,p[0],_graph, _path_graph,_M) + cur_weight
                    if cur_val >= max_val:
                        max_val = cur_val
                        max_i = p[0]

    if _end == 10945:
        print(max_i,max_val)
        print(_graph[2+N+_end])
    if _end <= _M:
        if max_val <= 0:
            max_val = 0
            max_i = 0
    _path_graph[_end] = max_val
    for i in xrange(len(_graph[2+N+_end])):
        v = _graph[2+N+_end][i]
        if v[0] == max_i:
            _graph[2+N+_end][i].append(max_val)

    #_path_graph[_end] = 0
    #print(_i,_j,_graph[_i][_j],_backtrack[_i][_j])
    return max_val

def task72():
    with open ("/Users/boolker/Desktop/tasks/bio/data/tasks/07/input22.txt", "r") as myfile:
        data=myfile.readlines()

    string_a = data[0].replace('\n','')
    string_a = string_a.replace('\r','')

    string_b = data[1].replace('\n','')
    string_b = string_b.replace('\r','')

    print(string_a)
    print(string_b)
    letters_a = len(string_a)
    letters_b = len(string_b)

    with open ("/Users/boolker/Desktop/tasks/bio/data/extra/blossum.txt", "r") as myfile:
        blossum_data=myfile.readlines()

    #print(blossum_data)
    vocab = blossum_data[0].strip().replace('\n','').replace('  ',' ').split(' ')
    #print(vocab)

    b_matrix = []
    vocab_size = len(vocab)
    row = [-1]*vocab_size
    for i in xrange(vocab_size):
        b_matrix.append([]+row)
        b_matrix[i][i] = 1
    #print(b_matrix)

    mu = -1


    min_i = -1
    glob_max_val = -sys.maxint - 1
    min_path_len = letters_a+letters_b+1
    fit_path = []
    res = []


    t0 = time.clock()
    graph = build_letter_graph_fit(string_a,string_b,vocab,b_matrix,mu)
    graph_time =  time.clock() - t0
    print(graph_time)

    size = graph[0][0]

    graph_size = graph[0][0]
    path_graph = [-sys.maxint - 1]*graph_size
    #path_row = [-sys.maxint - 1]*graph_size
    #for i in xrange(graph_size):
    #    path_graph.append([] + path_row)
    #backtrack = []
    path_graph[0] = 0

    for i in xrange(letters_a + 1):
        for j in xrange(letters_b + 1):
            cur_i = i + j*(letters_a + 1)
            t0 = time.clock()
            get_longest_path_val_fit(0,cur_i,graph,path_graph,letters_a+1)
            t1 = time.clock() - t0
            #print(i,j,t1)

    max_val = get_longest_path_val_fit(0,graph_size-1,graph,path_graph,letters_a+1)
    print(max_val)

    path = get_longest_path(0,graph_size-1,graph,path_graph)
    #print(path)
    res = get_longest_path_letters(graph, path,string_a,string_b)
    '''res_str = ''
    for p in path:
        res_str += str(p)
        res_str += '->'
    print(res_str)'''
    res_a = ''
    res_b = ''
    for r in res:
        res_a += r[0]
        res_b += r[1]
    print(res_a)
    print(res_b)


def build_letter_graph_overlap(str_a,str_b,_vocab,_b_matrix,_mu):
    letters_a = len(str_a)
    letters_b = len(str_b)

    M = letters_a + 1
    N = letters_b + 1

    max_node = M * N

    graph = []
    rev_graph = []

    graph_header = [0]*(max_node+1)
    graph.append([] + graph_header)
    graph.append([] + graph_header)
    graph_row = []
    for i in xrange(2*max_node):
        graph.append([] + graph_row)

    graph[0][0] = max_node
    for i in xrange(M):
        for j in xrange(N):
            cur_i = i + j*M

            #down
            if i < M-1:
                cur_w = _mu
                cur_j = cur_i + 1
                graph[2+cur_i].append([cur_j,cur_w])
                graph[max_node+cur_j+2].append([cur_i])
                graph[0][1+ cur_i] += 1
                graph[1][1+ cur_j] += 1

            #right
            if j < N-1:
                cur_w = _mu
                cur_j = cur_i + M
                graph[2+cur_i].append([cur_j,cur_w])
                graph[max_node+cur_j+2].append([cur_i])
                graph[0][1+ cur_i] += 1
                graph[1][1+ cur_j] += 1

            #diag
            if i < M - 1 and j < N - 1:
                cur_w  = get_weight(str_a[i],str_b[j],_vocab, _b_matrix)
                cur_j = cur_i + M + 1
                graph[2+cur_i].append([cur_j,cur_w])
                graph[max_node+cur_j+2].append([cur_i])
                graph[0][1+ cur_i] += 1
                graph[1][1+ cur_j] += 1

            #from start fo cur_i
            #if cur_i > 0 and cur_i != (M + 1) and cur_i != 1 and cur_i != :
            if j == 0 or i == 0:
                if cur_i > 0:
                    #graph[2].append([cur_i,0])
                    graph[max_node+cur_i+2].append([0])
                    #graph[0][1] += 1
                    graph[1][1+ cur_i] += 1


            #from cur_i to end
            #if cur_i < (max_node-1) and cur_i != (max_node - 1 - (M+1)):
            if i == M - 1 and cur_i != (max_node - 1 - (M+1)) and cur_i < (max_node-1):
                graph[2+cur_i].append([(max_node - 1),0])
                graph[max_node+(max_node - 1)+2].append([cur_i])
                graph[0][1 + cur_i] += 1
                graph[1][1+ (max_node - 1)] += 1
            if j == N - 1 and cur_i != (max_node - 1 - (M+1)) and cur_i < (max_node-1):
                graph[2+cur_i].append([(max_node - 1),0])
                graph[max_node+(max_node - 1)+2].append([cur_i])
                graph[0][1 + cur_i] += 1
                graph[1][1+ (max_node - 1)] += 1

    return graph

def get_longest_path_val_overlap(_start, _end, _graph, _path_graph,_M):
    if _start == _end:
        return 0
    if _end == 55:
        nn = 0

    if _path_graph[_end] > (-sys.maxint - 1):
        if _end == 10945:
            print('for end = 10945 vallues is: ', _path_graph[_end])
        return _path_graph[_end]

    max_val = (-sys.maxint - 1)
    max_i = 0
    N = _graph[0][0]
    pred_p = -1
    for p in _graph[2+N+_end]:
        if p[0] == 47:
            mm = 0
        w_list = _graph[2+p[0]]
        if _end == 10945:
            print(p[0], w_list)
        cur_weight = -sys.maxint - 1
        for w in w_list:
            #print(p[0], len(w_list))
            if w[0] == _end:
                cur_weight = w[1]
                if cur_weight > (-sys.maxint - 1):
                    cur_val = get_longest_path_val_overlap(_start,p[0],_graph, _path_graph,_M) + cur_weight
                    if cur_val >= max_val:
                        max_val = cur_val
                        max_i = p[0]

    if _end == 10945:
        print(max_i,max_val)
        print(_graph[2+N+_end])
    if _end <= _M or _end%_M == 0:
        if max_val <= 0:
            max_val = 0
            max_i = 0
    _path_graph[_end] = max_val
    for i in xrange(len(_graph[2+N+_end])):
        v = _graph[2+N+_end][i]
        if v[0] == max_i:
            _graph[2+N+_end][i].append(max_val)

    #_path_graph[_end] = 0
    #print(_i,_j,_graph[_i][_j],_backtrack[_i][_j])
    return max_val

def get_longest_path_letters_overlap(_graph, _path, str_a, str_b):
    result = []
    letters_a = len(str_a)
    letters_b = len(str_b)
    M = letters_a + 1
    N = letters_b + 1

    max_node = M * N
    prev_p  = -1
    prefix_size = 0
    postfix_size = 0
    score = 0
    for p in _path:
        if prev_p >=0:
            delta = p - prev_p
            if delta == 1:
                cur_i = p % M
                if prev_p > 0:
                    result.append([str_a[cur_i - 1],'-'])
                    score += -2
            elif delta == M:
                cur_j = p/M
                if prev_p > 0:
                    result.append(['-',str_b[cur_j - 1]])
                    score += -2
                    #print(score)
            elif delta == (M + 1):
                cur_j = p/M
                cur_i = p % M
                if prev_p == 0:
                    cur_weight = -sys.maxint - 1
                    max_w = -sys.maxint - 1
                    for _p in _graph[2]:
                        if _p[0] == p:
                            cur_weight = _p[1]
                            if cur_weight > max_w:
                                max_w = cur_weight
                    if max_w < 0:
                        max_w = 0
                    if max_w != 0:
                        result.append([str_a[cur_i - 1],str_b[cur_j - 1]])
                        if str_a[cur_i - 1] == str_b[cur_j - 1]:
                            score += 1
                            #print(score)
                        else:
                            score += -2
                            #print(score)
                else:
                    result.append([str_a[cur_i - 1],str_b[cur_j - 1]])
                    if str_a[cur_i - 1] == str_b[cur_j - 1]:
                        score += 1
                        #print(score)
                    else:
                        score += -2
                        #print(score)
            else:
                cur_i = p % M
                cur_j = p/M
                print(cur_i,cur_j,p,M,N,max_node)
                if cur_j == 0:
                    prefix_size = delta
                elif cur_j == N - 1:
                    prev_i = prev_p % M
                    prev_j = prev_p/M
                    if prev_i == M-1:
                        postfix_size = delta/M
                    elif prev_j == cur_j:
                        prefix_size = -1*delta
                elif cur_i == 0:
                    postfix_size = -1*delta/M
                elif cur_i == M-1:
                    postfix_size = delta/M 

        prev_p = p

    print(score)

    print(prefix_size,postfix_size)
    if prefix_size >= 0:
        for i in xrange(prefix_size):
            result = [[str_a[prefix_size - 1 - i],'-']] + result
        postfix_shift = letters_b - postfix_size
        for i in xrange(postfix_size):
            result.append(['-',str_b[postfix_shift + i]])
    else:
        for i in xrange(-1*postfix_size):
            result = [['-',str_b[-1*postfix_size - 1 - i]]] + result
            #print(result)
        prefix_shift = letters_a + prefix_size
        for i in xrange(-1*prefix_size):
            result.append([str_a[prefix_shift + i],'-'])
    result = [[score]] + result
    return result

def calc_overlap_score(res, _mu, _b_matrix):
    score = 0
    for r in res:
        if r[0] == '-' or res[1]=='-':
            score += -1
        else:
            if r[0] == r[1]:
                score += 1
            else:
                score += -1
    return score

def task73():
    with open ("/Users/boolker/Desktop/tasks/bio/data/tasks/07/input3.txt", "r") as myfile:
        data=myfile.readlines()

    string_a = data[0].replace('\n','')
    string_a = string_a.replace('\r','')

    string_b = data[1].replace('\n','')
    string_b = string_b.replace('\r','')

    print(string_a)
    print(string_b)
    letters_a = len(string_a)
    letters_b = len(string_b)

    with open ("/Users/boolker/Desktop/tasks/bio/data/extra/blossum.txt", "r") as myfile:
        blossum_data=myfile.readlines()

    #print(blossum_data)
    vocab = blossum_data[0].strip().replace('\n','').replace('  ',' ').split(' ')
    #print(vocab)

    b_matrix = []
    vocab_size = len(vocab)
    row = [-2]*vocab_size
    for i in xrange(vocab_size):
        b_matrix.append([]+row)
        b_matrix[i][i] = 1
    #print(b_matrix)

    mu = -2


    min_i = -1
    glob_max_val = -sys.maxint - 1
    min_path_len = letters_a+letters_b+1
    fit_path = []
    res = []


    t0 = time.clock()
    graph = build_letter_graph_overlap(string_a,string_b,vocab,b_matrix,mu)
    graph_time =  time.clock() - t0
    print(graph_time)

    size = graph[0][0]

    graph_size = graph[0][0]
    path_graph = [-sys.maxint - 1]*graph_size
    #path_row = [-sys.maxint - 1]*graph_size
    #for i in xrange(graph_size):
    #    path_graph.append([] + path_row)
    #backtrack = []
    path_graph[0] = 0

    for i in xrange(letters_a + 1):
        for j in xrange(letters_b + 1):
            cur_i = i + j*(letters_a + 1)
            t0 = time.clock()
            get_longest_path_val_overlap(0,cur_i,graph,path_graph,letters_a+1)
            t1 = time.clock() - t0
            #print(i,j,t1)

    max_val = get_longest_path_val_overlap(0,graph_size-1,graph,path_graph,letters_a+1)
    print(max_val)

    path = get_longest_path(0,graph_size-1,graph,path_graph)
    print(path)
    res = get_longest_path_letters_overlap(graph, path,string_a,string_b)
    res1 = get_longest_path_letters(graph, path,string_a,string_b)
    
    '''res_str = ''
    for p in path:
        res_str += str(p)
        res_str += '->'
    print(res_str)'''
    res_a = ''
    res_b = ''
    for r in res:
        if len(r)>1:
            res_a += r[0]
            res_b += r[1]
    print(res[0][0])
    #print(res_a)
    #print(res_b)

    res_a = ''
    res_b = ''
    for r in res1:
        if len(r)>1:
            res_a += r[0]
            res_b += r[1]
    #print(res[0][0])
    print(res_a)
    print(res_b)

def build_3_level_graph(str_a,str_b,_vocab,_b_matrix,_sigma,_eps):
    letters_a = len(str_a)
    letters_b = len(str_b)

    M = letters_a + 1
    N = letters_b + 1

    max_node = M * N * 3
    level_size = M * N

    graph = []
    rev_graph = []

    graph_header = [0]*(max_node+1)
    graph.append([] + graph_header)
    graph.append([] + graph_header)
    graph_row = []
    for i in xrange(2*max_node):
        graph.append([] + graph_row)

    graph[0][0] = max_node
    for i in xrange(M):
        for j in xrange(N):
            cur_i = i + j*M

            #down
            if i < M-1:
                cur_w = _eps
                cur_j = cur_i + 1
                graph[2+cur_i].append([cur_j,cur_w])
                graph[max_node+cur_j+2].append([cur_i])
                graph[0][1+ cur_i] += 1
                graph[1][1+ cur_j] += 1

            #right
            if j < N-1:
                cur_w = _eps
                right_i = cur_i + level_size
                cur_j = right_i + M 
                graph[2+right_i].append([cur_j,cur_w])
                graph[max_node+cur_j+2].append([right_i])
                graph[0][1+ right_i] += 1
                graph[1][1+ cur_j] += 1

            #diag
            down_i = cur_i
            right_i = cur_i + level_size
            #from diag to down and right
            diag_i = cur_i + 2*level_size

            if i < M - 1 and j < N - 1:
                diag_i = cur_i + 2*level_size
                cur_w  = get_weight(str_a[i],str_b[j],_vocab, _b_matrix)
                cur_j = diag_i + M + 1
                graph[2+diag_i].append([cur_j,cur_w])
                graph[max_node+cur_j+2].append([diag_i])
                graph[0][1+ diag_i] += 1
                graph[1][1+ cur_j] += 1

            
            cur_w = _sigma

            if i < M - 1:
                cur_j = down_i + 1
                graph[2+diag_i].append([cur_j,cur_w])
                graph[max_node+cur_j+2].append([diag_i])
                graph[0][1+ diag_i] += 1
                graph[1][1+ cur_j] += 1

            if j < N - 1:
                cur_j = right_i + M
                graph[2+diag_i].append([cur_j,cur_w])
                graph[max_node+cur_j+2].append([diag_i])
                graph[0][1+ diag_i] += 1
                graph[1][1+ cur_j] += 1

            #from right to diag
            cur_w = 0
            cur_j = diag_i
            graph[2+right_i].append([cur_j,cur_w])
            graph[max_node+cur_j+2].append([right_i])
            graph[0][1+ right_i] += 1
            graph[1][1+ cur_j] += 1

            #from down to diag
            cur_w = 0
            cur_j = diag_i
            graph[2+down_i].append([cur_j,cur_w])
            graph[max_node+cur_j+2].append([down_i])
            graph[0][1+ down_i] += 1
            graph[1][1+ cur_j] += 1

    #print(graph)
    return graph

def get_longest_3_level_path_letters(_graph, _path, str_a, str_b):
    result = []
    letters_a = len(str_a)
    letters_b = len(str_b)
    M = letters_a + 1
    N = letters_b + 1

    level_size = M * N
    #print('level size = ',level_size)
    max_node = level_size * 3 
    prev_p  = -1
    for p in _path:
        if prev_p >=0:
            delta = abs(p - prev_p)
            #print(p, delta, M, M+1)
            level_p = p % level_size

            if (delta % level_size) == 0:
                #print('between levels')
                mm =1
            elif delta == 1:
                #print('down')
                cur_i = level_p % M
                if prev_p > 0:
                    result.append([str_a[cur_i - 1],'-'])
            elif delta == M:
                #print('right')
                cur_j = level_p/M
                if prev_p > 0:
                    result.append(['-',str_b[cur_j - 1]])
            elif delta == (M + 1):
                cur_j = level_p/M
                cur_i = level_p % M
                #print('diag ',cur_i,cur_j)
                if prev_p == 0:
                    cur_weight = -sys.maxint - 1
                    max_w = -sys.maxint - 1
                    for _p in _graph[2]:
                        if _p[0] == p:
                            cur_weight = _p[1]
                            if cur_weight > max_w:
                                max_w = cur_weight
                    if max_w < 0:
                        max_w = 0
                    if max_w != 0:
                        result.append([str_a[cur_i - 1],str_b[cur_j - 1]])
                else:
                    result.append([str_a[cur_i - 1],str_b[cur_j - 1]])
            elif delta == (level_size - M):
                #print('to right')
                cur_j = level_p/M
                if prev_p > 0:
                    result.append(['-',str_b[cur_j - 1]])
            elif delta == (2*level_size - 1):
                #print('to down')
                cur_i = level_p % M
                if prev_p > 0:
                    result.append([str_a[cur_i - 1],'-'])
        prev_p = p

    return result



def task74():
    with open ("/Users/boolker/Desktop/tasks/bio/data/tasks/07/input41.txt", "r") as myfile:
        data=myfile.readlines()

    string_a = data[0].replace('\n','')
    string_a = string_a.replace('\r','')

    string_b = data[1].replace('\n','')
    string_b = string_b.replace('\r','')

    sigma = -11
    eps = -1
    print(string_a)
    print(string_b)
    letters_a = len(string_a)
    letters_b = len(string_b)

    M = letters_a + 1
    N = letters_b + 1
    level_size = M * N

    with open ("/Users/boolker/Desktop/tasks/bio/data/extra/blossum.txt", "r") as myfile:
        blossum_data=myfile.readlines()

    #print(blossum_data)
    vocab = blossum_data[0].strip().replace('\n','').replace('  ',' ').split(' ')
    #print(vocab)

    b_matrix = []
    b_matrix = []

    for b in blossum_data[1:]:
        row = b.strip().replace('\n','').replace('  ',' ').split(' ')
        b_row = [int(i) for i in row[1:]]
        b_matrix.append([]+b_row)

    min_i = -1
    glob_max_val = -sys.maxint - 1
    min_path_len = letters_a+letters_b+1
    fit_path = []
    res = []


    t0 = time.clock()
    graph = build_3_level_graph(string_a,string_b,vocab,b_matrix,sigma,eps)
    graph_time =  time.clock() - t0
    print(graph_time)

    size = graph[0][0]

    graph_size = graph[0][0]
    path_graph = [-sys.maxint - 1]*graph_size
    
    #return

    path_graph[0] = 0

    for i in xrange(letters_a + 1):
        for j in xrange(letters_b + 1):
            
            t0 = time.clock()
            cur_i = i + j*(letters_a + 1)
            get_longest_path_val(0,cur_i,graph,path_graph)

            cur_i += level_size
            get_longest_path_val(0,cur_i,graph,path_graph)

            cur_i += level_size
            get_longest_path_val(0,cur_i,graph,path_graph)
            t1 = time.clock() - t0
            #print(i,j,t1)

    max_val = get_longest_path_val(0,graph_size-1,graph,path_graph)
    print(max_val)

    path = get_longest_path(0,graph_size-1,graph,path_graph)
    #print(path)
    
    res = get_longest_3_level_path_letters(graph, path,string_a,string_b)
    
    res_a = ''
    res_b = ''
    for r in res:
        if len(r)>1:
            res_a += r[0]
            res_b += r[1]
    
    print(res_a)
    print(res_b)

def task75():
    with open ("/Users/boolker/Desktop/tasks/bio/data/tasks/07/input6.txt", "r") as myfile:
        data=myfile.readlines()

    string_a = data[0].replace('\n','')
    string_a = string_a.replace('\r','')

    string_b = data[1].replace('\n','')
    string_b = string_b.replace('\r','')

    mu = -5
    print(string_a)
    print(string_b)
    letters_a = len(string_a)
    letters_b = len(string_b)

    M = letters_a + 1
    N = letters_b + 1

    middle_len = max(N/2,N-N/2)
    print(N/2,N-N/2,middle_len)

    string_a_left = string_a
    string_b_left = string_b[:middle_len-1]

    string_a_right = string_a[::-1]
    string_b_right = string_b[middle_len-1:][::-1]

    with open ("/Users/boolker/Desktop/tasks/bio/data/extra/blossum.txt", "r") as myfile:
        blossum_data=myfile.readlines()

    #print(blossum_data)
    vocab = blossum_data[0].strip().replace('\n','').replace('  ',' ').split(' ')
    #print(vocab)

    b_matrix = []
    b_matrix = []

    for b in blossum_data[1:]:
        row = b.strip().replace('\n','').replace('  ',' ').split(' ')
        b_row = [int(i) for i in row[1:]]
        b_matrix.append([]+b_row)

    graph_left = build_letter_graph(string_a_left,string_b_left,vocab,b_matrix,mu)
    graph_right = build_letter_graph(string_a_right,string_b_right,vocab,b_matrix,mu)
    
    letters_a_left = len(string_a_left)
    letters_b_left = len(string_b_left)

    graph_size_left = graph_left[0][0]
    path_graph_left = [-sys.maxint - 1]*graph_size_left

    path_graph_left[0] = 0

    for i in xrange(letters_a_left + 1):
        for j in xrange(letters_b_left + 1):
            #t0 = time.clock()
            cur_i = i + j*(letters_a_left + 1)
            get_longest_path_val(0,cur_i,graph_left,path_graph_left)

    letters_a_right = len(string_a_right)
    letters_b_right = len(string_b_right)

    graph_size_right = graph_right[0][0]
    path_graph_right = [-sys.maxint - 1]*graph_size_right

    path_graph_right[0] = 0


    for i in xrange(letters_a_right + 1):
        for j in xrange(letters_b_right + 1):
            #t0 = time.clock()
            cur_i = i + j*(letters_a_right + 1)
            get_longest_path_val(0,cur_i,graph_right,path_graph_right)

    print('calculated')
    print(string_a,string_b[:middle_len-1])
    print(string_a[::-1],string_b[middle_len-1:][::-1])

    #print(path_graph_left)
    #print(path_graph_right)
    max_val = -sys.maxint -1
    middle_i = -1
    for i in xrange(letters_a_left + 1):
        cur_val = path_graph_left[(-i-1)] + path_graph_right[(-1*(letters_a_left + 1) + i)]
        if cur_val > max_val:
            max_val = cur_val
            middle_i = i 

    path_right = get_longest_path(0,graph_size_right - ((letters_a_right + 1) - middle_i),graph_right,path_graph_right)
    delta = path_right[-1] - path_right[-2]
    print(delta)
    middle_start_a = letters_a_left - middle_i
    middle_start_b = letters_b_left

    middle_end_a = letters_a_left - middle_i
    middle_end_b = letters_b_left

    if delta == 1:
        middle_end_a += 1
    elif delta == letters_a_left + 1:
        middle_end_b += 1
    elif delta == letters_a_left + 2:
        middle_end_a += 1
        middle_end_b += 1
    #print(path_right)
    #print(max_val, letters_a_left - middle_i)
    res_str = '('
    res_str += str(middle_start_a)
    res_str += ','
    res_str += str(middle_start_b)
    res_str += ') ('
    res_str += str(middle_end_a)
    res_str += ','
    res_str += str(middle_end_b)
    res_str += ')'
    print(res_str)

def get_longest_path_val_lin(_i, _j, _path_graph, _M, _mu, _b_matrix_val):

    if _path_graph[_i][_j] > (-sys.maxint - 1):
        return _path_graph[_i][_j]

    max_val = (-sys.maxint - 1)
    max_i = 0
    
    if (_i) != 0:
        cur_val = _path_graph[_i-1][_j] + _mu
        if cur_val > max_val:
            max_val = cur_val

    if _j > 0:
        cur_val = _path_graph[_i][_j-1] + _mu
        if cur_val > max_val:
            max_val = cur_val

        if _i != 0:
            cur_val = _path_graph[_i-1][_j-1] + _b_matrix_val
            if cur_val > max_val:
                max_val = cur_val

    _path_graph[_i][_j] = max_val
    return max_val

def get_weight_ex(str_a, _i, str_b, _j, _vocab, _b_matrix):
    result = 0
    if _i < 0 or _j <0:
        return 0
    ind_a = _vocab.index(str_a[_i])
    ind_b = _vocab.index(str_b[_j])
    result = _b_matrix[ind_a][ind_b]
    return result

def linear_space_allign_down(str_a, str_b, _mu, _vocab, _b_matrix, _path_graph_right,_start_i,_start_j):
    result = []
    #print('down',str_a,str_b,_start_i,_start_j)

    letters_a = len(str_a)
    letters_b = len(str_b)

    if letters_a == 0 and letters_b == 0:
        return []

    M = letters_a + 1
    N = letters_b + 1

    middle_len = max(N/2,N-N/2)
    #print(N/2,N-N/2,middle_len)

    string_a_left = str_a
    string_b_left = str_b[:middle_len-1]

    string_a_right = str_a[::-1]
    string_b_right = str_b[middle_len-1:][::-1]

    letters_a_left = len(string_a_left)
    letters_b_left = len(string_b_left)

    M_l = letters_a_left+1
    N_l = letters_b_left+1

    letters_a_right = len(string_a_right)
    letters_b_right = len(string_b_right)

    M_r = letters_a_right+1
    N_r = letters_b_right+1

    #print(M_l,N_l,M_r,N_r)
    
    path_graph_row = [-sys.maxint - 1]*N_l
    path_graph_left = []
    for i in xrange(M_l):    
        path_graph_left.append([]+path_graph_row)

    path_graph_left[0][0] = 0

    #print(string_a_left,string_b_left)

    for i in xrange(letters_a_left + 1):
        for j in xrange(letters_b_left + 1):
            get_longest_path_val_lin(i,j,path_graph_left, M_l, _mu, get_weight_ex(string_a_left,i-1,string_b_left,j-1,_vocab, _b_matrix))

    #print(path_graph_left)
    #print(_path_graph_right)
    max_val = -sys.maxint -1
    middle_i = -1
    for i in xrange(letters_a_left + 1):
        cur_val = path_graph_left[M_l - i - 1][N_l-1] + _path_graph_right[i][N_r-1]
        if cur_val >= max_val:
            max_val = cur_val
            middle_i = i 

    #print(max_val,middle_i)

    start_i = letters_a_left-middle_i
    start_j = letters_b_left

    result.append([start_i+_start_i,start_j+_start_j])
    #print(path_graph_right[middle_i][N_r-1],path_graph_right[middle_i-1][N_r-1],path_graph_right[middle_i][N_r-2],path_graph_right[middle_i-1][N_r-2])
    #print('before calling up: ',_start_i,_start_j)
    result_left = linear_space_allign_up(str_a[:start_i],str_b[:start_j],_mu,_vocab,_b_matrix,path_graph_left,_start_i,_start_j)
    
    if middle_i >0:
        if _path_graph_right[middle_i][N_r-1] - _path_graph_right[middle_i-1][N_r-1] == _mu:
             start_i += 1
        elif _path_graph_right[middle_i][N_r-1] - _path_graph_right[middle_i][N_r-2] == _mu:
            start_j += 1
        else:
            start_i += 1
            start_j += 1
    else:
        start_i += 1
        start_j += 1
    result.append([start_i+_start_i,start_j+_start_j])
             
    #print(result)


    result_right = linear_space_allign_down(str_a[start_i:],str_b[start_j:],_mu,_vocab,_b_matrix,_path_graph_right,start_i+_start_i,start_j+_start_j)
    #print('lower matrix: ',str_a,str_b,result_left,'result: ',result,result_right,'starts: ',_start_i,_start_j)
    result = result_left + result + result_right
    
    return result

def linear_space_allign_up(str_a, str_b, _mu, _vocab, _b_matrix, _path_graph_left,_start_i,_start_j):
    result = []
    #print('up',str_a,str_b,_start_i,_start_j)

    letters_a = len(str_a)
    letters_b = len(str_b)

    if letters_a == 0 and letters_b == 0:
        return []

    M = letters_a + 1
    N = letters_b + 1

    middle_len = max(N/2,N-N/2)
    #print(N/2,N-N/2,middle_len)

    string_a_left = str_a
    string_b_left = str_b[:middle_len-1]

    string_a_right = str_a[::-1]
    string_b_right = str_b[middle_len-1:][::-1]

    letters_a_left = len(string_a_left)
    letters_b_left = len(string_b_left)

    M_l = letters_a_left+1
    N_l = letters_b_left+1
    
    letters_a_right = len(string_a_right)
    letters_b_right = len(string_b_right)

    M_r = letters_a_right+1
    N_r = letters_b_right+1
    path_graph_row = [-sys.maxint - 1]*N_r
    path_graph_right = []
    for i in xrange(M_r):    
        path_graph_right.append([]+path_graph_row)

    path_graph_right[0][0] = 0

    for i in xrange(letters_a_right + 1):
        for j in xrange(letters_b_right + 1):
            #t0 = time.clock()
            cur_i = i + j*(letters_a_right + 1)
            #print(i,j)
            get_longest_path_val_lin(i,j,path_graph_right, letters_a_right+1, _mu, get_weight_ex(string_a_right,i-1,string_b_right,j-1,_vocab, _b_matrix))

    max_val = -sys.maxint -1
    middle_i = -1
    for i in xrange(letters_a_left + 1):
        cur_val = _path_graph_left[M_l - i - 1][N_l-1] + path_graph_right[i][N_r-1]
        if cur_val >= max_val:
            max_val = cur_val
            middle_i = i 

    #print(max_val,middle_i)

    start_i = letters_a_left-middle_i
    start_j = letters_b_left

    result.append([start_i+_start_i,start_j+_start_j])
    #print(path_graph_right[middle_i][N_r-1],path_graph_right[middle_i-1][N_r-1],path_graph_right[middle_i][N_r-2],path_graph_right[middle_i-1][N_r-2])
    #print('before calling up: ',_start_i,_start_j)
    result_left = linear_space_allign_up(str_a[:start_i],str_b[:start_j],_mu,_vocab,_b_matrix,_path_graph_left,_start_i,_start_j)
    
    if middle_i >0:
        if path_graph_right[middle_i][N_r-1] - path_graph_right[middle_i-1][N_r-1] == _mu:
             start_i += 1
        elif path_graph_right[middle_i][N_r-1] - path_graph_right[middle_i][N_r-2] == _mu:
            start_j += 1
        else:
            start_i += 1
            start_j += 1
    else:
        if path_graph_right[middle_i][N_r-1] - path_graph_right[middle_i][N_r-2] == _mu:
            start_j += 1
        else:
            start_i += 1
            start_j += 1
    result.append([start_i+_start_i,start_j+_start_j])
             
    #print(result)
    result_right = linear_space_allign_down(str_a[start_i:],str_b[start_j:],_mu,_vocab,_b_matrix,path_graph_right,start_i+_start_i,start_j+_start_j)
    
    #print('upper matrix: ',str_a,str_b,result_left,'result: ',result,result_right,'starts: ',_start_i,_start_j)
    result = result_left + result + result_right
    return result

def linear_space_allignment(str_a, str_b, _mu, _vocab, _b_matrix):
    print(str_a)
    print(str_b)
    letters_a = len(str_a)
    letters_b = len(str_b)

    M = letters_a + 1
    N = letters_b + 1
    print(M,N)

    middle_len = max(N/2,N-N/2)
    print(N/2,N-N/2,middle_len)

    string_a_left = str_a
    string_b_left = str_b[:middle_len-1]

    string_a_right = str_a[::-1]
    string_b_right = str_b[middle_len-1:][::-1]

    #graph_left = build_letter_graph(string_a_left,string_b_left,vocab,b_matrix,mu)
    #graph_right = build_letter_graph(string_a_right,string_b_right,vocab,b_matrix,mu)
    
    letters_a_left = len(string_a_left)
    letters_b_left = len(string_b_left)

    M_l = letters_a_left+1
    N_l = letters_b_left+1
    path_graph_row = [-sys.maxint - 1]*N_l
    path_graph_left = []
    for i in xrange(M_l):    
        path_graph_left.append([]+path_graph_row)

    path_graph_left[0][0] = 0

    #print(string_a_left,string_b_left)

    for i in xrange(letters_a_left + 1):
        for j in xrange(letters_b_left + 1):
            get_longest_path_val_lin(i,j,path_graph_left, M_l, _mu, get_weight_ex(string_a_left,i-1,string_b_left,j-1,_vocab, _b_matrix))

    letters_a_right = len(string_a_right)
    letters_b_right = len(string_b_right)

    M_r = letters_a_right+1
    N_r = letters_b_right+1
    path_graph_row = [-sys.maxint - 1]*N_r
    path_graph_right = []
    for i in xrange(M_r):    
        path_graph_right.append([]+path_graph_row)

    path_graph_right[0][0] = 0

    for i in xrange(letters_a_right + 1):
        for j in xrange(letters_b_right + 1):
            #t0 = time.clock()
            cur_i = i + j*(letters_a_right + 1)
            #print(i,j)
            get_longest_path_val_lin(i,j,path_graph_right, letters_a_right+1, _mu, get_weight_ex(string_a_right,i-1,string_b_right,j-1,_vocab, _b_matrix))


    print('calculated')

    max_val = -sys.maxint -1
    middle_i = -1
    for i in xrange(letters_a_left + 1):
        cur_val = path_graph_left[M_l - i - 1][N_l-1] + path_graph_right[i][N_r-1]
        if cur_val >= max_val:
            max_val = cur_val
            middle_i = i 

    print(max_val)
    
    result = []
    result.append([letters_a_left-middle_i,letters_b_left])
    #print(path_graph_right[middle_i][N_r-1],path_graph_right[middle_i-1][N_r-1],path_graph_right[middle_i][N_r-2],path_graph_right[middle_i-1][N_r-2])
    start_i = letters_a_left-middle_i
    start_j = letters_b_left

    result_left = linear_space_allign_up(str_a[:start_i],str_b[:start_j],_mu,_vocab,_b_matrix,path_graph_left,0,0)
    
    if middle_i >0:
        if path_graph_right[middle_i][N_r-1] - path_graph_right[middle_i-1][N_r-1] == _mu:
             start_i += 1
        elif path_graph_right[middle_i][N_r-1] - path_graph_right[middle_i][N_r-2] == _mu:
            start_j += 1
        else:
            start_i += 1
            start_j += 1
    else:
        start_i += 1
        start_j += 1
    result.append([start_i,start_j])
             
    #print(result)


    result_right = linear_space_allign_down(str_a[start_i:],str_b[start_j:],_mu,_vocab,_b_matrix,path_graph_right,start_i,start_j)
    result = result_left + result + result_right
    return result

def task76():
    with open ("/Users/boolker/Desktop/tasks/bio/data/tasks/07/input62.txt", "r") as myfile:
        data=myfile.readlines()

    string_a = data[0].replace('\n','')
    string_a = string_a.replace('\r','')

    string_b = data[1].replace('\n','')
    string_b = string_b.replace('\r','')

    mu = -5
    
    with open ("/Users/boolker/Desktop/tasks/bio/data/extra/blossum.txt", "r") as myfile:
        blossum_data=myfile.readlines()

    #print(blossum_data)
    vocab = blossum_data[0].strip().replace('\n','').replace('  ',' ').split(' ')
    #print(vocab)

    b_matrix = []
    b_matrix = []

    for b in blossum_data[1:]:
        row = b.strip().replace('\n','').replace('  ',' ').split(' ')
        b_row = [int(i) for i in row[1:]]
        b_matrix.append([]+b_row)

    res = linear_space_allignment(string_a,string_b,mu,vocab,b_matrix)
    print(res)
    #print(path_graph_left)

    #print(path_graph_right)
    res_str = []
    prev_r = []
    for r in res:
        if len(prev_r)>0:
            if r[0] != prev_r[0] or r[1] != prev_r[1]:
                delta_a = r[0] - prev_r[0]
                delta_b = r[1] - prev_r[1]
                if delta_a > 0:
                    if delta_b > 0:
                        print(prev_r[0])
                        print(string_a[prev_r[0]])
                        res_str.append([string_a[prev_r[0]],string_b[prev_r[1]]])
                    else:
                        res_str.append([string_a[prev_r[0]],'-'])
                else:
                    res_str.append(['-',string_b[prev_r[1]]])
        prev_r = []+r
    #print(res_str)
    res_a = ''
    res_b = ''
    for r in res_str:
        if len(r)>1:
            res_a += r[0]
            res_b += r[1]
    
    print(res_a)
    print(res_b)

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

def add_edge(_tree,_start,_end,_len):
    print("add edge from ", _start," to ",_end,"with len = ",_len)
    
    N = _tree[0][0]
    _tree[_start+2].append([_end,_len])
    _tree[N+_end+2].append([_start])
    _tree[0][1+ _start] += 1
    _tree[1][1+ _end] += 1

def print_phylogeny_tree(_tree):
    N = _tree[0][0]
    for i in xrange(N):
        ends = _tree[2+i]
        for end in ends:
            print(i,"->",end[0],":",int(end[1]))

def task43():
    # Implement AdditivePhylogeny to solve the Distance-Based Phylogeny Problem.
    # Input: An integer n followed by a space-separated n x n distance matrix.
    # Output: A weighted adjacency list for the simple tree fitting this matrix.
    #input_file_name = os.getcwd() + "/bio02/data/04/input031.txt"
    input_file_name = "/Users/boolker/Desktop/tasks/bio02/data/04/input034.txt"
    
    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()
    
    dim = int(data[0])    
    distances=[]
    for d in data[1:]:
        row_str = d.replace('\n','').replace('\t',' ')
        #print(row_str)
        row_data = [float(i) for i in row_str.split(' ')]
        distances.append(row_data)

    result_tree = []
    build_empty_graph(result_tree, dim)
    additive_phylogeny(result_tree,distances,dim)
    print_phylogeny_tree(result_tree)


if __name__ == "__main__":   
    task43() 
    
