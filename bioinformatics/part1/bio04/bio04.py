from __future__ import print_function
__author__ = 'drummer'
import math
from random import randrange


def task40():
    max_size = 50
    results = [0]*max_size
    for i in xrange(1000000):
        x = randrange(0,max_size-1)
        results[x] += 1
    print(results)

def split_dna(dna_str,read_len):
    results = []
    for i in xrange(len(dna_str)):
        substr = dna_str[i:i+read_len]
        if len(substr) == read_len:
            results.append(substr)
    results = sorted(results)
    return results

def split_dna_ex(dna_str,read_len):
    results = []
    for i in xrange(len(dna_str)):
        substr = dna_str[i:i+read_len]
        if len(substr) == read_len:
            results.append(substr)
    results = sorted(set(results))
    return results

def task41():
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\04\\input1.txt", "r") as myfile:
        data=myfile.readlines()

    str_size = int(data[0].replace('\n',''))
    dna_str = data[1].replace('\n','')

    results = []
    for i in xrange(len(dna_str)):
        substr = dna_str[i:i+str_size]
        if len(substr) == str_size:
            results.append(substr)
    results = sorted(results)
    #print results
    for r in results:
        print(r)

def get_dna_graph(dna_reads):
    dna_num = len(dna_reads)

    dna_reads_2 = [] + dna_reads

    dna_graph_row = [0]*dna_num
    dna_graph = []
    for i in xrange(dna_num):
        dna_graph.append([] + dna_graph_row)
    #dna_graph = [dna_graph_row]*dna_num
    results = []
    for i in xrange(len(dna_reads)):
        print(i)
        for j in xrange(len(dna_reads_2)):
        #for d2 in dna_reads_2:
            d = dna_reads[i]
            d2 = dna_reads_2[j]
            s_d = d[1:]
            s_d_2 = d2[:-1]
            if d[1:] == d2[:-1]:
                dna_graph[i][j] += 1
    return dna_graph

def get_dna_graph_ex(dna_str, dna_reads, read_len):
    dna_num = len(dna_reads)

    dna_reads_2 = [] + dna_reads

    dna_graph_row = [0]*dna_num
    dna_graph = []
    for i in xrange(dna_num):
        dna_graph.append([] + dna_graph_row)
    #dna_graph = [dna_graph_row]*dna_num

    for i in xrange(len(dna_str)):
        cur_substr = dna_str[i:i+read_len]
        next_substr = dna_str[i+1:i+1+read_len]
        if len(cur_substr) == read_len and len(next_substr) == read_len:
            cur_i = dna_reads.index(cur_substr)
            cur_j = dna_reads.index(next_substr)
            dna_graph[cur_i][cur_j] += 1

    '''for i in xrange(len(dna_reads)):
        print(i)
        for j in xrange(len(dna_reads_2)):
        #for d2 in dna_reads_2:
            d = dna_reads[i]
            d2 = dna_reads_2[j]
            s_d = d[1:]
            s_d_2 = d2[:-1]
            if d[1:] == d2[:-1]:
                dna_graph[i][j] += 1'''
    return dna_graph

def task42():
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\04\\input2.txt", "r") as myfile:
        data=myfile.readlines()

    dna_reads = []
    for d in data:
        dna_reads.append(d.replace('\n',''))
    dna_reads = sorted(dna_reads)
    dna_num = len(dna_reads)
    dna_reads_2 = [] + dna_reads

    dna_graph_row = [0]*dna_num
    dna_graph = []
    for i in xrange(dna_num):
        dna_graph.append([] + dna_graph_row)
    #dna_graph = [dna_graph_row]*dna_num
    results = []
    for i in xrange(len(dna_reads)):
        print(i)
        for j in xrange(len(dna_reads_2)):
        #for d2 in dna_reads_2:
            d = dna_reads[i]
            d2 = dna_reads_2[j]
            s_d = d[1:]
            s_d_2 = d2[:-1]
            if d[1:] == d2[:-1]:
                if dna_graph[i][j] == 0:
                    dna_graph[i][j] = 1
                    break

    #for r in results:
    #    print(r)
    f = open('G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\04\\output2.txt','w')
    for i in xrange(len(dna_reads)):
        for j in xrange(len(dna_reads_2)):
            if dna_graph[i][j] == 1:
                r = dna_reads[i] + ' -> ' + dna_reads_2[j]
                print(r, file=f)

    f.close()
    print('done')

def task43():
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\04\\input3.txt", "r") as myfile:
        data=myfile.readlines()

    str_size = int(data[0].replace('\n',''))
    dna_str = data[1].replace('\n','')
    dna_reads = split_dna_ex(dna_str,str_size-1)
    print(dna_reads)
    dna_num = len(dna_reads)
    dna_graph = get_dna_graph_ex(dna_str, dna_reads,str_size-1)
    #for r in results:
    #    print(r)
    f = open('G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\04\\output3.txt','w')
    read_len = str_size-1
    for i in xrange(len(dna_reads)):
        sum_row = 0
        for j in xrange(len(dna_reads)):
            if dna_graph[i][j] == 1:
                sum_row += 1
        k = 0
        r = ''
        for j in xrange(len(dna_reads)):
            if dna_graph[i][j] == 1:
                if k == 0:
                    r += dna_reads[i] + ' -> ' + dna_reads[j]
                else:
                    r += dna_reads[j]
                k += 1
                if k < sum_row:
                    r += ','
        if len(r) > 0:
            print(r, file=f)

    f.close()
    print('done')

def task44():
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\04\\input4.txt", "r") as myfile:
        data=myfile.readlines()

    dna_reads_full = []
    for d in data:
        dna_reads_full.append(d.replace('\n',''))

    dna_reads = []
    dna_lefts = []
    dna_rights = []
    for d in dna_reads_full:
        d2 = d[1:]
        d1 = d[:-1]
        dna_reads.append(d1)
        dna_reads.append(d2)
        dna_lefts.append(d1)
        dna_rights.append(d2)

    dna_reads = sorted(set(dna_reads))
    str_size = len(dna_reads[0])
    dna_reads_len = len(dna_reads)

    dna_graph_row = [0]*dna_reads_len
    dna_graph = []
    for i in xrange(dna_reads_len):
        dna_graph.append([] + dna_graph_row)

    #dna_graph = get_dna_graph(dna_reads)
    #1. add pairs
    for i in xrange(len(dna_lefts)):
        cur_i = dna_reads.index(dna_lefts[i])
        cur_j = dna_reads.index(dna_rights[i])
        dna_graph[cur_i][cur_j] += 1


    f = open('G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\04\\output4.txt','w')
    read_len = str_size-1
    for i in xrange(len(dna_reads)):
        sum_row = 0
        for j in xrange(len(dna_reads)):
            if dna_graph[i][j] == 1:
                sum_row += 1
        k = 0
        r = ''
        for j in xrange(len(dna_reads)):
            if dna_graph[i][j] == 1:
                if k == 0:
                    r += dna_reads[i] + ' -> ' + dna_reads[j]
                else:
                    r += dna_reads[j]
                k += 1
                if k < sum_row:
                    r += ','
        if len(r) > 0:
            print(r, file=f)
            #print(r)

    f.close()
    print('done')

from numpy import *

def build_graph_by_pairs(pairs):
    #graph = []
    lefts = []
    rights = []
    max_node = 0
    for p in pairs:
        start = int(p[0])
        end_list = p[2].split(',')
        ends = [int(i) for i in end_list]
        for e in ends:
            lefts.append(start)
            rights.append(e)
            if start > max_node:
                max_node = start
            if e > max_node:
                max_node = e

    l_len = len(lefts)
    graph = [max_node+1] + [0]*((max_node+2)*(max_node+1))
    '''graph = []
    graph_row = [0]*((max_node+1)*(max_node+1))
    for i in xrange(max_node+1):
        graph.append([] + graph_row)'''
    #graph = zeros((max_node+1,max_node+1),int)
    N = max_node+1
    for l in xrange(l_len):
        cur_i = lefts[l]
        cur_j = rights[l]
        graph[N*cur_i+cur_j+1 + N] += 1
        graph[1+ cur_i] += 1

    return graph

def purify_pairs(pairs):
    pairs = []
    for p in pairs:
        start = int(p[0])
        end_list = p[2].split(',')
        ends = [int(i) for i in end_list]
        for e in ends:
            pairs.append([start,e])
    return pairs

def build_graph_by_pure_pairs(pairs):
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
    graph = [max_node+1] + [0]*((max_node+2)*(max_node+1))
    '''graph = []
    graph_row = [0]*((max_node+1)*(max_node+1))
    for i in xrange(max_node+1):
        graph.append([] + graph_row)'''
    #graph = zeros((max_node+1,max_node+1),int)
    N = max_node+1
    for l in xrange(l_len):
        cur_i = lefts[l]
        cur_j = rights[l]
        graph[N*cur_i+cur_j+1 + N] += 1
        graph[1+ cur_i] += 1

    return graph

def build_graph_ex_by_pure_pairs(pairs):
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
    for i in xrange(N):
        graph.append([] + graph_row)
    #graph = zeros((max_node+1,max_node+1),int)
    graph[0][0] = N
    for l in xrange(l_len):
        cur_i = lefts[l]
        cur_j = rights[l]
        graph[cur_i+2].append(cur_j)
        graph[0][1+ cur_i] += 1
        graph[1][1+ cur_j] += 1

    return graph

def is_balanced_graph(graph):
    balanced = 1
    graph_size = graph[0]
    for i in xrange(graph_size):
        sum_row = 0
        sum_col = 0
        for j in xrange(graph_size):
            sum_row += graph[graph_size*i+j+1+graph_size]
            sum_col += graph[graph_size*j+i+1+graph_size]
        if sum_row != sum_col:
            balanced = 0
            break
    return balanced

def is_balanced_graph_ex(graph):
    balanced = 1
    graph_size = graph[0][0]
    for i in xrange(graph_size):
        sum_row = graph[0][i]
        sum_col = graph[1][i]
        if sum_row != sum_col:
            balanced = 0
            break
    return balanced

def is_balanced_graph_2d(graph):
    balanced = 1
    graph_size = graph[0][0]
    for i in xrange(graph_size):
        sum_row = 0
        sum_col = 0
        for j in xrange(graph_size):
            sum_row += graph[i+1][j]
            sum_col += graph[j+1][i]
        if sum_row != sum_col:
            balanced = 0
            break
    return balanced

def balance_graph(graph):
    pairs = []
    starts = []
    ends = []
    graph_size = graph[0]
    for i in xrange(graph_size):
        sum_row = 0
        sum_col = 0
        for j in xrange(graph_size):
            sum_row += graph[graph_size*i+j+1+graph_size]
            sum_col += graph[graph_size*j+i+1+graph_size]
        if sum_row != sum_col:
            balanced = 0
            if sum_row > sum_col:
                starts.append([i,sum_row-sum_col])
            else:
                ends.append([i,sum_col-sum_row])
    for e in ends:
        for i in xrange(e[1]):
            for j in xrange(len(starts)):
                start = starts[j]
                pairs.append([e[0],start[0]])
                starts[j][1] -= 1
                if starts[j][1] <=0:
                    del starts[j]
    return pairs

def balance_graph_ex(graph):
    pairs = []
    starts = []
    ends = []
    graph_size = graph[0][0]
    for i in xrange(graph_size):
        sum_row = graph[0][i+1]
        sum_col = graph[1][i+1]
        if sum_row != sum_col:
            balanced = 0
            if sum_row > sum_col:
                starts.append([i,sum_row-sum_col])
            else:
                ends.append([i,sum_col-sum_row])
    for e in ends:
        for i in xrange(e[1]):
            for j in xrange(len(starts)):
                start = starts[j]
                pairs.append([e[0],start[0]])
                starts[j][1] -= 1
                if starts[j][1] <=0:
                    del starts[j]
    return pairs

def balance_graph_2d(graph):
    pairs = []
    starts = []
    ends = []
    graph_size = graph[0][0]
    for i in xrange(graph_size):
        sum_row = 0
        sum_col = 0
        for j in xrange(graph_size):
            sum_row += graph[i+1][j]
            sum_col += graph[j+1][i]
        if sum_row != sum_col:
            balanced = 0
            if sum_row > sum_col:
                starts.append([i,sum_row-sum_col])
            else:
                ends.append([i,sum_col-sum_row])
    for e in ends:
        for i in xrange(e[1]):
            for j in xrange(len(starts)):
                start = starts[j]
                pairs.append([e[0],start[0]])
                starts[j][1] -= 1
                if starts[j][1] <=0:
                    del starts[j]
    return pairs

def get_sum_archs(graph):
    graph_size = graph[0]
    sum_archs = 0
    for i in xrange(graph_size):
        for j in xrange(graph_size):
            sum_archs += graph[graph_size*i+j+1+graph_size]
    return sum_archs

def get_sum_archs_ex(graph):
    graph_size = graph[0][0]
    sum_archs = 0
    for i in xrange(graph_size):
        sum_archs += graph[0][i+1]
    return sum_archs

import collections

import  time

def task45():
    #with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\04\\input51.txt", "r") as myfile:
    with open ("/Volumes/Pirogov N/1/input5.txt", "r") as myfile:
        data=myfile.readlines()

    pairs = []
    for d in data:
        pairs.append(d.replace('\n','').split(' '))
    #print(pairs)
    graph = build_graph_2d_by_pairs(pairs)
    #print(is_balanced_graph(graph))

    graph_size = graph[0]
    sum_archs = get_sum_archs(graph)
    #print(sum_archs)

    cycle = []
    left_archs = sum_archs
    start = 0
    cycle = collections.deque([0])
    build_time = 0
    build_end_time = 0
    shift_time = 0

    while left_archs > 0:
        #base_cycle = []

        #cur_graph = []
        #for g in graph:
        #    cur_graph.append(g[:])
        #find path
        t0 = time.clock()
        while 1:
            found = 0
            for i in xrange(graph_size):
                if graph[graph_size*start+i+1+graph_size] >0:
                    graph[graph_size*start+i+1+graph_size] -= 1
                    graph[start+1] -= 1
                    left_archs -=1
                    cycle.append(i)
                    start = i
                    found = 1
                    break
            if found == 0:
                #the end
                t1 = time.clock()
                #left_archs = get_sum_archs(graph)
                #find start pos
                for c in cycle:
                    if graph[c+1] > 0: # or graph[i][c] > 0:
                        start = c
                        break
                build_end_time += (time.clock() - t1)
                break
        build_time += (time.clock() - t0)
        print(left_archs,build_time,build_end_time,shift_time)
        #now we have new start
        t0 = time.clock()
        cycle.pop()
        last_i = -1
        for i in xrange(len(cycle)):
            if cycle[i] == start:
                last_i = i
        if last_i >= 0:
            cycle.rotate(-1*last_i)
        else:
            print('error')
            return
        cycle.append(cycle[0])
        shift_time += (time.clock() - t0)

    f = open('/Volumes/Pirogov N/1/out5.txt','w')
    res_str = ''
    for c in cycle:
        res_str+= str(c)
        res_str+= '->'
    print(res_str,file=f)
    f.close()

def find_euler_path(pairs,pure=0):
    graph = []
    if pure == 0:
        graph = build_graph_by_pairs(pairs)
    else:
        graph = build_graph_by_pure_pairs(pairs)

    print('built graph')
    graph_size = graph[0]
    balance_pairs = []
    if not is_balanced_graph(graph):
        print('graph is not balanced')
        balance_pairs = balance_graph(graph)
        for p in balance_pairs:
            graph[graph_size*p[0]+p[1]+1+graph_size] += 1
            graph[p[0]+1] += 1

    sum_archs = get_sum_archs(graph)
    print('found sum archs')
    #print(sum_archs)

    cycle = []
    left_archs = sum_archs
    start = 0
    cycle = collections.deque([0])
    build_time = 0
    build_end_time = 0
    shift_time = 0

    while left_archs > 0:
        #base_cycle = []

        #cur_graph = []
        #for g in graph:
        #    cur_graph.append(g[:])
        #find path
        t0 = time.clock()
        while 1:
            found = 0
            for i in xrange(graph_size):
                if graph[graph_size*start+i+1+graph_size] >0:
                    graph[graph_size*start+i+1+graph_size] -= 1
                    graph[start+1] -= 1
                    left_archs -=1
                    cycle.append(i)
                    start = i
                    found = 1
                    break
            if found == 0:
                #the end
                t1 = time.clock()
                #left_archs = get_sum_archs(graph)
                #find start pos
                for c in cycle:
                    if graph[c+1] > 0: # or graph[i][c] > 0:
                        start = c
                        break
                build_end_time += (time.clock() - t1)
                break
        build_time += (time.clock() - t0)
        print(left_archs,build_time,build_end_time,shift_time)
        #now we have new start
        t0 = time.clock()
        cycle.pop()
        last_i = -1
        for i in xrange(len(cycle)):
            if cycle[i] == start:
                last_i = i
        if last_i >= 0:
            cycle.rotate(-1*last_i)
        else:
            print('error')
            return
        cycle.append(cycle[0])
        shift_time += (time.clock() - t0)

    cycle.pop()
    brk = -1
    for i in xrange(len(cycle)-1):
        for p in balance_pairs:
            if cycle[i] == p[0] and cycle[i+1] == p[1]:
                brk = i+1
                break
        if brk != -1:
            break

    if brk == -1:
        for p in balance_pairs:
            if cycle[0] == p[1] and cycle[-1] == p[0]:
                brk = 0

    if brk >0:
        cycle.rotate(-1*brk)
    return cycle

def find_euler_path_ex(pairs,pure=0):
    graph = []
    if pure == 0:
        graph = build_graph_ex_by_pairs(pairs)
    else:
        graph = build_graph_ex_by_pure_pairs(pairs)

    print('built graph')
    graph_size = graph[0][0]
    balance_pairs = []
    if not is_balanced_graph_ex(graph):
        print('graph is not balanced')
        balance_pairs = balance_graph_ex(graph)
        for p in balance_pairs:
            cur_i = p[0]
            cur_j = p[1]
            graph[cur_i+2].append(cur_j)
            graph[0][1+ cur_i] += 1
            graph[1][1+ cur_j] += 1

    sum_archs = get_sum_archs_ex(graph)
    print('found sum archs')
    #print(sum_archs)

    cycle = []
    left_archs = sum_archs
    start = 0
    start_i = 0
    cycle = collections.deque([0])
    build_time = 0
    build_end_time = 0
    shift_time = 0

    while left_archs > 0:
        #base_cycle = []

        #cur_graph = []
        #for g in graph:
        #    cur_graph.append(g[:])
        #find path
        t0 = time.clock()
        done = 0
        while done == 0:
            ends = graph[start+2]
            if len(ends) > 0:
                i = ends[0]
                cycle.append(i)
                del ends[0]
                graph[start+2] = ends
                graph[0][start+1] -= 1
                graph[1][i+1] -= 1
                start = i
                left_archs -=1
            else:
                t1 = time.clock()
                #left_archs = get_sum_archs(graph)
                #find start pos
                k = 0
                for c in cycle:
                    if graph[0][c+1] > 0: # or graph[i][c] > 0:
                        start = c
                        start_i = k
                        done = 1
                        break
                    k+=1
                build_end_time += (time.clock() - t1)
                break

        build_time += (time.clock() - t0)
        print(left_archs,build_time,build_end_time,shift_time)
        #now we have new start
        #t0 = time.clock()

        cycle.pop()
        t0 = time.clock()
        last_i = start_i
        '''t0 = time.clock()
        for i in xrange(len(cycle)):
            if cycle[i] == start:
                last_i = i
        shift_time += (time.clock() - t0)'''
        if last_i >= 0:
            cycle.rotate(-1*last_i)
        else:
            print('error')
            return
        cycle.append(cycle[0])
        shift_time += (time.clock() - t0)

    cycle.pop()
    brk = -1
    for i in xrange(len(cycle)-1):
        for p in balance_pairs:
            if cycle[i] == p[0] and cycle[i+1] == p[1]:
                brk = i+1
                break
        if brk != -1:
            break

    if brk == -1:
        for p in balance_pairs:
            if cycle[0] == p[1] and cycle[-1] == p[0]:
                brk = 0

    if brk >0:
        cycle.rotate(-1*brk)
    return cycle

def find_euler_path_ex_by_graph(graph):

    print('built graph')
    graph_size = graph[0][0]
    balance_pairs = []
    if not is_balanced_graph_ex(graph):
        print('graph is not balanced')
        balance_pairs = balance_graph_ex(graph)
        for p in balance_pairs:
            graph[p[0]+1][p[1]] += 1
            graph[0][p[0]+1] += 1

    sum_archs = get_sum_archs_ex(graph)
    print('found sum archs')
    #print(sum_archs)

    cycle = []
    left_archs = sum_archs
    start = 0
    start_i = 0
    cycle = collections.deque([0])
    build_time = 0
    build_end_time = 0
    shift_time = 0

    while left_archs > 0:
        #base_cycle = []

        #cur_graph = []
        #for g in graph:
        #    cur_graph.append(g[:])
        #find path
        t0 = time.clock()
        done = 0
        while done == 0:
            ends = graph[start+2]
            if len(ends) > 0:
                i = ends[0]
                cycle.append(i)
                del ends[0]
                graph[start+2] = ends
                graph[0][start+1] -= 1
                graph[1][i+1] -= 1
                start = i
                left_archs -=1
            else:
                t1 = time.clock()
                #left_archs = get_sum_archs(graph)
                #find start pos
                k = 0
                for c in cycle:
                    if graph[0][c+1] > 0: # or graph[i][c] > 0:
                        start = c
                        start_i = k
                        done = 1
                        break
                    k+=1
                build_end_time += (time.clock() - t1)
                break

        build_time += (time.clock() - t0)
        print(left_archs,build_time,build_end_time,shift_time)
        #now we have new start
        #t0 = time.clock()

        cycle.pop()
        t0 = time.clock()
        last_i = start_i
        '''t0 = time.clock()
        for i in xrange(len(cycle)):
            if cycle[i] == start:
                last_i = i
        shift_time += (time.clock() - t0)'''
        if last_i >= 0:
            cycle.rotate(-1*last_i)
        else:
            print('error')
            return
        cycle.append(cycle[0])
        shift_time += (time.clock() - t0)

    cycle.pop()
    brk = -1
    for i in xrange(len(cycle)-1):
        for p in balance_pairs:
            if cycle[i] == p[0] and cycle[i+1] == p[1]:
                brk = i+1
                break
        if brk != -1:
            break

    if brk == -1:
        for p in balance_pairs:
            if cycle[0] == p[1] and cycle[-1] == p[0]:
                brk = 0

    if brk >0:
        cycle.rotate(-1*brk)
    return cycle

def task51():
    #with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\04\\input51.txt", "r") as myfile:
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\05\\input1.txt", "r") as myfile:
        data=myfile.readlines()

    pairs = []
    for d in data:
        pairs.append(d.replace('\n','').split(' '))
    #print(pairs)
    path = find_euler_path(pairs)

    f = open('G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\05\\output1.txt','w')
    res_str = ''
    for c in path:
        res_str+= str(c)
        res_str+= '->'
    print(res_str,file=f)
    #print(res_str)
    f.close()

def task52():
    #with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\04\\input51.txt", "r") as myfile:
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\05\\input2.txt", "r") as myfile:
        data=myfile.readlines()

    pairs = []
    for d in data:
        pairs.append(d.replace('\n','').split(' '))
    #print(pairs)
    substr_list = []
    for p in pairs:
        substr_list.append(p[0])
        substr_list.append(p[2])

    substr_list = sorted(substr_list)
    substr_list = list(set(substr_list))
    pure_pairs = []
    for p in pairs:
        cur_i = substr_list.index(p[0])
        cur_j = substr_list.index(p[2])
        pure_pairs.append([cur_i,cur_j])
    path = find_euler_path(pure_pairs,1)

    f = open('G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\05\\output2.txt','w')
    res_str = ''
    for c in path:
        if len(res_str) == 0:
            res_str+= substr_list[c]
        else:
            substr = substr_list[c]
            res_str+= substr[-1]
    #print(res_str,file=f)
    print(res_str)
    f.close()

def generate_strings(k):
    results = []
    letters = ['0','1']
    prev_results = ['0','1']
    for i in xrange(k-1):
        del results[:]
        for p in prev_results:
            for l in letters:
                results.append(p + l)
        prev_results = [] + results
    return results


def task53():
    #with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\04\\input51.txt", "r") as myfile:
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\05\\input3.txt", "r") as myfile:
        data=myfile.readlines()

    node_size = 18 #int(data[0].replace('\n',''))
    #generate
    node_list = generate_strings(node_size)
    print('generated strings')

    dna_reads = []
    dna_lefts = []
    dna_rights = []
    for d in node_list:
        d2 = d[1:]
        d1 = d[:-1]
        dna_reads.append([d1,d2])
        dna_lefts.append(d1)
        dna_rights.append(d2)

    #print(pairs)
    '''substr_list = []
    for d in dna_reads:
        substr_list.append(d[0])
        substr_list.append(d[1])

    substr_list = sorted(substr_list)
    substr_list = list(set(substr_list))'''
    substr_list = generate_strings(node_size-1)
    print('prapered substr list')
    pure_pairs = []
    for d in dna_reads:
        #cur_i = substr_list.index(d[0])
        #cur_j = substr_list.index(d[1])
        cur_i = int(d[0],2)
        cur_j = int(d[1],2)
        pure_pairs.append([cur_i,cur_j])
    print('prapered pure pairs', len(pure_pairs))
    path = find_euler_path_ex(pure_pairs,1)
    print('found path')
    f = open('G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\05\\output3.txt','w')
    res_str = ''
    for c in path:
        if len(res_str) == 0:
            res_str+= substr_list[c]
        else:
            substr = substr_list[c]
            res_str+= substr[-1]
    #print(res_str,file=f)
    del_len = len(substr_list[0]) - 1
    res_str = res_str[:-1*del_len]
    print(res_str)
    print(res_str,file=f)
    f.close()

def task540():
    in_str = 'TAATGCCATGGGATGTT'
    size = 3
    delta = 2
    res_str = ''
    pairs = []
    for i in xrange(len(in_str)):
        left = in_str[i:i+size]
        right = in_str[i+size+delta:i+size+delta+size]
        if len(left)==size and len(right)==size:
            pairs.append('('+left+'|'+right+'),')
    pairs = sorted(pairs)
    for p in pairs:
        res_str += p
    print(res_str)

def task54():
    #with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\04\\input51.txt", "r") as myfile:
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\05\\input6.txt", "r") as myfile:
        data=myfile.readlines()

    num = int(data[0].replace('\n',''))
    pairs_full = []
    for d in data[1:]:
        pairs_full.append(d.replace('\n','').split('|'))

    #print(pairs_full)

    dna_reads_full_up = []
    dna_reads_full_down = []

    dna_reads_up = []
    dna_lefts_up = []
    dna_rights_up = []

    dna_reads_down = []
    dna_lefts_down = []
    dna_rights_down = []

    pairs = []
    pairs_left = []
    pairs_right = []
    pairs_sum = []
    k_len = 0
    for p in pairs_full:
        up = p[0]
        down = p[1]
        if (len(up) - 1) > k_len:
            k_len = len(up) - 1
        pairs.append([up[1:],down[1:],up[1:]+down[1:]])
        pairs.append([up[:-1],down[:-1],up[:-1]+down[:-1]])
        pairs_left.append([up[:-1],down[:-1],up[:-1]+down[:-1]])
        pairs_right.append([up[1:],down[1:],up[1:]+down[1:]])
        pairs_sum.append(up[1:]+down[1:])
        pairs_sum.append(up[:-1]+down[:-1])


    pairs_sum = sorted(set(pairs_sum))
    #str_size = len(dna_reads[0])
    pairs_len = len(pairs_sum)

    #dna_graph = get_dna_graph(dna_reads)
    #1. add pairs
    N = pairs_len
    pair_graph = []
    graph_header = [0]*(N+1)
    pair_graph.append([] + graph_header)
    pair_graph.append([] + graph_header)
    graph_row = []
    for i in xrange(N):
        pair_graph.append([] + graph_row)
    #graph = zeros((max_node+1,max_node+1),int)
    pair_graph[0][0] = N

    for i in xrange(len(pairs_left)):
        cur_i = pairs_sum.index(pairs_left[i][2])
        cur_j = pairs_sum.index(pairs_right[i][2])
        #pair_graph[cur_i][cur_j] += 1
        pair_graph[cur_i+2].append(cur_j)
        pair_graph[0][1+ cur_i] += 1
        pair_graph[1][1+ cur_j] += 1

    pure_pairs = []
    for i in xrange(len(pairs_left)):
        cur_i = pairs_sum.index(pairs_left[i][2])
        cur_j = pairs_sum.index(pairs_right[i][2])
        pure_pairs.append([cur_i,cur_j])
    print('prapered pure pairs', len(pure_pairs))
    path = find_euler_path_ex(pure_pairs,1)

    f = open('G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\05\\output6.txt','w')
    res_str = ''
    for p in path:
        cur_pair = pairs_sum[p]
        pair_up = cur_pair[:k_len]
        pair_down = cur_pair[k_len:]
        if len(res_str) == 0:
            res_str+= pair_up
        else:
            res_str+= pair_up[-1]
    path_len = len(path)
    d = num
    start_down = path_len - k_len - 2 - d
    for i in xrange(path_len):
        if i > start_down:
            cur_pair = pairs_sum[path[i]]
            pair_down = cur_pair[k_len:]
            res_str+= pair_down[-1]
    print(res_str,file=f)
    f.close()

def build_contig(_contig, graph, nodes):
    contig = [] + _contig
    start = _contig[-1]
    ends = graph[start+2]
    starts_num = graph[1][start+1]
    if len(ends) == 1 and starts_num == 1:
        nodes.append(start)
        cur_contigs = []
        for e in ends:
            cur_contigs.append(_contig + [e])
        for c in cur_contigs:
            contig = build_contig(c,graph,nodes)
    return contig

def build_contigs(graph):
    contigs = []
    left_archs = get_sum_archs_ex(graph)

    start = 0
    start_i = 0
    #cycle = collections.deque([0])
    #build_time = 0
    #build_end_time = 0
    #shift_time = 0
    nodes = []
    graph_size = graph[0][0]
    for i in xrange(graph_size):
        if i == 15:
            m = 0
        if i not in nodes:
            ends = graph[i+2]
            starts_num = graph[1][i+1]
            if starts_num != 1 or len(ends) != 1:
                nodes.append(i)
                cur_contigs = []
                for e in ends:
                    cur_contigs.append([i, e])
                for c in cur_contigs:
                    contigs.append(build_contig(c,graph,nodes))

    return contigs

def task55():
    #with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\04\\input51.txt", "r") as myfile:
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\05\\input5.txt", "r") as myfile:
        data=myfile.readlines()

    dna_reads_full = []
    for d in data:
        dna_reads_full.append(d.replace('\n',''))

    dna_reads = []
    dna_lefts = []
    dna_rights = []
    substr_list = []
    for d in dna_reads_full:
        d2 = d[1:]
        d1 = d[:-1]
        dna_reads.append([d1,d2])
        substr_list.append(d1)
        substr_list.append(d2)

    substr_list = sorted(set(substr_list))
    str_size = len(dna_reads[0])
    dna_reads_len = len(substr_list)

    pure_pairs = []
    for d in dna_reads:
        cur_i = substr_list.index(d[0])
        cur_j = substr_list.index(d[1])
        pure_pairs.append([cur_i,cur_j])
    p_len = len(pure_pairs)

    print('prapered pure pairs', len(pure_pairs))
    graph = build_graph_ex_by_pure_pairs(pure_pairs)

    '''_i = substr_list.index('AACACACTGACTGTTATGCGCGATCGGAAATTGGTGGCGGTCCGGCTTGGCGTAGCACTGATCGGAT')
    _j = substr_list.index('AAACACACTGACTGTTATGCGCGATCGGAAATTGGTGGCGGTCCGGCTTGGCGTAGCACTGATCGGA')
    g_i = graph[2+_i]
    g_j = graph[2+_j]'''

    contigs = build_contigs(graph)
    print('found path')

    f = open('G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\05\\output5.txt','w')
    res_strings = []
    for contig in contigs:
        res_str = ''
        for c in contig:
            if len(res_str) == 0:
                res_str+= substr_list[c]
            else:
                substr = substr_list[c]
                res_str+= substr[-1]
        res_strings.append(res_str)
    res_strings = sorted(res_strings)
    for r in res_strings:
        print(r,file=f)

    f.close()
if __name__ == "__main__":
   task54()
