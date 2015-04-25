from __future__ import print_function
import sys
import collections

__author__ = 'drummer'


def task101():
    #with open ("/Users/boolker/Desktop/tasks/bio/data/tasks/07/input62.txt", "r") as myfile:
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\10\\input11.txt", "r") as myfile:
        data=myfile.readlines()

    genom_str = data[0].replace('\n','').replace('\r','')
    print(genom_str)

    lst = []
    for g in genom_str:
        lst.append(g)

    res_lst = []
    cycle = collections.deque(lst)
    g_len = len(genom_str)
    for i in xrange(g_len):
        cycle.rotate(-1)
        res_str = ''.join(cycle)
        res_lst.append(res_str)


    res_lst = sorted(res_lst)
    #print(res_lst)
    res_str = ''
    for r in res_lst:
        res_str += r[-1]
    print(res_str)

def task102():
    #with open ("/Users/boolker/Desktop/tasks/bio/data/tasks/07/input62.txt", "r") as myfile:
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\10\\input21.txt", "r") as myfile:
        data=myfile.readlines()

    genom_str = data[0].replace('\n','').replace('\r','')
    print(genom_str)

    last_col = []
    for g in genom_str:
        last_col.append(g)
    first_col = sorted(last_col)

    dict = {}
    prev_c = ''
    for i in xrange(len(first_col)):
        c = first_col[i]
        if c != prev_c:
            dict[c] = i
        prev_c = c
    #print(dict)

    res_str = ''
    end = 0
    cur_i = 0
    cur_sym = last_col[0]
    cur_sym_ind = dict[cur_sym]
    while not end:
        cur_sym = first_col[cur_i]
        first_sym_ind = dict[cur_sym]
        num_sym = cur_i - first_sym_ind
        #find
        count_sym = 0
        last_sym_ind = -1
        for i in xrange(len(last_col)):
            if last_col[i] == cur_sym:
                if count_sym == num_sym:
                    last_sym_ind = i
                    break
                count_sym +=1
        res_str += first_col[cur_i]
        cur_i = last_sym_ind
        #print(res_str)
        if cur_i == 0:
            break

    res_str = res_str[1:] + '$'
    print(res_str)

def last_to_first(_dict, _first_col, _last_col,i):
    return 0

def count_pattern(_dict, _first_col, _last_col,pattern):
    result = 0

    top = 0
    bottom = len(_first_col)
    rev_pattern = pattern[::-1]

    for p in rev_pattern:
        count_sym = 0
        start_count = 0
        for i in xrange(top):
            if _last_col[i] == p:
                start_count += 1
        for i in xrange(top,bottom):
            if _last_col[i] == p:
                count_sym+=1
        top = start_count + _dict[p]
        bottom = top + count_sym
    result = bottom - top
    return result

def count_pattern_ex(_dict, _first_col, _last_col,_count_dict, pattern):
    result = 0

    top = 0
    bottom = len(_first_col)-1
    rev_pattern = pattern[::-1]

    for p in rev_pattern:
        top =  _dict[p] + _count_dict[p][top]
        bottom = _dict[p] + _count_dict[p][bottom+1] - 1
    result = bottom - top + 1
    return result

def task103():
    #with open ("/Users/boolker/Desktop/tasks/bio/data/tasks/07/input62.txt", "r") as myfile:
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\10\\input3.txt", "r") as myfile:
        data=myfile.readlines()

    genom_str = data[0].replace('\n','').replace('\r','')
    print(genom_str)

    last_col = []
    for g in genom_str:
        last_col.append(g)
    first_col = sorted(last_col)

    dict = {}
    prev_c = ''
    for i in xrange(len(first_col)):
        c = first_col[i]
        if c != prev_c:
            dict[c] = i
        prev_c = c

    '''count_dict = {}
    l_to_f = [0]*len(first_col)
    for i in xrange(len(first_col)):
        p = last_col[i]
        cur_val = count_dict.get(p,0)
        l_to_f[i] = dict[p] + cur_val
        count_dict[p] = cur_val + 1
    print(l_to_f)'''

    count_dict = {}
    l_to_f = [0]*len(first_col)
    for d in dict:
        print(d)

    patterns = data[1].replace('\n','').replace('\r','').split(' ')
    #print(patterns)
    #print(first_col)
    #print(last_col)

    res_str = ''
    for p in patterns:
        res_str += str(count_pattern(dict,first_col,last_col,p)) + ' '
    print(res_str)

def task104():
    #with open ("/Users/boolker/Desktop/tasks/bio/data/tasks/07/input62.txt", "r") as myfile:
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\10\\input41.txt", "r") as myfile:
        data=myfile.readlines()

    genom_str = data[0].replace('\n','').replace('\r','')
    #print(genom_str)

    last_col = []
    for g in genom_str:
        last_col.append(g)
    first_col = sorted(last_col)

    dict = {}
    prev_c = ''
    for i in xrange(len(first_col)):
        c = first_col[i]
        if c != prev_c:
            dict[c] = i
        prev_c = c

    count_dict = {}
    #l_to_f = [0]*len(first_col)
    col_len = len(last_col)
    for i in xrange(len(last_col)):
        p = last_col[i]
        cur_val = count_dict.get(p,[0]*(col_len+1))
        cur_val[i+1] = cur_val[i] + 1
        count_dict[p] = [] + cur_val

        for k in count_dict.keys():
            if k != p:
                count_dict[k][i+1] = count_dict[k][i]
    #print(l_to_f)'''

    print('calculated counts')
    patterns = data[1].replace('\n','').replace('\r','').split(' ')

    res_str = ''
    for p in patterns:
        res_str += str(count_pattern_ex(dict,first_col,last_col,count_dict,p)) + ' '
    print(res_str)

def task105():
    #with open ("/Users/boolker/Desktop/tasks/bio/data/tasks/07/input62.txt", "r") as myfile:
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\10\\input51.txt", "r") as myfile:
        data=myfile.readlines()

    genom_str = data[0].replace('\n','').replace('\r','')
    k = int(data[1].replace('\n','').replace('\r',''))

    print(genom_str)
    print(k)

    lst = []
    for g in genom_str:
        lst.append(g)

    res_lst = []
    cycle = collections.deque(lst)
    g_len = len(genom_str)
    for i in xrange(g_len):
        cycle.rotate(-1)
        res_str = ''.join(cycle)
        res_lst.append(res_str)


    res_lst = sorted(res_lst)
    #print(res_lst)
    genom_len = len(genom_str)
    for i in xrange(len(res_lst)):
        r = res_lst[i]
        ind = genom_len - r.index('$') - 1
        if ind % k == 0:
            res_str = str(i)+','+str(ind)
            print(res_str)

def get_pattern_ids(_dict, _first_col, _last_col,_count_dict,_suf_array, pattern):
    result = []

    top = 0
    bottom = len(_first_col)-1
    rev_pattern = pattern[::-1]

    for p in rev_pattern:
        top =  _dict[p] + _count_dict[p][top]
        bottom = _dict[p] + _count_dict[p][bottom+1] - 1
    #result = bottom - top + 1
    for i in xrange(top,bottom+1):
        result.append(_suf_array[i])
    return result

def task106():
    #with open ("/Users/boolker/Desktop/tasks/bio/data/tasks/07/input62.txt", "r") as myfile:
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\10\\input61.txt", "r") as myfile:
        data=myfile.readlines()

    genom_str = data[0].replace('\n','').replace('\r','') + '$'
    patterns = []
    for d in data[1:]:
        patterns.append(d.replace('\n','').replace('\r',''))
    #print(patterns)

    #k = 100
    k = 1

    lst = []
    for g in genom_str:
        lst.append(g)

    res_lst = []
    cycle = collections.deque(lst)
    g_len = len(genom_str)
    for i in xrange(g_len):
        cycle.rotate(-1)
        res_str = ''.join(cycle)
        res_lst.append(res_str)

    last_col = []
    first_col = []
    res_lst = sorted(res_lst)
    for r in res_lst:
        #print(r)
        first_col.append(r[0])
        last_col.append(r[-1])
    #print(res_lst)
    suf_array = {}
    genom_len = len(genom_str)
    for i in xrange(len(res_lst)):
        r = res_lst[i]
        ind = genom_len - r.index('$') - 1
        if ind % k == 0:
            res_str = str(i)+','+str(ind)
            suf_array[i]=ind
    #print(suf_array)

    '''last_col = []
    for g in genom_str:
        last_col.append(g)
    first_col = sorted(last_col)'''

    dict = {}
    prev_c = ''
    for i in xrange(len(first_col)):
        c = first_col[i]
        if c != prev_c:
            dict[c] = i
        prev_c = c

    count_dict = {}
    #l_to_f = [0]*len(first_col)
    col_len = len(last_col)
    for i in xrange(len(last_col)):
        p = last_col[i]
        cur_val = count_dict.get(p,[0]*(col_len+1))
        cur_val[i+1] = cur_val[i] + 1
        count_dict[p] = [] + cur_val

        for k in count_dict.keys():
            if k != p:
                count_dict[k][i+1] = count_dict[k][i]

    res = []
    for p in patterns:
        res += [] + get_pattern_ids(dict,first_col,last_col,count_dict,suf_array,p)

    res = sorted(res)
    res_str = ''
    for r in res:
        res_str += str(r) + ' '
    print(res_str)

def mismatches(substr, pattern,max_mismatches):
    mismatch_num = 0
    ii = 0
    for ii in xrange(len(substr)):
        if substr[ii] != pattern[ii]:
            mismatch_num = mismatch_num +1
        if mismatch_num >= max_mismatches:
            return max_mismatches

    return mismatch_num

def get_pattern_ids_ex(_genom, _dict, _first_col, _last_col,_count_dict,_suf_array, _mismatches, pattern):
    result = []

    top = 0
    bottom = len(_first_col)-1
    rev_pattern = pattern[::-1]

    pattern_len = len(pattern)
    piece_size = len(pattern)/(_mismatches+1)
    _i = 0
    count = 0
    while _i < pattern_len:
        top = 0
        bottom = len(_first_col)-1
        sub_pat = pattern[_i:_i + piece_size]
        #print(sub_pat)

        rev_sub_pat = sub_pat[::-1]
        for p in rev_sub_pat:
            top =  _dict[p] + _count_dict[p][top]
            bottom = _dict[p] + _count_dict[p][bottom+1] - 1
            #result = bottom - top + 1
        for i in xrange(top,bottom+1):
            suf_start = _suf_array[i]-_i
            mis_count = mismatches(_genom[suf_start:suf_start+pattern_len],pattern,_mismatches+1)
            if mis_count <= _mismatches:
                res_i = _suf_array[i]-_i
                if res_i >=0 and res_i not in result:
                    result.append(res_i)

        _i+= piece_size
        count += 1

    '''for p in rev_pattern:
        top =  _dict[p] + _count_dict[p][top]
        bottom = _dict[p] + _count_dict[p][bottom+1] - 1
    #result = bottom - top + 1
    for i in xrange(top,bottom+1):
        result.append(_suf_array[i])'''
    return result

def build_suf_array(_first_col,_last_col,_k):
    suf_array = []
    return suf_array

import  time
def task107():
    with open ("/Users/boolker/Desktop/tasks/bio/data/tasks/10/input71.txt", "r") as myfile:
    #with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\10\\input71.txt", "r") as myfile:
        data=myfile.readlines()

    genom_str = data[0].replace('\n','').replace('\r','')
    patterns = data[1].replace('\n','').replace('\r','').split(' ')
    mismatches = int(data[2].replace('\n','').replace('\r',''))
    print(len(patterns),'patterns')
    genom_str = genom_str[:50000] + '$'
    #print(patterns)
    k = 1
    lst = []
    for g in genom_str:
        lst.append(g)
    res_lst = []
    g_len = len(genom_str)
    for i in xrange(1,g_len+1):
        res_str = genom_str[-1*i:]+genom_str[:-1*i]
        res_lst.append(res_str)

    last_col = []
    first_col = []
    res_lst = sorted(res_lst)
    for r in res_lst:
        #print(r)
        first_col.append(r[0])
        last_col.append(r[-1])
    #print(res_lst)
    print('built bwt')


    genom_len = len(genom_str)
    suf_array = [0]*(genom_len+1)
    for i in xrange(len(res_lst)):
        r = res_lst[i]
        ind = genom_len - r.index('$') - 1
        if ind % k == 0:
            res_str = str(i)+','+str(ind)
            suf_array[i]=ind
    #print(suf_array)
    print('built suf array')

    dict = {}
    prev_c = ''
    for i in xrange(len(first_col)):
        c = first_col[i]
        if c != prev_c:
            dict[c] = i
        prev_c = c

    count_dict = {}
    #l_to_f = [0]*len(first_col)
    col_len = len(last_col)
    for i in xrange(len(last_col)):
        p = last_col[i]
        cur_val = count_dict.get(p,[0]*(col_len+1))
        cur_val[i+1] = cur_val[i] + 1
        count_dict[p] = [] + cur_val

        for k in count_dict.keys():
            if k != p:
                count_dict[k][i+1] = count_dict[k][i]

    print('built count dict')

    res = []
    t0 = time.clock()
    for p in patterns:
        res += [] + get_pattern_ids_ex(genom_str,dict,first_col,last_col,count_dict,suf_array,mismatches,p)
    build_time = time.clock() - t0
    print('built res',build_time)

    res = sorted(res)
    res_str = ''
    for r in res:
        res_str += str(r) + ' '
    print(res_str)

if __name__ == "__main__":
   task107()