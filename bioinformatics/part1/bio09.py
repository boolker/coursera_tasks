from __future__ import print_function
import sys
__author__ = 'drummer'

def print_nodes(_nodes,root_id):
    for n in _nodes:
        print(root_id, _nodes[n][0], n)
        print_nodes(_nodes[n][1],_nodes[n][0])

def task91():
    #with open ("/Users/boolker/Desktop/tasks/bio/data/tasks/07/input62.txt", "r") as myfile:
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\09\\input11.txt", "r") as myfile:
        data=myfile.readlines()

    #print(data)
    patterns = []
    for d in data:
        patterns.append(d.replace('\n','').replace('\r',''))

    #print(patterns)

    nodes = {}
    cur_node_ind = 1
    for pattern in patterns:
        cur_root = 1
        cur_dict = nodes
        for p in pattern:
            dct = cur_dict.get(p)
            if dct == None:
                #print('add node')
                cur_node_ind +=1
                cur_dict[p] = [cur_node_ind,{}]
                cur_dict = cur_dict[p][1]
            else:
                cur_dict = dct[1]
    #print(nodes)
    print_nodes(nodes,1)

def match_trie(_str,_nodes):
    matches = 0
    cur_root = _nodes
    for s in _str:
        dct = cur_root.get(s)
        if dct != None:
            cur_root = dct[1]
            if len(cur_root) == 0:
                return 1
        else:
            return 0
    if len(cur_root) > 0:
        return 0

    return matches

def match_trie_ex(_str,_nodes):
    matches = 0
    cur_root = _nodes
    for s in _str:
        dct = cur_root.get(s)
        if dct != None:
            cur_root = dct[1]
            if len(cur_root) == 0:
                return 1
        else:
            return 0

    return 1

def task92():
    #with open ("/Users/boolker/Desktop/tasks/bio/data/tasks/07/input62.txt", "r") as myfile:
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\09\\input21.txt", "r") as myfile:
        data=myfile.readlines()

    #print(data)
    patterns = []
    genom_str = data[0].replace('\n','').replace('\r','')
    for d in data[1:]:
        patterns.append(d.replace('\n','').replace('\r',''))

    #print(patterns)

    nodes = {}
    cur_node_ind = 1
    for pattern in patterns:
        cur_root = 1
        cur_dict = nodes
        for p in pattern:
            dct = cur_dict.get(p)
            if dct == None:
                #print('add node')
                cur_node_ind +=1
                cur_dict[p] = [cur_node_ind,{}]
                cur_dict = cur_dict[p][1]
            else:
                cur_dict = dct[1]
    print(nodes)

    str_res = ''
    for i in xrange(len(genom_str)):
        substr = genom_str[i:]
        if len(substr)>0:
            if match_trie(substr,nodes):
                str_res += str(i)+' '
    print(str_res)

def task93():
    #with open ("/Users/boolker/Desktop/tasks/bio/data/tasks/07/input62.txt", "r") as myfile:
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\09\\input31.txt", "r") as myfile:
        data=myfile.readlines()

    #print(data)
    genom_str = data[0].replace('\n','').replace('\r','')
    print(genom_str)
    #genom_str += '$'
    suffixes = []
    max_len = 200
    for i in xrange(len(genom_str)):
        substr = genom_str[-i:]
        suffixes.append(substr[:max_len])
    print(len(suffixes))

    nodes = {}
    cur_len = 0
    max_len_str = ''
    m_len = 0
    while cur_len < max_len:
        cur_len+=1
        cur_node_ind = 1
        for pattern in suffixes:
            cur_root = 1
            cur_dict = nodes
            created = 0
            sub_pat = pattern[:cur_len]
            for p in sub_pat:
                dct = cur_dict.get(p)
                if dct == None:
                    #print('add node')
                    cur_node_ind +=1
                    cur_dict[p] = [cur_node_ind,{}]
                    cur_dict = cur_dict[p][1]
                    created = 1
                else:
                    cur_dict = dct[1]
            if created == 0:
                if len(pattern[:cur_len]) > len(max_len_str):
                    max_len_str = pattern[:cur_len]
                    m_len = len(max_len_str)
        if m_len != cur_len:
            break

    print(max_len_str)

g_id = 0

def generate_node_id():
    global g_id
    g_id +=1
    return g_id

class STreeNode:
    def __init__(self, _label, _str, _id):
        self.nodes = []
        self.label = _label
        self.str = _str
        self.id = _id

    def print(self,levels = 0, ids = 0, labels = 0, lev = 0):
        #print(self.id,self.label,self.str)
        res_str = ''
        if levels:
            res_str += '\t'*lev
        if ids:
            res_str += str(self.id) + ' '
        if labels:
            res_str += str(self.label) + ' '
        res_str += self.str
        print(res_str)
        for n in self.nodes:
           n.print(levels,ids,labels,lev+1)

    def contains(self,symbols):
        result = 0
        len_sym = len(symbols)
        _syms = []
        for s in symbols:
            if s in self.str:
                if s not in _syms:
                    _syms.append(s)
        return _syms

    def find_deepest_contained(self,symbols):
        result = ''
        if self.contains(symbols):
            return result

        _syms = []
        for n in self.nodes:
            n_syms = n.contains(symbols)
            for s in n_syms:
                if s not in _syms:
                    _syms.append(s)

        if len(_syms) != len(symbols):
            return result

        max_r = ''
        for n in self.nodes:
            r = n.find_deepest_contained(symbols)
            if len(r) > len(max_r):
                max_r = r
        result = self.str + max_r
        return result

    def add_suffix(self, suf_str, _i):
        if len(suf_str) == 0:
            return
        found = 0
        pref_str = suf_str #suf_str[:-1]
        if len(pref_str)>0:
            for n in self.nodes:
                if pref_str.startswith(n.str):
                    found = 1
                    xlen = len(n.str)
                    n.add_suffix(pref_str[xlen:],_i)
        if not found:
            self_len = len(self.str)
            suf_len = len(suf_str)
            added = 0
            sub_suf = suf_str[:-1]
            if len(sub_suf)>0:
                for n in self.nodes:
                    if n.str.startswith(sub_suf):
                        added = 1
                        if n.str[suf_len-1] == suf_str[-1]:
                            #do nothing
                            tmp = 0
                        else:
                            #new edge
                            tmp = 0
                            if n.label == 6:
                                tmp = 0
                            if n.id == 9:
                                tmp = 0
                            nnode1 = STreeNode(_i,suf_str[-1],generate_node_id())
                            #n.nodes.append(nnode1)
                            nnode2 = STreeNode(n.label,n.str[suf_len-1:],generate_node_id())
                            for nn in n.nodes:
                                nnode2.nodes.append(nn)
                            del n.nodes[:]
                            n.nodes.append(nnode1)
                            n.nodes.append(nnode2)
                            n.str = sub_suf
                            n.label = -1

            else:
                for n in self.nodes:
                    if n.str.startswith(suf_str):
                        added = 1

            if not added:
                if self.label >= 0:
                    self.str += suf_str
                else:
                    nnode = STreeNode(_i,suf_str,generate_node_id())
                    self.nodes.append(nnode)

    def construct(self,_genom,array_el,lcp_el,_str):
        xlen = len(_str)
        _len  = len(self.str)
        prev_len = len(_str) - _len
        if prev_len < lcp_el:
            if len(self.nodes) > 0:
                n = self.nodes[-1]
                n_len = len(n.str)
                n.construct(_genom[xlen:],array_el,lcp_el-n_len,_str+self.str)
            else:
                #add
                suffix = _genom[array_el:]
                left_str = self.str[lcp_el:]
                right_str = suffix[lcp_el:]
                self.str = self.str[:lcp_el]

                nnode_left = STreeNode(self.label,left_str,generate_node_id())
                nnode_right = STreeNode(array_el,right_str,generate_node_id())
                self.label = 0
                self.nodes.append(nnode_left)
                self.nodes.append(nnode_right)
        else:
            nnode = STreeNode(array_el,_genom[array_el:],generate_node_id())
            self.nodes.append(nnode)

def task94():
    #with open ("/Users/boolker/Desktop/tasks/bio/data/tasks/07/input62.txt", "r") as myfile:
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\09\\input41.txt", "r") as myfile:
        data=myfile.readlines()

    genom_str = data[0].replace('\n','').replace('\r','')
    print(genom_str)

    root = STreeNode(-1,'',0)

    for i in xrange(len(genom_str)+1):
        #print('Iteration ',i)
        if i == 11:
            tmp = 0
        for j in xrange(i+1):
            suf_str = genom_str[j:i]
            '''if j == 8 and i == 11:
                print('before')
                root.print(1,1)'''
            root.add_suffix(suf_str,j)
            '''if j == 8 and i == 11:
                print('after')
                root.print(1,1)'''

    root.print()


    #for n in nodes:
    #    print(genom_str[n[0]:n[1]])

def task95():
    with open ("/Users/boolker/Desktop/tasks/bio/input95.txt", "r") as myfile:
    #with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\09\\input5.txt", "r") as myfile:
        data=myfile.readlines()

    genom_str_a = data[0].replace('\n','').replace('\r','')
    genom_str_b = data[1].replace('\n','').replace('\r','')
    genom_str = genom_str_a + '$' + genom_str_b + '#'
    print(genom_str)

    #genom_str = genom_str[:-1]
    '''suffixes = []
    for i in xrange(len(genom_str)):
        substr = genom_str[i:]
        suffixes.append(substr)
    print(len(suffixes))
    print(suffixes)'''

    root = STreeNode(-1,'',0)

    for i in xrange(len(genom_str)+1):
        #print('Iteration ',i)
        if i == 11:
            tmp = 0
        for j in xrange(i+1):
            suf_str = genom_str[j:i]
            '''if j == 8 and i == 11:
                print('before')
                root.print(1,1)'''
            root.add_suffix(suf_str,j)

    print('tree done')
    #root.print()
    #result = root.find_deepest_contained(['$','#'])
    #print(result)

def task97():
    #with open ("/Users/boolker/Desktop/tasks/bio/data/tasks/07/input62.txt", "r") as myfile:
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\09\\input71.txt", "r") as myfile:
        data=myfile.readlines()

    genom_str = data[0].replace('\n','').replace('\r','')
    print(genom_str)

    alphabet = ['$','A','C','G','T']
    suf_array = {}
    xlen = len(genom_str)

    doneTill = 1
    step = 1
    sortIndex = []
    sort_row = []
    for i in xrange(xlen):
        ind = alphabet.index(genom_str[i])
        sort_row.append(ind)
    sortIndex.append(sort_row)

    L = []
    while doneTill < xlen:
        del L[:]
        for i in xrange(xlen):
            iTill = -1
            if i+doneTill < xlen-1:
                iTill = sortIndex[step - 1][i+doneTill]
            L.append([sortIndex[step - 1][i],iTill,i])
        L = sorted(L)
        sort_row = [0]*xlen
        for i in xrange(xlen):
            tVal = L[i][2]
            if i > 0:
                tPrev = L[i-1][2]
                if L[i-1][0]==L[i][0] and L[i-1][1]==L[i][1]:
                    sort_row[tVal] = sort_row[tPrev]
                else:
                    sort_row[tVal] = i
            else:
                sort_row[tVal] = i
        sortIndex.append([]+sort_row)

        step+=1
        doneTill *= 2

    #print(L)
    res_str = ''
    for l in L:
        res_str += str(l[2]) + ', '
    f = open("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\09\\out71.txt", "w")
    print(res_str,file=f)
    f.close()

    '''for i in xrange(1,len(genom_str)+1):
        suf_array[genom_str[-i:]] = xlen - i

    res_str = ''
    for key in sorted(suf_array.iterkeys()):
        #print(suf_array[key], key)
        res_str += str(suf_array[key]) + ', '
        #print(suf_array[key])
    print(res_str)

    list = []
    list.append([1,2])
    list.append([1,5])
    list.append([1,2])
    list.append([2,6])
    list.append([2,4])
    list.append([2,5])
    print(list)
    list = sorted(list)
    print(list)'''

def task98():
    with open ("/Users/boolker/Desktop/tasks/bio/data/tasks/09/input8.txt", "r") as myfile:
    #with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\09\\input8.txt", "r") as myfile:
        data=myfile.readlines()

    genom_str = data[0].replace('\n','').replace('\r','')

    suf_array_str = data[1].replace('\n','').replace('\r','').split(',')
    suf_array = [int(i) for i in suf_array_str]

    lcp_str = data[2].replace('\n','').replace('\r','').split(',')
    lcp = [int(i) for i in lcp_str]

    print(genom_str)
    print(suf_array)
    print(lcp)

    for i in xrange(len(suf_array)):
        print(suf_array[i],'\t',lcp[i],'\t',genom_str[suf_array[i]:])

    root = STreeNode(-1,'',0)

    for i in xrange(len(genom_str)):
        root.construct(genom_str,suf_array[i],lcp[i],'')
        print('iter ',i)
        root.print(1,1,1)

    root.print()


if __name__ == "__main__":
   task95()
