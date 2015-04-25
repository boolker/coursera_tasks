from __future__ import print_function
import sys
__author__ = 'drummer'

def print_numbers(nums,_f):
    res_str = '('
    for i in xrange(len(nums)):
        n = nums[i]
        if n>0:
            res_str += '+' + str(n)
        else:
            res_str += str(n)
        if i < len(nums) - 1:
            res_str += ' '
    res_str += ')'
    print(res_str,file = _f)

def task81():
    #with open ("/Users/boolker/Desktop/tasks/bio/data/tasks/07/input62.txt", "r") as myfile:
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\08\\input11.txt", "r") as myfile:
        data=myfile.readlines()

    string_a = data[0].replace('\n','')
    string_a = string_a.replace('\r','')
    string_a = string_a.replace('(','')
    string_a = string_a.replace(')','')

    numbers_str = string_a.split(' ')
    numbers = [int(i) for i in numbers_str]

    numbers = [0] + numbers + [len(numbers)+1]
    num_len = len(numbers)
    end = 0
    f = open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\08\\output11.txt", "w")
    while end == 0:
        prev_n = 0
        last_sorted = -sys.maxint - 1
        for j in xrange(1,num_len):
            n = numbers[j]
            if n - prev_n != 1 and last_sorted == (-sys.maxint - 1):
                last_sorted = prev_n
            if abs(n) - abs(last_sorted) == 1:
                #print(numbers[last_sorted+1:j+1])
                numbers[last_sorted+1:j+1] = (numbers[last_sorted+1:j+1][::-1])
                numbers[last_sorted+1:j+1] = [i * (-1) for i in numbers[last_sorted+1:j+1]]
                #print(numbers[last_sorted+1:j+1])
                #print(numbers[1:-1])
                print_numbers(numbers[1:-1],f)
            prev_n = n
        if last_sorted == (-sys.maxint - 1):
            break
    #print(numbers[1:-1])
    f.close()
    return

def task82():
    #with open ("/Users/boolker/Desktop/tasks/bio/data/tasks/07/input62.txt", "r") as myfile:
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\08\\input21.txt", "r") as myfile:
        data=myfile.readlines()

    string_a = data[0].replace('\n','')
    string_a = string_a.replace('\r','')
    string_a = string_a.replace('(','')
    string_a = string_a.replace(')','')

    numbers_str = string_a.split(' ')
    numbers = [int(i) for i in numbers_str]

    numbers = [0] + numbers + [len(numbers)+1]
    num_len = len(numbers)

    prev_n = 0
    breaks = 0
    for n in numbers[1:]:
        if n - prev_n != 1:
            breaks += 1
        prev_n = n

    #print(numbers[1:-1])
    print(breaks)

def build_breakpoint_graph(nums_a,nums_b,max_val):
    graph = []
    graph_row = [0]*(max_val*2)

    graph_row = []
    for i in xrange(2*max_val):
        graph.append([])

    #for i in xrange(max_val*2):
    #    graph.append([]+graph_row)

    for nums in nums_a:
        prev_n = -sys.maxint -1
        start_n = -sys.maxint -1
        for n in nums:
            if prev_n == -sys.maxint -1:
                start_n = n
            else:
                #add
                i = (abs(prev_n)-1)*2+1
                if prev_n < 0:
                    i -= 1
                j = (abs(n)-1)*2
                if n < 0:
                    j += 1
                graph[i].append(j)
                graph[j].append(i)
            prev_n = n
        #add
        i = (abs(prev_n)-1)*2+1
        if prev_n < 0:
            i -= 1
        j = (abs(start_n)-1)*2
        if start_n < 0:
            j += 1
        graph[i].append(j)
        graph[j].append(i)

    for nums in nums_b:
        prev_n = -sys.maxint -1
        start_n = -sys.maxint -1
        for n in nums:
            if prev_n == -sys.maxint -1:
                start_n = n
            else:
                #add
                i = (abs(prev_n)-1)*2+1
                if prev_n < 0:
                    i -= 1
                j = (abs(n)-1)*2
                if n < 0:
                    j += 1
                graph[i].append(j)
                graph[j].append(i)
            prev_n = n
        #add
        i = (abs(prev_n)-1)*2+1
        if prev_n < 0:
            i -= 1
        j = (abs(start_n)-1)*2
        if start_n < 0:
            j += 1
        graph[i].append(j)
        graph[j].append(i)

    return graph


def task83():
    #with open ("/Users/boolker/Desktop/tasks/bio/data/tasks/07/input62.txt", "r") as myfile:
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\08\\input31.txt", "r") as myfile:
        data=myfile.readlines()

    string_a = data[0].replace('\n','')
    string_a = string_a.replace('\r','')
    string_a = string_a.replace('(','')
    string_a = string_a.replace(')',',')

    numbers_a = []
    substrs = string_a.split(',')
    for s in substrs:
        if len(s) > 0:
            numbers_str = s.split(' ')
            numbers = [int(i) for i in numbers_str]
            numbers_a.append([]+numbers)
    #print(numbers_a)


    string_b = data[1].replace('\n','')
    string_b = string_b.replace('\r','')
    string_b = string_b.replace('(','')
    string_b = string_b.replace(')',',')

    numbers_b = []
    substrs = string_b.split(',')
    for s in substrs:
        if len(s) > 0:
            numbers_str = s.split(' ')
            numbers = [int(i) for i in numbers_str]
            numbers_b.append([]+numbers)
    #print(numbers_b)

    max_val = 0
    for nums in numbers_a:
        for n in nums:
            if abs(n) > max_val:
                max_val = abs(n)

    for nums in numbers_b:
        for n in nums:
            if abs(n) > max_val:
                max_val = abs(n)
    print(max_val)

    graph = build_breakpoint_graph(numbers_a,numbers_b,max_val)

    cycles_num = 0
    has_arcs = 1
    while has_arcs == 1:
        start_i = -1
        for i in xrange(max_val*2):
            if len(graph[i]) > 0:
                if start_i == -1:
                    start_i = i
                    break
        if start_i == -1:
            break
        cycles_num +=1
        cur_i = start_i
        start = 1
        while cur_i != start_i or start == 1:
            start = 0
            for j in xrange(max_val*2):
                if len(graph[cur_i]) > 0:
                    cur_j = graph[cur_i][0]
                    del graph[cur_i][0]
                    for j in xrange(len(graph[cur_j])):
                        if graph[cur_j][j] == cur_i:
                            del graph[cur_j][j]
                            break
                    cur_i = cur_j
    print(cycles_num)

    print(max_val - cycles_num)

def revert_kmer(str):
    dict = {'A':'T','C':'G','T':'A','G':'C'}
    res = ''
    for s in str:
        res = dict[s] + res
    return res

def task84():
    with open ("/Users/boolker/Desktop/tasks/bio/data/tasks/08/input4.txt", "r") as myfile:
    #with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\08\\input4.txt", "r") as myfile:
        data=myfile.readlines()

    #print(data)
    k_len = int(data[0].replace('\n',''))

    string_a = data[1].replace('\n','')
    string_a = string_a.replace('\r','')

    string_b = data[2].replace('\n','')
    string_b = string_b.replace('\r','')

    #print(k_len,string_a,string_b)

    substrs_a = []
    substrs_b = []
    unsubstrs_b = []

    for i in xrange(len(string_a)):
        sub_a = string_a[i:i+k_len]
        if len(sub_a) == k_len:
            substrs_a.append(sub_a)

    for i in xrange(len(string_b)):
        sub_b = string_b[i:i+k_len]
        if len(sub_b) == k_len:
            substrs_b.append(sub_b)
            rev_sub_b = revert_kmer(sub_b)
            unsubstrs_b.append(rev_sub_b)

    print(len(substrs_a),len(substrs_b),len(unsubstrs_b))
    '''print(substrs_a[3522],substrs_b[424],unsubstrs_b[424])
    print(substrs_a[3522],substrs_b[424],unsubstrs_b[424])
    print(substrs_a[6310],substrs_b[3190],unsubstrs_b[3190])'''


    for i in xrange(len(substrs_a)):
        start_j = 0
        found_j = []
        while substrs_a[i] in substrs_b[start_j:]:
            j = substrs_b[start_j:].index(substrs_a[i])
        #if substrs_a[i] == substrs_b[j] or substrs_a[i] == unsubstrs_b[j]:
            res_str = '('+str(i)+', '+str(start_j+j)+')'
            #print(res_str,substrs_a[i],substrs_b[start_j+j])
            print(res_str)
            found_j.append(start_j+j)
            start_j += j+1

        start_j = 0
        while substrs_a[i] in unsubstrs_b[start_j:]:
            j = unsubstrs_b[start_j:].index(substrs_a[i])
            if (start_j+j) not in found_j:
                res_str = '('+str(i)+', '+str(start_j+j)+')'
                print(res_str)
                #print(res_str,substrs_a[i],unsubstrs_b[start_j+j],substrs_b[start_j+j])
            start_j += j+1
            #print(res_str)
    '''for i in xrange(len(string_a)):
        sub_a = string_a[i:i+k_len]
        if len(sub_a) == k_len:
            for j in xrange(len(string_b)):
                sub_b = string_b[j:j+k_len]
                if len(sub_b) == k_len:
                    #print(sub_a,sub_b,revert_kmer(sub_b))
                    rev_sub_b = revert_kmer(sub_b)
                    if sub_a == sub_b or sub_a == rev_sub_b:
                        print('(',i,', ',j,')')'''


if __name__ == "__main__":
   task84()
