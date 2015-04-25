__author__ = 'admin'
'''
#task 1
with open ("c:\\WinPython\\input.txt", "r") as myfile:
    data=myfile.read().replace('\n', '')

print data

datalen = len(data);
print datalen

window = 11
i=0

dict = {};

max_num = 0

while i < datalen:
    substr = data[i:i+window]
    sublen = len(substr)
    if sublen == window:
        dict_val = dict.get(substr,0) + 1
        dict[substr] = dict_val
        if dict_val > max_num:
            max_num = dict_val;
    i = i+1

for d in dict:
    if dict[d] == max_num:
        print d

#task 2
with open ("c:\\WinPython\\input2.txt", "r") as myfile:
    data=myfile.read().replace('\n', '')

print data

datalen = len(data);
print datalen

dict = {"A":"T","C":"G","T":"A","G":"C"};
result = ""
i = 0

while i <= (datalen - 1):
    letter = data[datalen - 1 - i]
    pair = dict.get(letter,0)
    if pair != 0:
      result += pair

    i = i+1

print result

#task 3
with open ("c:\\WinPython\\input3.txt", "r") as myfile:
    data=myfile.readlines()

print "pattern:"
pattern =  data[0].replace('\n','')
window = len(pattern)

print "data:"
dna =  data[1]

datalen = len(dna);
print datalen

i = 0
res_str = ""

while i < datalen:
    substr = dna[i:i+window]
    if substr == pattern:
        res_str += str(i)
        res_str += ' '
    i = i+1

print res_str

#task 4
with open ("c:\\WinPython\\input4.txt", "r") as myfile:
    data=myfile.readlines()

print "dna:"
dna =  data[0].replace('\n','')
dna_size = len(dna)
print dna_size

print "params:"
params =  data[1].split(' ')
print params

window = int(params[0])
datalen = int(params[1])
t = int(params[2])

i=0

dict = {};

max_num = 0
start_i = 0

while start_i < dna_size - datalen:
    tmp_dict = {}
    max_num = 0
    i = 0
    while i < datalen:
        substr = dna[start_i + i: start_i + i + window]
        sublen = len(substr)
        if sublen == window:
            dict_val = tmp_dict.get(substr,0) + 1
            tmp_dict[substr] = dict_val
            if dict_val > max_num:
                max_num = dict_val;
        i = i+1
    start_i = start_i + 1

    for d in tmp_dict:
        if tmp_dict[d] >= t:
            d_val = dict.get(d,0)
            if d_val < tmp_dict[d]:
                dict[d] = tmp_dict[d]

print dict
'''

'''
#task 41
with open ("c:\\WinPython\\input41.txt", "r") as myfile:
    data=myfile.readlines()

print "dna:"
dna =  data[0].replace('\n','')
dna_size = len(dna)
print dna_size


window = 9
datalen = 500
t = 3

i=0

dict = {};

max_num = 0
start_i = 0

while start_i <= dna_size - datalen:
    tmp_dict = {}
    max_num = 0
    i = 0
    while i < datalen:
        substr = dna[start_i + i: start_i + i + window]
        sublen = len(substr)
        if sublen == window:
            dict_val = tmp_dict.get(substr,0) + 1
            tmp_dict[substr] = dict_val
            if dict_val > max_num:
                max_num = dict_val;
        i = i+1
    start_i = start_i + 1
    if start_i % 50000 == 0:
        print start_i

    for d in tmp_dict:
        if tmp_dict[d] >= t:
            d_val = dict.get(d,0)
            if d_val < tmp_dict[d]:
                dict[d] = tmp_dict[d]

print len(dict)
print dict

'''


'''#task 5
with open ("c:\\WinPython\\input5.txt", "r") as myfile:
    data=myfile.readlines()

print "dna:"
dna =  data[0].replace('\n','')
dna_size = len(dna)
print dna_size
g_count = 0;
c_count = 0;
delta = [];
min_delta = dna_size

datalen = len(dna);
print datalen

i = 0
res_str = ""

while i < datalen:
    cur = dna[i]
    if cur == 'C':
        c_count = c_count +1
    elif cur == 'G':
        g_count = g_count + 1
    delta.append(g_count - c_count)
    if delta[i] < min_delta:
        min_delta = delta[i]
    i = i + 1

#print delta
print min_delta

i = 0
while i < dna_size:
    if delta[i] == min_delta:
        print i + 1
    i = i+1

'''

def mismatches(substr, pattern,max_mismatches):
    mismatch_num = 0
    ii = 0
    for ii in xrange(len(substr)):
        if substr[ii] != pattern[ii]:
            mismatch_num = mismatch_num +1
        if mismatch_num >= max_mismatches:
            return max_mismatches

    return mismatch_num

#task 6
'''
def mismatches(substr, pattern):
    mismatch_num = 0
    ii = 0
    for ii in xrange(len(substr)):
        if substr[ii] != pattern[ii]:
            mismatch_num = mismatch_num +1

    return mismatch_num

with open ("c:\\WinPython\\input6.txt", "r") as myfile:
    data=myfile.readlines()

print "pattern:"
pattern =  data[0].replace('\n','')
pattern_size = len(pattern)
print pattern_size

print "dna:"
dna =  data[1].replace('\n','')

print "errors:"
errors =  int(data[2].replace('\n',''))


dna_len = len(dna)


i=0
res_str = ""

while i < dna_len:
    substr = dna[i: i + pattern_size]
    sublen = len(substr)
    if sublen == pattern_size:
        res = mismatches(substr, pattern)
        if res <= errors:
            res_str += str(i)
            res_str += ' '

    i = i+1

print res_str
'''

#task 7

'''def find_with_mismatches(substr, _dict, _err):
    for d in _dict:
        if mismatches(d,substr) <= _err:
            return d
    return substr'''

def find_with_mismatches(substr, _dict, _err):
    for d in _dict:
        if mismatches(d,substr,_err+1) <= _err:
            return 1
    return 0

def find_id_with_mismatches(id, _dict, _window, _err):
    for d in _dict:
        if count_mismatches(d,id,_window,_err+1) <= _err:
            return 1
    return 0

def generate_word(num,digits, _letters):
    word = ''
    for ii in xrange(digits):
        val = (num >> 2*(digits - 1 - ii)) & (3)
        word += _letters[val]
    return word

def generate_id(word, dict_letters):
    id = 0
    l = len(word) - 1
    for w in word:
        id += dict_letters[w]*(4**l)
        l -= 1
    return id

def count_mismatches(_id, pattern_id,_len,max_mismatches):
    mismatch_num = 0
    res_count = _id ^ pattern_id
    i=0
    for i in xrange(_len):
        letter = (res_count >> 2*i) & 0x3
        if letter != 0:
            mismatch_num +=1
        if mismatch_num >= max_mismatches:
            return max_mismatches

    return mismatch_num



def task7():
    with open ("c:\\WinPython\\input7.txt", "r") as myfile:
        data=myfile.readlines()

    print "dna:"
    f_input =  data[0].replace('\n','').split(' ')
    dna = f_input[0]
    dna_size = len(dna)
    print dna_size

    k = f_input[1]

    errors = int(f_input[2])

    window = int(f_input[1])

    i=0

    dict = [];

    max_num = 0
    start_i = 0

    letters = 'ACTG'
    dict_l = {'A':0,'C':1,'T':2,'G':3}

    max_count_with_errors = 0
    in_ids = []

    while i < dna_size:
        substr = dna[i: i + window]
        sublen = len(substr)
        if sublen == window:
            id = generate_id(substr,dict_l)
            in_ids.append(id)
        i = i+1

    print in_ids
    for i in xrange(2**(2*window)):
        count_with_errors = 0
        if i == 46:
            k = 0
        for id in in_ids:
            mis_num = count_mismatches(id,i,window,errors+1)
            if mis_num <= errors:
                count_with_errors = count_with_errors + 1

        if count_with_errors == max_count_with_errors:
            #dict_res = find_id_with_mismatches(i,dict,window,errors)
            #if dict_res == 0:
            dict.append(i)
        elif count_with_errors > max_count_with_errors:
            del dict[:]
            dict.append(i)
            max_count_with_errors = count_with_errors
        i = i+1
        if i % 50000 == 0:
            print i

    for d in dict:
        word  = generate_word(d,window,letters)
        print word, max_count_with_errors

def get_complementary(_id,_len):
    complementary = 0
    for i in xrange(_len):
        num = ((_id >> 2*i) & 0x3)
        num = (~num) & 0x3
        complementary += num*(4**(_len-1-i))

    return complementary

def task8():
    with open ("c:\\WinPython\\input8.txt", "r") as myfile:
        data=myfile.readlines()

    print "dna:"
    dna =  data[0].replace('\n','')

    dna_size = len(dna)
    print dna_size
    params = data[1].replace('\n','').split(' ')

    errors = int(params[1])

    window = int(params[0])

    i=0

    dict = {};
    compl_dict = {}
    res_dict = []

    max_num = 0
    start_i = 0

    letters = 'ACGT'
    dict_l = {'A':0,'C':1,'G':2,'T':3}

    max_count_with_errors = 0
    in_ids = []

    while i < dna_size:
        substr = dna[i: i + window]
        sublen = len(substr)
        if sublen == window:
            id = generate_id(substr,dict_l)
            in_ids.append(id)
        i = i+1

    print in_ids
    for i in xrange(2**(2*window)):
        compl_dict[i] = get_complementary(i,window)
        count_with_errors = 0
        if i == 46:
            k = 0
        for id in in_ids:
            mis_num = count_mismatches(id,i,window,errors+1)
            if mis_num <= errors:
                count_with_errors = count_with_errors + 1

        dict[i] = count_with_errors
        if i % 50000 == 0:
            print i


    #print dict
    #print compl_dict
    cur_num = 0;
    max_num = 0;
    for d in dict:
        cur_num = dict[d] + dict[compl_dict[d]]
        if cur_num == max_num:
            res_dict.append(d)
        elif cur_num > max_num:
            del res_dict[:]
            res_dict.append(d)
            max_num = cur_num

    for d in res_dict:
        word  = generate_word(d,window,letters)
        print word


if __name__ == "__main__":
   task8()
