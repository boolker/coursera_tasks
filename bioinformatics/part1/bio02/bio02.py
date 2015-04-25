__author__ = 'admin'

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


#task 7



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
'''

def task21():
    with open ("c:\\bio\\tasks\\02\\input1.txt", "r") as myfile:
        data=myfile.readlines()

    print "dna:"
    dna =  data[0].replace('\n','')
    print dna

    dna_size = len(dna)

    with open ("c:\\bio\\extra\\RNA_codon_table_1.txt", "r") as extra_myfile:
        extra_data=extra_myfile.readlines()

    codons = {}
    for d in extra_data:
        pair = d.replace('\n','').split(' ')
        if len(pair) == 2:
            if len(pair[1])>0:
                codons[pair[0]] = pair[1]
    #print codons

    result = ''
    codons_num = dna_size/3
    for i in xrange(codons_num):
        c = dna[i*3:(i+1)*3]
        decode = codons.get(c)
        if decode != None:
            result += decode

    print result

def reverse_dna(_dna):
    datalen = len(_dna);

    dict = {"A":"T","C":"G","T":"A","G":"C"};
    result = ""
    i = 0

    while i <= (datalen - 1):
        letter = _dna[datalen - 1 - i]
        pair = dict.get(letter,0)
        if pair != 0:
          result += pair

        i = i+1

    return result

def task22():
    with open ("c:\\bio\\tasks\\02\\input2.txt", "r") as myfile:
        data=myfile.readlines()

    print "dna:"
    dna =  data[0].replace('\n','')
    u_dna = dna.replace('T','U')
    print dna

    rev_dna = reverse_dna(dna)
    u_rev_dna = rev_dna.replace('T','U')
    print rev_dna

    pattern = data[1].replace('\n','')
    print pattern

    pattern_size = len(pattern)

    dna_size = len(dna)

    with open ("c:\\bio\\extra\\RNA_codon_table_1.txt", "r") as extra_myfile:
        extra_data=extra_myfile.readlines()

    codons = {}
    for d in extra_data:
        pair = d.replace('\n','').split(' ')
        if len(pair) == 2:
            if len(pair[1])>0:
                codons[pair[0]] = pair[1]
    #print codons

    print ''
    print u_dna
    print u_rev_dna
    print ''
    result = ''
    start_i = 0
    for start_i in xrange(3):
        result = ''
        codons_num = dna_size/3
        for i in xrange(codons_num):
            c = u_dna[start_i + i*3:start_i + (i+1)*3]
            decode = codons.get(c,' ')
            if decode != None:
                result += decode
       # print result
        i = 0
        for i in xrange(len(result)):
            sub_res = result[i:i+pattern_size]
            if len(sub_res) == pattern_size:
                if sub_res ==  pattern:
                    print dna[start_i+i*3:start_i+(i+pattern_size)*3]

    start_i = 0
    for start_i in xrange(3):
        result = ''
        codons_num = dna_size/3
        for i in xrange(codons_num):
            c = u_rev_dna[start_i + i*3:start_i + (i+1)*3]
            decode = codons.get(c,' ')
            if decode != None:
                result += decode
        #print result
        i = 0
        for i in xrange(len(result)):
            sub_res = result[i:i+pattern_size]
            if len(sub_res) == pattern_size:
                if sub_res ==  pattern:
                    #print i, rev_dna[start_i+i*3:start_i+(i+pattern_size)*3]
                    print dna[dna_size - (start_i+(i+pattern_size)*3):dna_size - (start_i+i*3)]

    #print result

def task221():
    with open ("c:\\bio\\extra\\B_brevis.txt", "r") as myfile:
        data=myfile.readlines()

    dna = ''
    print "dna:"
    for d in data:
        dna += d.replace('\n','')
    #dna =  data[0].replace('\n','')
    u_dna = dna.replace('T','U')
    print len(dna)
    rev_dna = reverse_dna(dna)
    u_rev_dna = rev_dna.replace('T','U')

    pattern = 'VKLFPWFNQY'
    print pattern

    pattern_size = len(pattern)

    dna_size = len(dna)

    with open ("c:\\bio\\extra\\RNA_codon_table_1.txt", "r") as extra_myfile:
        extra_data=extra_myfile.readlines()

    codons = {}
    for d in extra_data:
        pair = d.replace('\n','').split(' ')
        if len(pair) == 2:
            if len(pair[1])>0:
                codons[pair[0]] = pair[1]
    #print codons


    count = 0
    result = ''
    start_i = 0
    for start_i in xrange(3):
        result = ''
        codons_num = dna_size/3
        for i in xrange(codons_num):
            c = u_dna[start_i + i*3:start_i + (i+1)*3]
            decode = codons.get(c,' ')
            if decode != None:
                result += decode
       # print result
        i = 0
        for i in xrange(len(result)):
            sub_res = result[i:i+pattern_size]
            if len(sub_res) == pattern_size:
                if sub_res ==  pattern:
                    #print dna[start_i+i*3:start_i+(i+pattern_size)*3]
                    count += 1

    start_i = 0
    for start_i in xrange(3):
        result = ''
        codons_num = dna_size/3
        for i in xrange(codons_num):
            c = u_rev_dna[start_i + i*3:start_i + (i+1)*3]
            decode = codons.get(c,' ')
            if decode != None:
                result += decode
        #print result
        i = 0
        for i in xrange(len(result)):
            sub_res = result[i:i+pattern_size]
            if len(sub_res) == pattern_size:
                if sub_res ==  pattern:
                    #print i, rev_dna[start_i+i*3:start_i+(i+pattern_size)*3]
                    #print dna[dna_size - (start_i+(i+pattern_size)*3):dna_size - (start_i+i*3)]
                    count += 1

    #print result
    print count

def get_mass(_str):
    masses = {'A':71,'C':103,'D':115,'E':129,'F':147,'G':57,'H':137,'I':113,'K':128,'L':113,'M':131,'N':114,'P':97,'Q':128,'R':156,'S':87,'T':101,'V':99,'W':186,'Y':163}

    mass = 0
    for c in _str:
        mass += masses.get(c,0)

    return mass

def generate_variants(in_str,in_dict,_variants):

    for i in xrange(len(in_dict)):
        cur_str = in_str + in_dict[i]
        #if cur_str not in _variants:
        _variants.append(cur_str)

        generate_variants(cur_str,in_dict[i+1:],_variants)

def generate_variants_cyclo(in_str,_variants):

    j = 1
    l = len(in_str)
    while j < l:
        for i in xrange(l):
            if i+j <= l:
                cur_str = in_str[i:i+j]
            else:
                cur_str = in_str[i:]
                cur_str += in_str[:j-(l-i)]
            #if cur_str not in _variants:
            #cur_str = ''.join(sorted(cur_str))
            #if cur_str not in _variants:
            _variants.append(cur_str)

        j +=1

    _variants.append(in_str)


def generate_variants_cyclo_int(in_masses,_variants):

    j = 1
    l = len(in_masses)
    while j < l:
        for i in xrange(l):
            if i+j <= l:
                cur_str = in_masses[i:i+j]
            else:
                cur_str = in_masses[i:]
                cur_str += in_masses[:j-(l-i)]
            #if cur_str not in _variants:
            #cur_str = ''.join(sorted(cur_str))
            #if cur_str not in _variants:
            _variants.append(cur_str)

        j +=1

    _variants.append(in_masses)
        #generate_variants(cur_str,in_dict[i+1:],_variants)



def task23():
    with open ("c:\\bio\\tasks\\02\\input3.txt", "r") as myfile:
        data=myfile.readlines()

    peptide = data[0].replace('\n','')
    #peptide = peptide.replace('K','Q')
    #peptide = peptide.replace('L','I')
    #print peptide
    #peptide = ''.join(sorted(peptide))
    print peptide
    variants = ['']
    generate_variants_cyclo(peptide,variants)

    print variants

    masses = []
    for v in variants:
        cur_mass = get_mass(v)
        if cur_mass == 186:
            print v

        masses.append(cur_mass)

    masses.sort()
    print masses
    result = ''
    for m in masses:
        result += str(m)
        result += ' '
    print result

def get_variant_mass(variant):
    mass = 0
    for v in variant:
        mass += v
    return mass

def get_spectrum(peptide):
    variants = []
    generate_variants_cyclo(peptide,variants)

    spectrum = [0]
    for v in variants:
        cur_mass = get_variant_mass(v)

        spectrum.append(cur_mass)

    spectrum.sort()
    return  spectrum

def print_variant(variant):
    _str = ''
    l = len(variant)
    for i in xrange(l):
        _str += str(variant[i])
        if i < l-1:
            _str += '-'

    return _str

def task24():
    with open ("c:\\bio\\tasks\\02\\input4.txt", "r") as myfile:
        data=myfile.readlines()

    masses = [71,103,115,129,147,57,137,113,131,114,97,128,156,87,101,99,186,163]
    masses.sort()

    in_masses = data[0].replace('\n','').split(' ')
    print masses
    print in_masses

    result_mass = int(in_masses[-1])

    print result_mass
    new_variants = []
    results = []
    start_masses = []
    for m in in_masses:
        int_m = int(m)
        if int_m != 0:
            if int_m in masses:
                m = [int_m]
                start_masses.append(int_m)
                new_variants.append(m)

    print start_masses

    while len(new_variants)>0:
        tmp_variants = []
        prev_variants = new_variants

        for v in new_variants:
            left_masses = [] + start_masses

            for v_part in v:
                for i in xrange(len(left_masses)):
                    if left_masses[i] == v_part:
                        del left_masses[i]
                        break


            '''for s_m in start_masses:
                mass = get_variant_mass(v) + s_m
                if str(mass) in in_masses:
                    n_v = v + [s_m]
                    #print n_v
                    tmp_variants.append(n_v)
                    if mass == result_mass:
                        results.append(n_v)'''

            for l_m in left_masses:
                mass = get_variant_mass(v) + l_m

                sub_var = v[1:] + [l_m]
                if sub_var not in prev_variants:
                    continue

                if str(mass) in in_masses:
                    n_v = v + [l_m]
                    #print n_v
                    tmp_variants.append(n_v)
                    if mass == result_mass:
                        back_mass = l_m + v[0]
                        if str(back_mass) in in_masses:
                            if n_v not in results:
                                results.append(n_v)


        new_variants = tmp_variants

    out = ''
    for r in results:
        out += print_variant(r)#, ' - ', get_variant_mass(r)
        out += ' '
    print len(results)
    print out

def calc_score(variant, _spectrum,_xspec):
    _score = 0
    #s_var = sorted(variant)
    s_var = variant
    prev_j = 0
    xi = xrange(len(s_var))
    #for j, _s in enumerate(_spectrum):
    '''for i in xi:
        for j,val in  enumerate(_spectrum[prev_j:]):
            if val == s_var[i]:
                if j > prev_j:
                    prev_j = j
                    _score += 1
                    break
        #if v in _spectrum:
         #   _score +=1'''
    res = set(variant) & set(_spectrum)
    _score = len(res)
    return _score

import  time

def task25():
    with open ("c:\\bio\\tasks\\02\\input5.txt", "r") as myfile:
        data=myfile.readlines()

    masses = [71,103,115,129,147,57,137,113,131,114,97,128,156,87,101,99,186,163]
    masses.sort()

    winners = int(data[0].replace('\n',''))
    spectrum_str = data[1].replace('\n','').split(' ')
    #print masses
    spectrum = [int(x) for x in spectrum_str]
    print spectrum

    xspec = enumerate(spectrum)#xrange(len(spectrum))

    result_mass = int(spectrum[-1])

    print result_mass

    top_scores = {}
    leaders = {}

    new_variants = []
    results = []
    start_masses = []

    for m in masses:
        int_m = int(m)
        if int_m != 0:
            m = [int_m]
            score = calc_score(m, spectrum,xspec)
            scores = top_scores.get(score,[])
            scores.append(m)

            top_scores[score] = scores
            #new_variants.append(m)

    print top_scores

    max_score = 0
    results = []
    iter = 1
    while len(top_scores)>0:
        spec_dur = 0
        score_dur = 0
        mass_dur = 0
        start_iter = time.clock()
        total_count = 0
        bOverdraft = 0
        score_vals = top_scores.keys()
        score_vals = reversed(sorted(score_vals))  # order them in some way

        prev_variants = []
        for top in score_vals:
            if bOverdraft == 0:
                total_count += len(top_scores[top])
                prev_variants += top_scores[top]
                if total_count > winners:
                    bOverdraft = 1
            else:
                del top_scores[top]

        tmp_variants = prev_variants
        #print tmp_variants

        top_scores.clear()
        iter +=1

        for v in tmp_variants:
            left_masses = [] + masses

            for l_m in left_masses:
                t0 = time.clock()
                mass = get_variant_mass(v) + l_m
                mass_dur += (time.clock() - t0)
                if mass > result_mass:
                    continue
                cur_var = v + [l_m]
                t0 = time.clock()
                cur_spectrum = get_spectrum(cur_var)
                spec_dur += (time.clock() - t0)

                t0 = time.clock()
                cur_score = calc_score(cur_spectrum,spectrum,xspec)
                score_dur += (time.clock() - t0)

                if mass == result_mass:
                    if cur_score > max_score:
                        max_score = cur_score
                        results = [cur_var]
                    elif cur_score == max_score:
                        results.append(cur_var)


                scores = top_scores.get(cur_score,[])
                scores.append(cur_var)
                top_scores[cur_score] = scores

        iter_dur = time.clock() - start_iter
        print iter, iter_dur, mass_dur, spec_dur, score_dur,  len(tmp_variants)

    print results
    out = ''
    for r in results:
        out += print_variant(r)#, ' - ', get_variant_mass(r)
        out += ' '
    print len(results)
    print out

def get_convolution(_spectrum):
    result = []
    for i,i_val in  enumerate(_spectrum):
        for j,j_val in  enumerate(_spectrum):
            if i>j:
                if i_val > j_val:
                    result.append(i_val - j_val)

    result.sort()
    return result

def task26():
    with open ("c:\\bio\\tasks\\02\\input6.txt", "r") as myfile:
        data=myfile.readlines()

    spectrum_str = data[0].replace('\n','').split(' ')
    #print masses
    spectrum = [int(x) for x in spectrum_str]
    spectrum.sort()

    result = get_convolution(spectrum)
    print result

    result_str = ''
    for r in result:
        result_str += str(r)
        result_str += ' '
    print result_str

def task27():
    with open ("c:\\bio\\tasks\\02\\input7.txt", "r") as myfile:
        data=myfile.readlines()

    spectrum_str = data[2].replace('\n','').split(' ')
    spectrum = [int(x) for x in spectrum_str]
    spectrum.sort()

    convolution_threshhold = int(data[0].replace('\n',''))

    leader_threshhold = int(data[1].replace('\n',''))

    print convolution_threshhold, leader_threshhold, spectrum

    convolution = get_convolution(spectrum)
    print convolution

    top_convolution = {}

    '''for c in convolution:
        if c >=57 and c<=200:
            convs = top_convolution.get(c,[])
            convs.append(c)

            top_convolution[c] = convs'''

    for i in xrange(57,201):
        el = [x for x in convolution if x == i]
        el_count = len(el)
        if el_count > 0:
            els = top_convolution.get(el_count,[])
            els.append(i)
            top_convolution[el_count] = els

    conv_vals = top_convolution.keys()
    conv_vals = reversed(sorted(conv_vals))  # order them in some way

    start_masses = []
    bOverdraft = 0
    total_count = 0
    for top in conv_vals:
        if bOverdraft == 0:
            total_count += len(top_convolution[top])
            start_masses += top_convolution[top]
            if total_count > convolution_threshhold:
                bOverdraft = 1
        else:
            break

    print ' top convolution: ', top_convolution
    start_masses = sorted(start_masses)
    print 'start masses: ', len(start_masses), start_masses

    #main alg
    xspec = enumerate(spectrum)#xrange(len(spectrum))

    result_mass = int(spectrum[-1])

    print result_mass

    top_scores = {}
    leaders = {}

    new_variants = []
    results = []
    #start_masses = []

    for m in start_masses:
        int_m = int(m)
        if int_m != 0:
            m = [int_m]
            score = calc_score(m, spectrum,xspec)
            scores = top_scores.get(score,[])
            scores.append(m)

            top_scores[score] = scores
            #new_variants.append(m)

    print top_scores

    max_score = 0
    results = []
    iter = 1
    while len(top_scores)>0:
        spec_dur = 0
        score_dur = 0
        mass_dur = 0
        start_iter = time.clock()
        total_count = 0
        bOverdraft = 0
        score_vals = top_scores.keys()
        score_vals = reversed(sorted(score_vals))  # order them in some way

        prev_variants = []
        for top in score_vals:
            if bOverdraft == 0:
                total_count += len(top_scores[top])
                prev_variants += top_scores[top]
                if total_count > leader_threshhold:
                    bOverdraft = 1
            else:
                del top_scores[top]

        tmp_variants = prev_variants
        #print tmp_variants

        top_scores.clear()
        iter +=1

        for v in tmp_variants:
            left_masses = [] + start_masses

            for l_m in left_masses:
                t0 = time.clock()
                mass = get_variant_mass(v) + l_m
                mass_dur += (time.clock() - t0)
                if mass > result_mass:
                    continue
                cur_var = v + [l_m]
                t0 = time.clock()
                cur_spectrum = get_spectrum(cur_var)
                spec_dur += (time.clock() - t0)

                t0 = time.clock()
                cur_score = calc_score(cur_spectrum,spectrum,xspec)
                score_dur += (time.clock() - t0)

                if mass == result_mass:
                    if cur_score > max_score:
                        max_score = cur_score
                        results = [cur_var]
                    elif cur_score == max_score:
                        results.append(cur_var)


                scores = top_scores.get(cur_score,[])
                scores.append(cur_var)
                top_scores[cur_score] = scores

        iter_dur = time.clock() - start_iter
        print iter, iter_dur, mass_dur, spec_dur, score_dur,  len(tmp_variants)

    print results
    out = ''
    for r in results:
        out += print_variant(r)#, ' - ', get_variant_mass(r)
        out += ' '
    print len(results)
    print out


if __name__ == "__main__":
   task27()
