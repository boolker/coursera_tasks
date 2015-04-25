__author__ = 'drummer'

def mismatches(substr, pattern,max_mismatches):
    mismatch_num = 0
    ii = 0
    for ii in xrange(len(substr)):
        if substr[ii] != pattern[ii]:
            mismatch_num = mismatch_num +1
        if mismatch_num >= max_mismatches:
            return max_mismatches

    return mismatch_num

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

def generate_dna_strings(k):
    results = []
    letters = ['A','C','G','T']
    prev_results = ['A','C','G','T']
    for i in xrange(k-1):
        del results[:]
        for p in prev_results:
            for l in letters:
                results.append(p + l)
        prev_results = [] + results
    return results

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

def get_motifs(m_size, _max_mutations, _sequences):
    motif_size = m_size
    max_mutations = _max_mutations

    all_strings = generate_dna_strings(motif_size)
    errors = max_mutations

    seq_num = len(_sequences)

    results = []

    for a in all_strings:
        count = 0
        for s in _sequences:
            seq_size = len(s)
            for i in xrange(seq_size - motif_size + 1):
                sub_seq = s[i:i+motif_size]
                err = mismatches(a,sub_seq,max_mutations+1)
                if err <= max_mutations:
                    count += 1
                    break
        if count == seq_num:
            results.append(a)
    return results

def task31():
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\03\\input1.txt", "r") as myfile:
        data=myfile.readlines()

    params = data[0].replace('\n','').split(' ')
    motif_size = int(params[0])
    max_mutations = int(params[1])

    sequences = []
    for d in data[1:]:
        sequences.append(d.replace('\n',''))

    results = get_motifs(motif_size,max_mutations,sequences)

    print  results
    r_str = ''
    for r in results:
        r_str += r
        r_str += ' '

    print  r_str

def get_min_motifs(m_size, _max_mutations, _sequences):
    motif_size = m_size
    max_mutations = _max_mutations

    all_strings = generate_dna_strings(motif_size)
    errors = max_mutations

    seq_num = len(_sequences)

    results = []

    min_sum = motif_size*len(_sequences)

    for a in all_strings:
        count = 0
        sum = 0
        for s in _sequences:
            seq_size = len(s)
            min_sub_sum = motif_size
            for i in xrange(seq_size - motif_size + 1):
                sub_seq = s[i:i+motif_size]
                err = mismatches(a,sub_seq,max_mutations+1)
                if err < min_sub_sum:
                    min_sub_sum = err
            sum += min_sub_sum

        if sum < min_sum:
            results = []
            results.append(a)
            min_sum = sum
        elif sum == min_sum:
            results.append(a)
    return results

def task32():
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\03\\input2.txt", "r") as myfile:
        data=myfile.readlines()

    motif_size = int(data[0].replace('\n',''))
    sequences = []
    for d in data[1:]:
        sequences.append(d.replace('\n',''))

    print motif_size, sequences

    '''min_seq = 0
    for s in sequences:
        s_len = len(s)
        if min_seq > 0:
            if s_len < min_seq:
                min_seq = s_len
        else:
            min_seq = s_len

    consensus_stings = ['']
    for i in xrange(min_seq):
        scores = {}
        for s in sequences:
            cur_score = scores.get(s[i],0)
            scores[s[i]] = cur_score + 1
        max_val = 0
        max_letters = []
        for sc in scores:
            cur_val = scores[sc]
            if cur_val > max_val:
                max_val = cur_val
                max_letters = []
                max_letters.append(sc)
            elif cur_val == max_val:
                max_letters.append(sc)

        tmp_consencus_strings = []
        for cs in consensus_stings:
            for l in max_letters:
                tmp_consencus_strings.append(cs + l)
        consensus_stings = tmp_consencus_strings

    print  consensus_stings
    for c in consensus_stings:
        _sum = 0
        for s in sequences:
            _sum += mismatches(c,s,len(c)+1)
        print c, _sum'''

    results = get_min_motifs(motif_size,motif_size,sequences)
    print results
    #for r in results:

import math

def task321():
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\03\\input21.txt", "r") as myfile:
        data=myfile.readlines()

    sequences = []
    for d in data:
        sequences.append(d.replace('\n',''))

    min_seq = 0
    for s in sequences:
        s_len = len(s)
        if min_seq > 0:
            if s_len < min_seq:
                min_seq = s_len
        else:
            min_seq = s_len

    consensus_stings = ['']
    sum_entropy = 0
    seq_len = len(sequences)
    for i in xrange(min_seq):
        scores = {}
        entropy = 0
        for s in sequences:
            cur_score = scores.get(s[i],0)
            scores[s[i]] = cur_score + 1

        for sc in scores:
            p = float(scores[sc])/seq_len
            entropy += p*math.log(p,2)
        entropy *= -1
        sum_entropy += entropy
    print sum_entropy

def task33():
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\03\\input3.txt", "r") as myfile:
        data=myfile.readlines()

    dna = data[0].replace('\n','')
    motif_size = int(data[1].replace('\n',''))
    letters = data[2].replace('\n','').split(' ')
    matrix = []
    for i in xrange(motif_size):
        row_str = data[3+i].replace('\n','').split(' ')
        row = [float(x) for x in row_str]

        matrix.append(row)

    print dna
    print motif_size, letters
    print matrix

    dna = dna.replace('A','0')
    dna = dna.replace('C','1')
    dna = dna.replace('G','2')
    dna = dna.replace('T','3')
    print dna

    max_p = 0
    max_substr = ''
    for i in xrange(len(dna)):
        p = 0
        substr = dna[i:i+motif_size]
        #print substr
        if len(substr) == motif_size:
            for j in xrange(motif_size):
                p += matrix[j][int(substr[j])]
            if p > max_p:
                max_p = p
                max_substr = substr

    max_substr = max_substr.replace('0','A')
    max_substr = max_substr.replace('1','C')
    max_substr = max_substr.replace('2','G')
    max_substr = max_substr.replace('3','T')

    print  'max: ', max_p, max_substr

def form_profile(_sequences,seq_size):
    profile = []

    seq_num = len(_sequences)
    for i in xrange(seq_size):
        profile_col = [0,0,0,0]
        for s in _sequences:
            profile_col[int(s[i])] += 1.0/seq_num
        profile.append(profile_col)

    return profile

def form_profile_pseudo(_sequences,seq_size):
    profile = []

    seq_num = len(_sequences) + 4
    pseudo = 1.0/seq_num
    for i in xrange(seq_size):
        profile_col = [pseudo,pseudo,pseudo,pseudo]
        for s in _sequences:
            profile_col[int(s[i])] += 1.0/seq_num
        profile.append(profile_col)

    return profile

def get_pro_most_motif(_str,profile,_size):
    motif = ''
    max_p = -1
    for i in xrange(len(_str)):
        p = 1
        substr = _str[i:i+_size]
        #print substr
        if len(substr) == _size:
            for j in xrange(_size):
                p *= profile[j][int(substr[j])]
            if p > max_p:
                max_p = p
                motif = substr
    return motif


def get_motif_probability(_motif,profile,_size):
    p = 1
    for j in xrange(_size):
        p *= profile[j][int(_motif[j])]
    return p

def get_pro_most_motif_old(_str,profile,_size):
    motif = ''
    max_p = -1
    for i in xrange(len(_str)):
        p = 1
        substr = _str[i:i+_size]
        #print substr
        if len(substr) == _size:
            for j in xrange(_size):
                p *= profile[j][int(substr[j])]
            if p > max_p:
                max_p = p
                motif = substr
    return motif

def calc_motifs_score(_motifs,_size):
    sum_score = 0
    motifs_size = len(_motifs)
    for i in xrange(_size):
        scores = [0,0,0,0]
        for s in _motifs:
            scores[int(s[i])] += 1
        max_score = 0
        for sc in scores:
            if sc > max_score:
                max_score = sc

        sum_score += (motifs_size - max_score)

    return sum_score

def task34():
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\03\\input4.txt", "r") as myfile:
        data=myfile.readlines()

    params = data[0].replace('\n','').split(' ')
    motif_size = int(params[0])
    str_sequences = []
    for d in data[1:]:
        str_sequences.append(d.replace('\n',''))
    seq_num = len(str_sequences)

    sequences = []
    for s in str_sequences:
        s_ = s.replace('A','0')
        s_ = s_.replace('C','1')
        s_ = s_.replace('G','2')
        s_ = s_.replace('T','3')
        sequences.append(s_)

    print motif_size, seq_num, sequences
    print str_sequences

    best_motifs = []
    #initial best motifs
    for s in sequences:
        best_motifs.append(s[:motif_size])

    best_score = calc_motifs_score(best_motifs,motif_size)
    print best_motifs
    best_i = 0

    s1 = sequences[0]
    for i in xrange(len(s1)):
        print i
        substr = s1[i:i+motif_size]
        if i == 143:
            k=0

        if len(substr) == motif_size:
            motifs = []
            motifs.append(substr)
            for j in xrange(seq_num-1):
                #profile
                seq_str = sequences[j+1]
                profile = form_profile(motifs,motif_size)
                motifs.append(get_pro_most_motif_old(seq_str,profile,motif_size))
            cur_score = calc_motifs_score(motifs, motif_size)
            if cur_score < best_score:
                best_score = cur_score
                best_motifs = motifs
                best_i = i

    print best_motifs
    best_motifs_str = []
    for s in best_motifs:
        s_ = s.replace('0','A')
        s_ = s_.replace('1','C')
        s_ = s_.replace('2','G')
        s_ = s_.replace('3','T')
        best_motifs_str.append(s_)
    print   best_i, best_score, best_motifs_str

    result = ''
    for s in best_motifs_str:
        print s

def task35():
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\03\\input5.txt", "r") as myfile:
        data=myfile.readlines()

    params = data[0].replace('\n','').split(' ')
    motif_size = int(params[0])
    str_sequences = []
    for d in data[1:]:
        str_sequences.append(d.replace('\n',''))
    seq_num = len(str_sequences)

    sequences = []
    for s in str_sequences:
        s_ = s.replace('A','0')
        s_ = s_.replace('C','1')
        s_ = s_.replace('G','2')
        s_ = s_.replace('T','3')
        sequences.append(s_)

    print motif_size, seq_num, sequences
    print str_sequences

    best_motifs = []
    #initial best motifs
    for s in sequences:
        best_motifs.append(s[:motif_size])

    best_score = calc_motifs_score(best_motifs,motif_size)
    print best_motifs
    best_i = 0

    s1 = sequences[0]
    for i in xrange(len(s1)):
        print i
        substr = s1[i:i+motif_size]
        if i == 143:
            k=0

        if len(substr) == motif_size:
            motifs = []
            motifs.append(substr)
            for j in xrange(seq_num-1):
                #profile
                seq_str = sequences[j+1]
                profile = form_profile_pseudo(motifs,motif_size)
                motifs.append(get_pro_most_motif(seq_str,profile,motif_size))
            cur_score = calc_motifs_score(motifs, motif_size)
            if cur_score < best_score:
                best_score = cur_score
                best_motifs = motifs
                best_i = i

    print best_motifs
    best_motifs_str = []
    for s in best_motifs:
        s_ = s.replace('0','A')
        s_ = s_.replace('1','C')
        s_ = s_.replace('2','G')
        s_ = s_.replace('3','T')
        best_motifs_str.append(s_)
    print   best_i, best_score, best_motifs_str

    result = ''
    for s in best_motifs_str:
        print s

def task341():
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\03\\input42.txt", "r") as myfile:
        data=myfile.readlines()

    str_sequences = []
    for d in data:
        str_sequences.append(d.replace('\n',''))
    seq_num = len(str_sequences)

    sequences = []
    for s in str_sequences:
        s_ = s.replace('A','0')
        s_ = s_.replace('C','1')
        s_ = s_.replace('G','2')
        s_ = s_.replace('T','3')
        sequences.append(s_)
    motif_size = len(s)
    print calc_motifs_score(sequences,motif_size)

from random import randrange

def task36():
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\03\\input6.txt", "r") as myfile:
        data=myfile.readlines()

    params = data[0].replace('\n','').split(' ')
    motif_size = int(params[0])
    str_sequences = []
    for d in data[1:]:
        str_sequences.append(d.replace('\n',''))
    seq_num = len(str_sequences)

    sequences = []
    for s in str_sequences:
        s_ = s.replace('A','0')
        s_ = s_.replace('C','1')
        s_ = s_.replace('G','2')
        s_ = s_.replace('T','3')
        sequences.append(s_)

    print motif_size, seq_num, sequences
    print str_sequences

    g_best_motifs = []
    g_best_score = seq_num * motif_size
    N = 300
    for i in xrange(N):
        print i
        best_motifs = []
        for s in sequences:
            start_i = randrange(0,len(s)-motif_size-1)
            best_motifs.append(s[start_i:start_i + motif_size])

        best_score = calc_motifs_score(best_motifs,motif_size)
        #print best_motifs
        best_i = 0

        while 1:
            motifs = []
            profile = form_profile_pseudo(best_motifs,motif_size)
            for s in sequences:
                motifs.append(get_pro_most_motif(s,profile,motif_size))

            cur_score = calc_motifs_score(motifs, motif_size)
            if cur_score < best_score:
                best_score = cur_score
                best_motifs = motifs
                best_i = i
            else:
                break

        if best_score < g_best_score:
            g_best_score = best_score
            g_best_motifs = best_motifs


    #initial best motifs

    print g_best_motifs
    best_motifs_str = []
    for s in g_best_motifs:
        s_ = s.replace('0','A')
        s_ = s_.replace('1','C')
        s_ = s_.replace('2','G')
        s_ = s_.replace('3','T')
        best_motifs_str.append(s_)
    print   g_best_score, best_motifs_str

    result = ''
    for s in best_motifs_str:
        print s

from scipy import stats

def get_random_pick(prob_list):
    custm = stats.rv_discrete(name='custm', values=(range(0,len(prob_list)), prob_list))
    pick = 0
    '''test_pick = randrange(0,1000)
    sum_ = 0
    for i in xrange(len(prob_list)):
        sum_ += prob_list[i]*1000
        if test_pick > sum_:
            pick = i
        else:
            break'''
    pick = custm.rvs()
    return pick

def get_motif_probabilities(profile, dna, motif_size):
    _probabilities = []
    for i in xrange(len(dna)):
        substr = dna[i:i+motif_size]
        if len(substr) == motif_size:
            _probabilities.append(get_motif_probability(substr,profile,motif_size))

    sum_prob = 0
    for p in _probabilities:
        sum_prob += p

    probabilities = []
    for p in _probabilities:
        probabilities.append(p/sum_prob)

    return probabilities

def task37():
    with open ("G:\\Myown\\coursera\\tasks\\bio\\data\\tasks\\03\\input71.txt", "r") as myfile:
        data=myfile.readlines()

    params = data[0].replace('\n','').split(' ')
    motif_size = int(params[0])
    N = int(params[2])
    str_sequences = []
    for d in data[1:]:
        str_sequences.append(d.replace('\n',''))
    seq_num = len(str_sequences)

    sequences = []
    for s in str_sequences:
        s_ = s.replace('A','0')
        s_ = s_.replace('C','1')
        s_ = s_.replace('G','2')
        s_ = s_.replace('T','3')
        sequences.append(s_)

    print motif_size, seq_num, sequences
    print str_sequences

    g_best_motifs = []
    g_best_score = seq_num * motif_size

    N_starts = 2

    for j in xrange(N_starts):

        best_motifs = []
        for s in sequences:
            start_i = randrange(0,len(s)-motif_size-1)
            best_motifs.append(s[start_i:start_i + motif_size])

        best_score = calc_motifs_score(best_motifs,motif_size)

        for i in xrange(N):

            del_str_i = randrange(seq_num)
            #print 'iter ', i, del_str_i

            motifs = []
            for m in xrange(seq_num):
                if m != del_str_i:
                    motifs.append(best_motifs[m])

            profile = form_profile_pseudo(motifs,motif_size)

            probabilities = get_motif_probabilities(profile,sequences[del_str_i],motif_size)

            _pick = get_random_pick(probabilities)
            del_string_dna = sequences[del_str_i]
            best_motifs[del_str_i] = del_string_dna[_pick:_pick+motif_size]

            best_i = 0
            cur_score = calc_motifs_score(best_motifs, motif_size)
            if cur_score < g_best_score:
                g_best_score = cur_score
                g_best_motifs = [] + best_motifs

            #print i, cur_score, best_motifs


    #initial best motifs

    print g_best_motifs
    best_motifs_str = []
    for s in g_best_motifs:
        s_ = s.replace('0','A')
        s_ = s_.replace('1','C')
        s_ = s_.replace('2','G')
        s_ = s_.replace('3','T')
        best_motifs_str.append(s_)
    print   g_best_score, best_motifs_str

    result = ''
    for s in best_motifs_str:
        print s

if __name__ == "__main__":
   task37()
