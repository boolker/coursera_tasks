from __future__ import print_function
import sys, os, math, random
import copy
__author__ = 'drummer'

import matplotlib.pyplot as plt

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




def to_int(s):
    try:
        return int(s)
    except ValueError:
        return -1

def get_distance_to_centers(_point,_points,_centers,_dim):
    res = sys.maxint
    for c in _centers:
        dist = 0
        for i in xrange(_dim):
            dist += (_point[i]-_points[c][i])*(_point[i]-_points[c][i])
        dist = math.sqrt(dist)
        if dist < res:
            res = dist
    return res

def get_distance(_a,_b,_dim):
    res = 0
    for i in xrange(_dim):
        res += (_a[i]-_b[i])*(_a[i]-_b[i])
        #dist = math.sqrt(dist)
    res = math.pow(res,1.0/2)
    return res

def get_distance_to_centers_ex(_point,_points,_centers,_dim):
    res = sys.maxint
    for c in _centers:
        dist = 0        
        for i in xrange(_dim):
            dist += (_point[i]-c[i])*(_point[i]-c[i])
        #dist = math.sqrt(dist)
        dist = math.pow(dist,1.0/2)
        if dist < res:
            res = dist    
    return res

def distortion(_points,_centers,_dim):
    res = 0
    n = len(_points)
    print(n)
    for i in xrange(n):
        dist = get_distance_to_centers_ex(_points[i],_points,_centers,_dim)
        res += dist*dist
    print(res)
    res = res/n
    return res

def max_distance(_points,_centers,_dim):
    max_dist = 0    
    for i in xrange(len(_points)):
        pt_dist = get_distance_to_centers_ex(_points[i],_points,_centers,_dim)
        if pt_dist > max_dist:
            max_dist = pt_dist            
    return max_dist

def task71():
    # Implement the FarthestFirstTraversal clustering heuristic.
    # Input: Integers k and m followed by a set of points Data in m-dimensional space.
    # Output: A set Centers consisting of k points (centers) resulting from applying
    # FarthestFirstTraversal(Data, k), where the first point from Data is chosen as the
    # first center to initialize the algorithm.
    input_file_name = os.getcwd() + "/part2/data/07/input12.txt"
    #input_file_name = "/Users/boolker/Desktop/tasks/bio02/data/04/input034.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()
    
    k = int((data[0].replace('\n','').split(' '))[0])
    dim = int((data[0].replace('\n','').split(' '))[1])
    print(k)
    points = []
    for d in data[1:]:
        point_strings = d.replace('\n','').split(' ')
        points.append([float(x) for x in point_strings])
    #print(points)

    centers = [0]
    while len(centers) < k:
        max_dist = 0
        max_i = -1
        for i in xrange(len(points)):
            if i in centers:
                continue
            pt_dist = get_distance_to_centers(points[i],points,centers,dim)
            if pt_dist > max_dist:
                max_dist = pt_dist
                max_i = i
        centers.append(max_i)        
    #print(centers)
    for c in centers:
        str_c = ""
        for i in xrange(dim):
            str_c += str(points[c][i]) + " "
        print(str_c)

def task720():  
    points = [[1.0,6.0],[1.0,3.0],[3.0,4.0],[5.0,2.0],[5.0,6.0],[7.0,1.0],[8.0,7.0],[10.0,3.0]]
    centers = [[3.0,4.5],[6.0,1.5],[9.0,5.0]]
    centers1 = [[5/3.0,13.0/3],[6.5,6.5],[22.0/3,2.0]]

    print(max_distance(points,centers,2),distortion(points,centers,2))
    print(max_distance(points,centers1,2),distortion(points,centers1,2))

def task72():
    # Solve the Squared Error Distortion Problem.
    # Input: Integers k and m, followed by a set of centers Centers and a set of points Data.
    # Output: The squared error distortion Distortion(Data, Centers).
    input_file_name = os.getcwd() + "/part2/data/07/input22.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    k = int((data[0].replace('\n','').split(' '))[0])
    dim = int((data[0].replace('\n','').split(' '))[1])

    centers = []
    points = []
    for i in xrange(k):
        cnt_string = data[i+1].replace('\n','').split(' ')
        centers.append([float(x) for x in cnt_string])

    for d in data[k+2:]:        
        pt_string = d.replace('\n','').split(' ')
        points.append([float(x) for x in pt_string])

    #print(centers)
    #print(points)
    res = distortion(points,centers,dim)
    print(res)

def calc_centers(_clusters,_dim):
    _centers = []
    for cluster in _clusters:
        c_len = len(cluster)
        cnt = [0 for i in xrange(_dim)]
        for i in xrange(_dim):
            for c in cluster:            
                cnt[i] += c[i]
            cnt[i] = cnt[i]/c_len
        _centers.append(cnt)

    return _centers

def calc_centers_delta(_old,_cur,_dim):
    delta = 0
    _k = len(_old)
    for i in xrange(_k):
        delta += get_distance(_cur[i],_old[i],_dim)
    delta = delta/_k
    return delta

def init_centers(_points,_k,_dim):
    centers = []
    n = len(_points)
    centers.append(random.randrange(0,n))    
    while len(centers) < _k:
        rand_pt = 0
        distances = []
        sum_dist = 0
        for i in xrange(len(_points)):
            if i in centers:
                distances.append(0)
                continue
            dist = get_distance_to_centers(_points[i],_points,centers,_dim)
            distances.append(dist)
            sum_dist += dist
            
        #print("sum = ",sum_dist)
        #for i in xrange(len(_points)):
        #    print(i,distances[i]/sum_dist)
        choice = random.random()
        choice_num = -1
        cur_val = 0
        for d in distances:        
            cur_val += d/sum_dist
            choice_num +=1
            if cur_val > choice:
                break
        #print(choice, choice_num)
        centers.append(choice_num)
    #print(centers)
    return centers

def lloyd_algorythm(_points,_k,_dim):
    old_centers = []
    cur_centers = []
    rand_centers = init_centers(_points,_k,_dim)
    for i in xrange(_k):        
        #cur_centers.append(_points[rand_centers[i]])
        cur_centers.append(_points[i])
        old_centers.append([0 for i in xrange(_dim)])

    delta = calc_centers_delta(old_centers,cur_centers,_dim)
    #print("delta is ", delta)
    while delta > 0.0000001:
        clusters = []
        for i in xrange(_k):        
            clusters.append([])

        for i in xrange(len(_points)):
            min_dist = sys.maxint
            min_j = -1
            for j in xrange(_k):
                dist = get_distance(_points[i],cur_centers[j],_dim)
                #print(i,j,dist)
                if dist < min_dist:
                    min_dist = dist
                    min_j = j
            clusters[min_j].append(_points[i])
        #print(clusters)
        old_centers = cur_centers[:]
        cur_centers = calc_centers(clusters,_dim)
        #print(cur_centers)
        delta = calc_centers_delta(old_centers,cur_centers,_dim)
        #print("delta is ", delta)

    return cur_centers

def task73():
    # Implement the Lloyd algorithm for k-means clustering.
    # Input: Integers k and m followed by a set of points Data in m-dimensional space.
    # Output: A set Centers consisting of k points (centers) resulting from applying the
    # Lloyd algorithm to Data and Centers, where the first k points from Data are selected
    # as the first k centers.

    input_file_name = os.getcwd() + "/part2/data/07/input3.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    k = int((data[0].replace('\n','').split(' '))[0])
    dim = int((data[0].replace('\n','').split(' '))[1])

    
    points = []
    for d in data[1:]:        
        pt_string = d.replace('\n','').split(' ')
        points.append([float(x) for x in pt_string])

    print(points)
    centers = lloyd_algorythm(points,k,dim)
    
    print(centers)
    for c in centers:
        str_c = ""
        for i in xrange(dim):
            str_c += "{0:.3f}".format(c[i])
            str_c += " "
        print(str_c)

def task731():
    # Implement the Lloyd algorithm for k-means clustering.
    # Input: Integers k and m followed by a set of points Data in m-dimensional space.
    # Output: A set Centers consisting of k points (centers) resulting from applying the
    # Lloyd algorithm to Data and Centers, where the first k points from Data are selected
    # as the first k centers.

    input_file_name = os.getcwd() + "/part2/data/07/input32.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    k = 6
    dim = 7

    
    points = []
    for d in data[1:]:        
        pt_string = d.replace('\n','').split('\t')
        points.append([float(x) for x in pt_string[3:]])

    #print(points)
    centers = lloyd_algorythm(points,k,dim)
    
    print(centers)
    for c in centers:
        str_c = ""
        for i in xrange(dim):
            str_c += "{0:.3f}".format(c[i])
            str_c += " "
        print(str_c)

def soft_k_means(_points,_k,_dim,_beta,_inter_num):
    old_centers = []
    cur_centers = []
    
    hidden_matrix = []
    matrix_row = [0 for i in xrange(len(_points))]
    
    for i in xrange(_k):
        cur_centers.append(_points[i])
        hidden_matrix.append(matrix_row[:])

    print("start") 
    for _iter in xrange(_inter_num):
        print(_iter)
        #E-step
        for j in xrange(len(_points)):
            sum_mtx_elements = 0
            for i in xrange(_k):
                hidden_matrix[i][j] = math.exp(-1*_beta*get_distance(cur_centers[i],_points[j],_dim))
                sum_mtx_elements += hidden_matrix[i][j]
            for i in xrange(_k):
                hidden_matrix[i][j] = hidden_matrix[i][j]/sum_mtx_elements
    
        #print(hidden_matrix)
        #M-step
        for i in xrange(_k):
            cur_centers[i] = [0 for m in xrange(_dim)]
            sum_row = 0
            for k in xrange(len(_points)):
                sum_row += hidden_matrix[i][k]
                for j in xrange(_dim):
                    cur_centers[i][j] += hidden_matrix[i][k]*_points[k][j]                    

            for j in xrange(_dim):
                cur_centers[i][j] = cur_centers[i][j]/sum_row

        #print(cur_centers)

    return cur_centers

def soft_k_means_test():
    old_centers = []
    cur_centers = []

    _points = [[-3],[-2],[0],[2],[3]]
    _k = 2
    _dim = 1
    _beta = 1    
    _inter_num = 100
    
    hidden_matrix = []
    matrix_row = [0 for i in xrange(len(_points))]
    
    for i in xrange(_k):
        hidden_matrix.append(matrix_row[:])

    cur_centers.append([-2.5])
    cur_centers.append([2.5])
    

    print("start") 
    for _iter in xrange(_inter_num):
        print(_iter)
        #E-step
        for j in xrange(len(_points)):
            sum_mtx_elements = 0
            for i in xrange(_k):
                hidden_matrix[i][j] = math.exp(-1*_beta*get_distance(cur_centers[i],_points[j],_dim))
                sum_mtx_elements += hidden_matrix[i][j]
            for i in xrange(_k):
                hidden_matrix[i][j] = hidden_matrix[i][j]/sum_mtx_elements
    
        #print(hidden_matrix)
        #M-step
        for i in xrange(_k):
            cur_centers[i] = [0 for m in xrange(_dim)]
            sum_row = 0
            for k in xrange(len(_points)):
                for j in xrange(_dim):
                    cur_centers[i][j] += hidden_matrix[i][k]*_points[k][j]
                    sum_row += hidden_matrix[i][k]

            for j in xrange(_dim):
                cur_centers[i][j] = cur_centers[i][j]/sum_row

        #print(cur_centers)

    return cur_centers

def task81():
    # Implement the Lloyd algorithm for k-means clustering.
    # Input: Integers k and m followed by a set of points Data in m-dimensional space.
    # Output: A set Centers consisting of k points (centers) resulting from applying the
    # Lloyd algorithm to Data and Centers, where the first k points from Data are selected
    # as the first k centers.

    input_file_name = os.getcwd() + "/part2/data/08/input12.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    k = int((data[0].replace('\n','').split(' '))[0])
    dim = int((data[0].replace('\n','').split(' '))[1])

    beta = float(data[1].replace('\n',''))
    print(k,dim,beta)

    points = []
    for d in data[2:]:        
        pt_string = d.replace('\n','').split(' ')
        points.append([float(x) for x in pt_string])

    #print(points)
    centers = soft_k_means(points,k,dim,beta,100)
    #centers = soft_k_means_test()
    for cnt in centers:
        str_cnt = ""
        for i in xrange(dim):
            str_cnt += "{0:.3f}".format(cnt[i]) + " "
        print(str_cnt)

def task811():
    # Implement the Lloyd algorithm for k-means clustering.
    # Input: Integers k and m followed by a set of points Data in m-dimensional space.
    # Output: A set Centers consisting of k points (centers) resulting from applying the
    # Lloyd algorithm to Data and Centers, where the first k points from Data are selected
    # as the first k centers.

    input_file_name = os.getcwd() + "/part2/data/07/input32.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    k = 6
    dim = 7

    
    points = []
    for d in data[1:]:        
        pt_string = d.replace('\n','').split('\t')
        points.append([float(x) for x in pt_string[3:]])

    #print(points)
    centers = lloyd_algorythm(points,k,dim)
    
    print(centers)
    for c in centers:
        str_c = ""
        for i in xrange(dim):
            str_c += "{0:.3f}".format(c[i])
            str_c += " "
        print(str_c)

def build_empty_graph(_graph, _vnum):
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

def add_edge(_tree,_start,_end,_len):
    #print("add edge from ", _start," to ",_end,"with len = ",_len)    
    N = _tree[0][0]
    _tree[_start+2].append([_end,_len])
    _tree[N+_end+2].append([_start])
    _tree[0][1+ _start] += 1
    _tree[1][1+ _end] += 1


def cluster_distance_avg(_distances, cluster_a, cluster_b):
    res = 0
    for a in cluster_a:
        for b in cluster_b:
            res += _distances[a][b]
    res = res / (len(cluster_a)*len(cluster_b))
    return res

def cluster_distance_min(_distances, cluster_a, cluster_b):
    res = sys.maxint
    for a in cluster_a:
        for b in cluster_b:
            if _distances[a][b] < res:
                res = _distances[a][b]
    
    return res

def hierarchical_clustering(_distances,_size):
    clusters = []    
    cluster_ids = []
    ages = []
    
    result_tree = []    
    for i in xrange(_size):
        clusters.append([i])    
        cluster_ids.append(i)
        ages.append(0)

    build_empty_graph(result_tree,_size)    

    cluster_distances = []
    for d in _distances:
        cluster_distances.append(d[:])
    
    while len(cluster_ids)>1:
        # 1. find min distance between clusters
        min_dist = sys.maxint
        min_i = -1
        min_j = -1
        clusters_len = len(cluster_ids)
        for i in xrange(clusters_len):
            for j in xrange(clusters_len):
                if i<j:
                    c_i = cluster_ids[i]
                    c_j = cluster_ids[j]                    
                    c_distance = cluster_distance(_distances,clusters[c_i],clusters[c_j])
                    if c_distance< min_dist:
                        min_dist = c_distance
                        min_i = c_i
                        min_j = c_j
        #print(min_i,min_j,min_dist)
        if min_i <0 or min_j <0:
            print("error searching minimum distance")
            return []
        # 2. create cluster
        new_cluster = clusters[min_i][:] + clusters[min_j][:]    
        #print(new_cluster)            
        str_c = ""
        for c in new_cluster:
            str_c += str(c+1) + " "
        print(str_c)
        clusters.append(new_cluster)
        ages.append(min_dist/2.0)
        
        v_num = add_vertex_alone(result_tree)
        cluster_ids.append(v_num)
                
        cluster_ids.remove(min_j)
        cluster_ids.remove(min_i)
        
        add_edge(result_tree,min_i,v_num,0)
        add_edge(result_tree,min_j,v_num,0)
        add_edge(result_tree,v_num,min_i,0)
        add_edge(result_tree,v_num,min_j,0)

    #print(clusters)           
    
    return result_tree

def task91():
    # Solve the Probability of a Hidden Path Problem.
    # Given: A hidden path pi followed by the states States and transition matrix Transition of an HMM
    # ( States, Transition, Emission).
    # Return: The probability of this path, Pr().

    input_file_name = os.getcwd() + "/part2/data/09/input11.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    path = data[0].replace('\n','')
    print(path)
    elems_str = data[2].replace('\n','').replace('\t',' ')        
    elems = [e for e in elems_str.split(' ')]
    print(elems)
    elem_size = len(elems)
    mtx = []
    for i in xrange(elem_size):
        mtx_str = data[5+i].replace('\n','').replace('\t',' ')        
        mtx_els = [float(i) for i in (mtx_str.split())[1:]]
        mtx.append(mtx_els)

    res = 0.5
    for i in xrange(len(path)):
        if i == 0:
            continue
        
        c_from = path[i-1]
        c_to = path[i]
        from_ind = elems.index(c_from)
        to_ind = elems.index(c_to)        
        res = res * mtx[from_ind][to_ind]
    print(res)

def task92():
    # Solve the Probability of an Outcome Given a Hidden Path Problem.
    # Input: A string x, followed by the alphabet from which x was constructed, followed by
    # a hidden path pi, followed by the states States and emission matrix Emission of an HMM
    # (sigma, States, Transition, Emission).
    # Output: The conditional probability Pr(x|pi) that x will be emitted given that the HMM
    # follows the hidden path pi.

    input_file_name = os.getcwd() + "/part2/data/09/input21.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    dest_path = data[0].replace('\n','')    
    elems_str = data[2].replace('\n','').replace('\t',' ')        
    dest_elems = [e for e in elems_str.split(' ')]

    src_path = data[4].replace('\n','')
    elems_str = data[6].replace('\n','').replace('\t',' ')        
    src_elems = [e for e in elems_str.split(' ')]
    
    print(dest_path)
    print(src_path)
    print(dest_elems)
    print(src_elems)
    
    src_elem_size = len(src_elems)
    mtx = []
    for i in xrange(src_elem_size):
        mtx_str = data[9+i].replace('\n','').replace('\t',' ')        
        mtx_els = [float(i) for i in (mtx_str.split())[1:]]
        mtx.append(mtx_els)

    print(mtx)

    res = 1
    for i in xrange(len(src_path)):        
        c_from = src_path[i]
        c_to = dest_path[i]
        from_ind = src_elems.index(c_from)
        to_ind = dest_elems.index(c_to)        
        res = res * mtx[from_ind][to_ind]
    print(res)


def get_optimal_path(_transition,_emission,_path,_src,_dest):    

    res_mtx = []
    path_points = []

    src_elem_size = len(_src)
    for i in xrange(src_elem_size):
        res_mtx.append([0 for j in xrange(len(_path))])
        path_points.append([0 for j in xrange(len(_path))])

    for _iter in xrange(len(_path)):        
        path_ind = _dest.index(_path[_iter])
        if _iter == 0:
            res = 0.5            
            for i in xrange(len(_src)):
                res_mtx[i][_iter] = 1*0.5*_emission[i][path_ind]
        else:
            for i in xrange(len(_src)):
                max_val = 0
                max_ind = -1
                for j in xrange(len(_src)):
                    cur_val = res_mtx[j][_iter-1]*_transition[j][i]*_emission[i][path_ind]
                    #if _iter == 4:
                    #    print(_iter,i,j,res_mtx[j][_iter-1],_transition[j][i],_emission[i][path_ind],cur_val)
                    if cur_val > max_val:
                        max_val = cur_val
                        max_ind = j
                res_mtx[i][_iter] = max_val
                path_points[i][_iter] = max_ind

    for s in res_mtx:
        str_s = ""
        for _s in s:
            if _s>0:
                #str_s += str(math.log(_s,2)) + " "
                str_s += str(_s) + " "
            else:
                str_s += "0"
        print(str_s)

    for s in path_points:
        str_s = ""
        for _s in s:            
            str_s += str(_s) + " "
            
        print(str_s)

    res_str = ""
    cur_max_ind = -1
    max_val = 0        
    next_point = -1
    for i in xrange(len(_src)):                
        if res_mtx[i][-1] > max_val:
            max_val = res_mtx[i][-1]
            cur_max_ind = i    
    res_str += _src[cur_max_ind]

    for i in xrange(len(_path)-2,-1,-1):
        next_point = path_points[cur_max_ind][i+1]
        res_str = _src[next_point] + res_str
        cur_max_ind = next_point

    print(res_str)

def task93():
    # Implement the Viterbi algorithm solving the Decoding Problem.
    # Input: A string x, followed by the alphabet from which x was constructed,
    # followed by the states States, transition matrix Transition, and emission matrix
    # Emission of an HMM (sigma, States, Transition, Emission).
    # Output: A path that maximizes the (unconditional) probability Pr(x, pi) over all possible paths pi.

    input_file_name = os.getcwd() + "/part2/data/09/input32.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    path = data[0].replace('\n','')    
    elems_str = data[2].replace('\n','').replace('\t',' ')        
    dest_elems = [e for e in elems_str.split(' ')]

    elems_str = data[4].replace('\n','').replace('\t',' ')        
    src_elems = [e for e in elems_str.split(' ')]
    
    print(path)    
    print(dest_elems)
    print(src_elems)
    
    src_elem_size = len(src_elems)
    transition_mtx = []
    for i in xrange(src_elem_size):
        mtx_str = data[7+i].replace('\n','').replace('\t',' ')        
        mtx_els = [float(i) for i in (mtx_str.split())[1:]]
        transition_mtx.append(mtx_els)

    print(transition_mtx)

    emission_mtx = []
    dest_elem_size = len(dest_elems)
    for i in xrange(src_elem_size):
        ind = 9 + src_elem_size + i        
        mtx_str = data[ind].replace('\n','').replace('\t',' ')        
        mtx_els = [float(i) for i in (mtx_str.split())[1:]]
        emission_mtx.append(mtx_els)


    print(emission_mtx)

    get_optimal_path(transition_mtx,emission_mtx,path,src_elems,dest_elems)    


def likehood(_transition,_emission,_path,_src,_dest):    

    res_mtx = []
    
    src_elem_size = len(_src)
    for i in xrange(src_elem_size):
        res_mtx.append([0 for j in xrange(len(_path))])
    
    for _iter in xrange(len(_path)):        
        path_ind = _dest.index(_path[_iter])
        if _iter == 0:
            res = 0.5            
            for i in xrange(len(_src)):
                res_mtx[i][_iter] = 1*0.5*_emission[i][path_ind]
        else:
            for i in xrange(len(_src)):
                max_val = 0
                max_ind = -1
                for j in xrange(len(_src)):
                    cur_val = res_mtx[j][_iter-1]*_transition[j][i]*_emission[i][path_ind]
                    #if _iter == 4:
                    #    print(_iter,i,j,res_mtx[j][_iter-1],_transition[j][i],_emission[i][path_ind],cur_val)
                    res_mtx[i][_iter] += cur_val
                   
                
    print(res_mtx)
    prob = 0
    for i in xrange(len(_src)):
        prob += res_mtx[i][-1]
    print(prob)
                

def task94():
    # Solve the Outcome Likelihood Problem.
    # Input: A string x, followed by the alphabet from which x was constructed,
    # followed by the states States, transition matrix Transition, and emission matrix
    # Emission of an HMM (sum, States, Transition, Emission).
    # Output: The probability Pr(x) that the HMM emits x.

    input_file_name = os.getcwd() + "/part2/data/09/input41.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    path = data[0].replace('\n','')    
    elems_str = data[2].replace('\n','').replace('\t',' ')        
    dest_elems = [e for e in elems_str.split(' ')]

    elems_str = data[4].replace('\n','').replace('\t',' ')        
    src_elems = [e for e in elems_str.split(' ')]
    
    print(path)    
    print(dest_elems)
    print(src_elems)
    
    src_elem_size = len(src_elems)
    transition_mtx = []
    for i in xrange(src_elem_size):
        mtx_str = data[7+i].replace('\n','').replace('\t',' ')        
        mtx_els = [float(i) for i in (mtx_str.split())[1:]]
        transition_mtx.append(mtx_els)

    print(transition_mtx)

    emission_mtx = []
    dest_elem_size = len(dest_elems)
    for i in xrange(src_elem_size):
        ind = 9 + src_elem_size + i        
        mtx_str = data[ind].replace('\n','').replace('\t',' ')        
        mtx_els = [float(i) for i in (mtx_str.split())[1:]]
        emission_mtx.append(mtx_els)


    print(emission_mtx)
    likehood(transition_mtx,emission_mtx,path,src_elems,dest_elems)    
    
if __name__ == "__main__":   
    task94() 
    
