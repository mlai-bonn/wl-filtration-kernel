import numpy as np

# adds the next WL node labels for all depth values and filtrations
def add_wl_labels(g_list, weights, h):
    weights.sort(key=None, reverse=True)
    
    maxlbl_per_depth = []
    
    conc2lbl = {}
    lbl_idx = 0 
    
    for g in g_list:
        for n in g.nodes(data=True):
            n[1]['wl'] = []
            n_lbl = n[1]['label']
            if n_lbl >= lbl_idx: 
                lbl_idx = n_lbl+1
            # setup h*k matrix for labels of each node
            # i.e., [[lbl_d=0_w=0,...,lbl_d=0_w=k],..., [lbl_d=h_w=0,...,lbl_d=h_w=k]] 
            # where lbl_d_w is the label at depth d and filtration w
            n[1]['wl'].append([n_lbl]*len(weights))
            # empty lists for next values h
            for i in range(h):
                n[1]['wl'].append([])
    maxlbl_per_depth.append(lbl_idx-1)
    
    # generate WL node labels for all depth values over all filtrations
    for i in range(h):
        for w_idx, w in enumerate(weights):
            for g in g_list:
                for n in g.nodes(data=True):
                    # get WL node labels of node n and its neighbors at depth i and filtration w_idx 
                    # in order to generate the label at depth i+1 and filtration w_idx
                    n_lbl = n[1]['wl'][i][w_idx]
                    nbrs_lbls = []
                    for nbr in g.neighbors(n[0]):
                        nbr_lbl = g.nodes[nbr]['wl'][i][w_idx]
                        n_nbr_edge_weight = g.edges[(n[0],nbr)]['weight']
                        if n_nbr_edge_weight >= w: nbrs_lbls.append(nbr_lbl)
                    nbrs_lbls.sort(key=None, reverse=False)
                    concLbl = (n_lbl, tuple(nbrs_lbls))
                    # add new label to dictionary if not already contained
                    new_lbl = conc2lbl.get(concLbl)
                    if not new_lbl: 
                        conc2lbl[concLbl] = lbl_idx
                        new_lbl = lbl_idx
                        lbl_idx += 1
                    # safe new WL node label
                    n[1]['wl'][i+1].append(new_lbl)
        # 
        maxlbl_per_depth.append(lbl_idx-1)
    
    print('   # of labels:', lbl_idx)
    return maxlbl_per_depth
    #return lbl_idx # return the number of labels
    
    '''
    print('\nNode labels:')
    for i,g in enumerate(g_list):
        print('Graph', i)
        for n in g.nodes(data=True):
            print(str(n[0])+": "+str(n[1]['wl']))
    print('\nWL dictionary:')
    for ll in conc2lbl:
        print(str(conc2lbl[ll])+": "+str(ll))
    '''
            

# Returns a list of size h containing dictionaries which map from a label to a histogram.
# The histograms are cumulative over all nodes in graph g. 
# A histogram records the number of nodes which have the considered label at each filtration.
def get_label_histograms(nx_graph_list, k):
    
    # 
    g2wllabelhistograms = [] 
    g2lblfreqs = []
    for i,g in enumerate(nx_graph_list):
    
        # setup dictionary mapping labels to histograms
        lbl2histogram = {}
        for n in g.nodes(data=True):
            wl = n[1]['wl']
            # lbl_seq the filtration sequence of labels
            for lbl_seq in wl:
                for seq_idx, lbl in enumerate(lbl_seq):
                    lbl2histogram.setdefault(lbl, np.zeros(k))
                    lbl2histogram[lbl][seq_idx] += 1
        g2wllabelhistograms.append(lbl2histogram)
                    
        #
        lbl2freq = {}
        #print('---------------- graph', i, '----------------')
        for lbl in lbl2histogram:
            #print('\n', lbl)
            hist = lbl2histogram[lbl]
            #print('hist', hist)
            # store label frequencies
            hist_mass = np.sum(hist)
            lbl2freq[lbl] = hist_mass
            # normalize mass
            hist /= hist_mass
            #print('hist_normed', hist)
            # cumulative 
            hist = np.cumsum(hist)
            #print('hist_cumsum', hist)
            # 
            lbl2histogram[lbl] = hist
            #print('freq', hist_mass)
        g2lblfreqs.append(lbl2freq)
    
    return g2wllabelhistograms, g2lblfreqs





'''
SAMPLE OUTPUT:
Node labels:
Graph 0
0: [[0, 0, 0], [3, 3, 4], [16, 16, 37]]
1: [[0, 0, 0], [3, 3, 4], [16, 16, 37]]
2: [[0, 0, 0], [3, 5, 4], [16, 26, 38]]
3: [[0, 0, 0], [4, 11, 11], [17, 27, 39]]
4: [[0, 0, 0], [5, 11, 11], [18, 28, 40]]
5: [[0, 0, 0], [3, 5, 4], [16, 26, 38]]
6: [[0, 0, 0], [3, 5, 4], [16, 26, 38]]
7: [[0, 0, 0], [3, 5, 4], [16, 26, 38]]
8: [[0, 0, 0], [5, 11, 11], [18, 29, 40]]
9: [[0, 0, 0], [4, 11, 11], [17, 27, 39]]
10: [[0, 0, 0], [3, 5, 4], [16, 26, 38]]
11: [[0, 0, 0], [3, 5, 4], [16, 30, 41]]
12: [[0, 0, 0], [6, 12, 12], [19, 31, 42]]
13: [[0, 0, 0], [3, 4, 4], [16, 32, 32]]
14: [[1, 1, 1], [7, 7, 13], [20, 33, 43]]
15: [[2, 2, 2], [8, 8, 14], [21, 21, 44]]
16: [[2, 2, 2], [8, 8, 14], [21, 21, 44]]
Graph 1
0: [[0, 0, 0], [3, 3, 4], [16, 16, 37]]
1: [[0, 0, 0], [3, 5, 4], [16, 26, 38]]
2: [[0, 0, 0], [5, 11, 11], [18, 34, 45]]
3: [[0, 0, 0], [3, 5, 4], [16, 26, 38]]
4: [[0, 0, 0], [3, 3, 4], [16, 16, 46]]
5: [[0, 0, 0], [3, 3, 10], [16, 16, 47]]
6: [[1, 1, 1], [9, 7, 15], [22, 33, 48]]
7: [[0, 0, 0], [4, 12, 12], [23, 35, 49]]
8: [[0, 0, 0], [10, 12, 12], [24, 36, 50]]
9: [[0, 0, 0], [3, 5, 4], [16, 30, 41]]
10: [[1, 1, 1], [7, 7, 13], [25, 33, 43]]
11: [[2, 2, 2], [8, 8, 14], [21, 21, 44]]
12: [[2, 2, 2], [8, 8, 14], [21, 21, 44]]

WL dictionary:
3: (0, ())
4: (0, (0, 0))
5: (0, (0,))
6: (0, (1,))
7: (1, (0,))
8: (2, ())
9: (1, ())
10: (0, (0, 1))
11: (0, (0, 0, 0))
12: (0, (0, 0, 1))
13: (1, (0, 2, 2))
14: (2, (1,))
15: (1, (0, 0))
16: (3, ())
17: (4, (4, 5))
18: (5, (4,))
19: (6, (7,))
20: (7, (6,))
21: (8, ())
22: (9, ())
23: (4, (5, 10))
24: (10, (4, 7))
25: (7, (10,))
26: (5, (11,))
27: (11, (5, 11, 11))
28: (11, (5, 5, 11))
29: (11, (4, 5, 11))
30: (5, (12,))
31: (12, (4, 5, 7))
32: (4, (11, 12))
33: (7, (12,))
34: (11, (5, 5, 12))
35: (12, (7, 11, 12))
36: (12, (5, 7, 12))
37: (4, (4, 4))
38: (4, (4, 11))
39: (11, (4, 11, 11))
40: (11, (4, 4, 11))
41: (4, (4, 12))
42: (12, (4, 4, 13))
43: (13, (12, 14, 14))
44: (14, (13,))
45: (11, (4, 4, 12))
46: (4, (4, 10))
47: (10, (4, 15))
48: (15, (10, 12))
49: (12, (11, 12, 15))
50: (12, (4, 12, 13))

Graph 1 histograms:
0 {0: {0: 14, 1: 14, 2: 14}, 1: {0: 1, 1: 1, 2: 1}, 2: {0: 2, 1: 2, 2: 2}}
1 {3: {0: 9, 1: 2}, 4: {2: 9, 0: 2, 1: 1}, 5: {1: 6, 0: 2}, 11: {1: 4, 2: 4}, 6: {0: 1}, 12: {1: 1, 2: 1}, 7: {0: 1, 1: 1}, 13: {2: 1}, 8: {0: 2, 1: 2}, 14: {2: 2}}
2 {16: {0: 9, 1: 2}, 37: {2: 2}, 26: {1: 5}, 38: {2: 5}, 17: {0: 2}, 27: {1: 2}, 39: {2: 2}, 18: {0: 2}, 28: {1: 1}, 40: {2: 2}, 29: {1: 1}, 30: {1: 1}, 41: {2: 1}, 19: {0: 1}, 31: {1: 1}, 42: {2: 1}, 32: {1: 1, 2: 1}, 20: {0: 1}, 33: {1: 1}, 43: {2: 1}, 21: {0: 2, 1: 2}, 44: {2: 2}}

Graph 2 histograms:
0 {0: {0: 9, 1: 9, 2: 9}, 1: {0: 2, 1: 2, 2: 2}, 2: {0: 2, 1: 2, 2: 2}}
1 {3: {0: 6, 1: 3}, 4: {2: 5, 0: 1}, 5: {1: 3, 0: 1}, 11: {1: 1, 2: 1}, 10: {2: 1, 0: 1}, 9: {0: 1}, 7: {1: 2, 0: 1}, 15: {2: 1}, 12: {1: 2, 2: 2}, 13: {2: 1}, 8: {0: 2, 1: 2}, 14: {2: 2}}
2 {16: {0: 6, 1: 3}, 37: {2: 1}, 26: {1: 2}, 38: {2: 2}, 18: {0: 1}, 34: {1: 1}, 45: {2: 1}, 46: {2: 1}, 47: {2: 1}, 22: {0: 1}, 33: {1: 2}, 48: {2: 1}, 23: {0: 1}, 35: {1: 1}, 49: {2: 1}, 24: {0: 1}, 36: {1: 1}, 50: {2: 1}, 30: {1: 1}, 41: {2: 1}, 25: {0: 1}, 43: {2: 1}, 21: {0: 2, 1: 2}, 44: {2: 2}}


'''