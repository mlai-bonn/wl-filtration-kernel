import networkx as nx
import numpy as np
from sklearn.cluster import KMeans


# returns set of edge weights appearing in graph
def get_weights_of_graph(g):
    weights = set()
    for e in g.edges(data=True):
        weights.add(e[2]['weight'])
    return weights

        
# return g with subset of edges with weight at least w
def get_filtered_graph(g, w):
    g_filtered = nx.Graph()
    g_filtered.add_nodes_from(g.nodes(data=True))
    selected_edges = []
    for e in g.edges(data=True):
        if e[2]['weight'] >= w: selected_edges.append(e)
    g_filtered.add_edges_from(selected_edges)
    return g_filtered


# weighs edges according to cumulative number of random walks of length at most k
def add_random_walk_weights(g, depth):
    # edge index lookup table
    e2idx ={}
    for i,e in enumerate(g.edges):
        e2idx[(e[0],e[1])] = i
        e2idx[(e[1],e[0])] = i
    # count edge appearances in random walks starting from each node for depth 1
    n2randwalkcounts = {}
    for n in g.nodes:
        n_rw_counts = np.zeros(len(g.edges))
        n2randwalkcounts[n] = n_rw_counts
        for e in g.edges(n):
            eidx = e2idx[e]
            n2randwalkcounts[n][eidx] += 1
        #print(n2randwalkcounts[n])
    # count edge appearances in random walks starting from each node for consecutive depths
    for _ in range(depth):
        n2randwalkcounts_nextlevel = {}
        for n in g.nodes:
            randwalkcounts = np.zeros(len(g.edges))
            n2randwalkcounts_nextlevel[n] = randwalkcounts
            for nbr in g.neighbors(n):
                nbr_rw_sum = np.sum(n2randwalkcounts[nbr])
                eidx = e2idx[(n,nbr)]
                randwalkcounts[eidx] += nbr_rw_sum
                randwalkcounts += n2randwalkcounts[nbr]
            #print(n, randwalkcounts)
        n2randwalkcounts = n2randwalkcounts_nextlevel
    # sum edge appearances over random walks of all nodes
    eweight_array = np.zeros(len(g.edges))
    for n in g.nodes:
        eweight_array += n2randwalkcounts[n]
    #print(g, eweight_array)
    # write weights into edge attributes and return
    for e in g.edges(data=True):
        eidx = e2idx[(e[0],e[1])]
        weight = eweight_array[eidx]
        attrs = e[2]
        attrs['weight'] = weight
    return g


# edges are weighted 
def add_max_degree_weights(g):
    for e in g.edges(data=True):
        max_deg = max(g.degree[e[0]], g.degree[e[1]])
        e[2]['weight'] = max_deg
    return g

# 
def add_core_number_weights(g):
    n2core = nx.core_number(g)
    for e in g.edges(data=True):
        e[2]['weight'] = max(n2core[e[0]], n2core[e[1]])
    return g

# 
def add_triangle_weights(g):
    n2tris = {}
    for n in g.nodes:
        n2tris[n] = 0
    for e in g.edges(data=True):
        u_neighbors = set(g.neighbors(e[0]))
        v_neighbors = set(g.neighbors(e[1]))
        uv_common_neighbors = u_neighbors & v_neighbors
        n2tris[e[0]] += len(uv_common_neighbors)
        n2tris[e[1]] += len(uv_common_neighbors)
    for e in g.edges(data=True):
        e[2]['weight'] = int( max(n2tris[e[0]], n2tris[e[1]]) / 2)
    return g


# extracts a subset of input weights using k-means clustering in order to shrink the number of filtrations to k
def get_weight_subset(weights, k):
    # do nothing if number of weights is already at most k
    if len(set(weights)) <= k: return list(set(weights))
    # perform k-means on weights
    points = np.array([[w,0] for w in weights])
    kmeans = KMeans(n_clusters=k, init='k-means++').fit(points)
    # extract clusters
    clusters = [[] for i in range(k)]
    for i,x in enumerate(points):
        cl_idx = kmeans.labels_[i]
        val = x[0]
        clusters[cl_idx].append(val)
    # from every cluster, extract the minimum element as cutoff point
    weights_subset = []
    for cl in clusters:
        min_weight = min(cl)
        weights_subset.append(min_weight)
    weights_subset.sort(key=None, reverse=True)
    return weights_subset
    
    


