import GraphDataToGraphList
import Filtration
import WeisfeilerLehman
import Wasserstein
import Misc
import SVM
import numpy as np
import networkx as nx
import multiprocessing as mp
import argparse


def main(nx_graph_list, graph_labels, edge_weight_func, h_list, k_list, gamma_list, c_list, n_fold, cv_fold, n_jobs):
    
    # add edge weights 
    for g in nx_graph_list:
        g.remove_edges_from(nx.selfloop_edges(g)) # fix data
        #Filtration.add_random_walk_weights(g, 5)
        edge_weight_func(g)
    
    # collect set of weights over all graphs
    weights = []
    for g in nx_graph_list:
        weights += Filtration.get_weights_of_graph(g)
    weights = list(weights)
    
    # check wether k values are valid
    distinct_weights = len(set(weights))
    k_list = [k for k in k_list if k <= distinct_weights]
    
    # 
    params2grams = {}
    for k in k_list:
        print('Starting k =', k, '...')
        
        # reduce number of filtrations by extracting cut-off edge weights
        cutoff_weights = Filtration.get_weight_subset(weights, k)
        print('cut-off weights:', cutoff_weights)
        
        # add Weisfeiler-Lehman labels (returns number of labels)
        print('Assigning node labels ...')
        maxlbl_per_depth = WeisfeilerLehman.add_wl_labels(nx_graph_list, cutoff_weights, max(h_list))
        
        # create label histograms for each graph and label
        print('Generating histograms ...')
        g2wllabelhistograms, g2lblfreqs = WeisfeilerLehman.get_label_histograms(nx_graph_list, k)
    
        # compute EMD distances between histograms of graphs for all labels
        print('Computing Gram matrices ...')
        grams = Wasserstein.get_gram_mats(g2wllabelhistograms, g2lblfreqs, cutoff_weights, h_list, maxlbl_per_depth, gamma_list, n_jobs)
        params2grams.update(grams)

    # start SVM
    print('----------------')
    print('Starting SVM (cross validation)...')
    y = np.array(graph_labels)
    accs = SVM.cross_validation(params2grams, y, h_list, k_list, gamma_list, c_list, n_fold, cv_fold, n_jobs)
    
    # return n_fold accuracies
    return accs



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Weisfeiler-Lehman Filtration Kernel')
    parser.add_argument('db', type=str, help='Dataset name')
    #parser.add_argument('--ewf', default=False, action='store_true', help='Edge weight function')
    parser.add_argument('--h', 
                        type=int, nargs='+', 
                        required=False, default=[5], 
                        help='List of WL depths')
    parser.add_argument('--k', 
                        type=int, nargs='+', 
                        required=False, 
                        default=[1,2,3], 
                        help='List of filtration lengths')
    parser.add_argument('--gamma', 
                        type=int, nargs='+', 
                        required=False, 
                        default=2.0 ** np.array([-12,-8,-5,-3,-1,0,1,3,5,8,12]), 
                        help='List of gammas')
    parser.add_argument('--c', 
                        type=int, nargs='+', 
                        required=False, 
                        default=2.0 ** np.array([-12,-8,-5,-3,-1,0,1,3,5,8,12]), 
                        help='List of Cs (SVM parameter)')
    parser.add_argument('--jobs', 
                        type=int, 
                        required=False, 
                        default=mp.cpu_count(), 
                        help='Number of parallel processes')
    args = parser.parse_args()
    


    print('Dataset:', args.db)
    X, y, attrs = GraphDataToGraphList.graph_data_to_graph_list("Data/", args.db)
    Misc.simplify_node_labels(X)
    
    print('#graphs:', len(X))
    
    edge_weight_func = Filtration.add_max_degree_weights
    #edge_weight_func = Filtration.add_core_number_weights
    #edge_weight_func = Filtration.add_triangle_weights
    
    h_list = args.h
    k_list = args.k
    gamma_list = args.gamma
    c_list = args.c
    cv_fold = 10  # cross-validation
    gs_fold = 3  # grid-search
    n_jobs = args.jobs
    
    print('edge_weight_func:', edge_weight_func.__name__)
    print('h_list:', h_list)
    print('k_list', k_list)
    print('gamma_list', gamma_list)
    print('c_list', c_list)
    #print('cv_fold:', cv_fold)
    #print('gs_fold:', gs_fold)
    print('n_jobs:', n_jobs)
    print('----------------')
    
    accs = main(X, y, edge_weight_func, h_list, k_list, gamma_list, c_list, cv_fold, gs_fold, n_jobs)
    print('----------------')
    print('> Mean accuracy', np.mean(accs))
    print('> Standard deviation', np.std(accs))
    
