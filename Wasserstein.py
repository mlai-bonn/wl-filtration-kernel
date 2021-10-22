import tqdm
import numpy as np
import multiprocessing as mp

g2hists = None
g2freqs = None

# returns "EMD similarities" (i.e. exp(-emd)) between histograms 
# as well as corresponding mass products for all labels
def get_gram_mats(g2wllabelhistograms, g2lblfreqs, weights, h_list, maxlbl_per_depth, gammas, n_jobs):
    
    # set input g2wllabelhistograms as shared memory
    global g2hists
    g2hists = g2wllabelhistograms
    global g2freqs
    g2freqs = g2lblfreqs
    
    # generate list of jobs
    job_args = []
    nmb_graphs = len(g2wllabelhistograms)
    for i in range(nmb_graphs): 
        job_args.append((i, weights, h_list, maxlbl_per_depth, gammas))
                
    # run jobs in parallel
    with mp.Pool(processes=n_jobs) as pool:
        partialgram_collection = list(tqdm.tqdm(pool.imap(emd1d_similarities, job_args), total=len(job_args)))
    
    # prepare results
    k = len(weights)
    params2gram = {}
    for h in h_list:
        for gamma in gammas:
            partialgrams_at_params = []
            for pg in partialgram_collection:
                partialgrams_at_params.append(pg[(h,gamma)])
                del pg[(h,gamma)]
            flat_gram_mat = np.concatenate(partialgrams_at_params)
            params2gram[(h, k, gamma)] = flat_gram_mat
        
        #####
        #np.set_printoptions(precision=2)
        #print(gamma, '\t', flat_gram_mat)
        #####
        
    return params2gram


# return EMD similarities between histograms of g1 and g2 over all labels
# together with the product of their histograms masses
def emd1d_similarities(args):
    g1_idx, weights, h_list, maxlbl_per_depth, gammas  = args
    
    # 
    nmb_graphs = len(g2hists)
    partial_gram_mats = {}
    for h in h_list:
        for gamma in gammas:
            partial_gram_mats[(h,gamma)] = np.zeros(nmb_graphs-g1_idx, dtype=np.float32)
    
    # normalize weights
    diff_weights = np.array(weights) - weights[-1]
    if len(diff_weights) == 1: diff_weights[0] = 1.0 # fix div by zero case
    diff_weights = diff_weights / diff_weights[0]
    diff_weights = np.diff(diff_weights) * -1.0
    diff_weights = np.append(diff_weights, 0.0)
    
    # 
    for g2_idx in range(g1_idx, nmb_graphs):
    
        # iterate over all labels appearing in both graphs
        g1_labels = g2hists[g1_idx].keys()
        g2_labels = g2hists[g2_idx].keys()
        mutual_labels = g1_labels & g2_labels
        mutual_labels = list(mutual_labels)
        mutual_labels.sort(key=None, reverse=False)
        
        # 
        emd_dists = np.zeros(len(mutual_labels), dtype=np.float32)
        mass_prods = np.zeros(len(mutual_labels), dtype=np.float32)
               
        # 
        for i,lbl in enumerate(mutual_labels):
            
            # extract histograms for current label
            g1_hist = g2hists[g1_idx][lbl]
            g2_hist = g2hists[g2_idx][lbl]
            
            # compute EMD distance
            cum_l1 = np.abs(g1_hist - g2_hist)
            emd_dist = np.dot(cum_l1, diff_weights)
            emd_dists[i] = emd_dist
            
            # get histogram masses
            g1_hist_mass = g2freqs[g1_idx][lbl]
            g2_hist_mass = g2freqs[g2_idx][lbl]
            mass_product = g1_hist_mass * g2_hist_mass
            mass_prods[i] = mass_product
                        
        # 
        for h in h_list:
            max_h_lbl = maxlbl_per_depth[h]
            # select indices corresponding to labels which belong to depth at most h
            cutoff_idx = 0
            while cutoff_idx < len(mutual_labels) and mutual_labels[cutoff_idx] <= max_h_lbl: cutoff_idx += 1
            for gamma in gammas:
                emd_sims = np.exp(emd_dists)
                emd_sims **= (-1.0 * gamma)
                partial_gram_mats[(h,gamma)][g2_idx-g1_idx] = np.dot(emd_sims[0:cutoff_idx], mass_prods[0:cutoff_idx])
                
    return partial_gram_mats


