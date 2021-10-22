import numpy as np
import multiprocessing as mp
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

# global memory
params2grams = None

# 
def cross_validation(grams, y, h_list, k_list, gamma_list, c_list, cv_fold, gs_fold, n_jobs):

    # shared memory
    global params2grams
    params2grams = grams
    
    # 
    accs = []
    skf = StratifiedKFold(n_splits=cv_fold, shuffle=True)
    X = np.arange(len(y)) # X is set of indices
    for train_indices, test_indices in skf.split(X, y):
        
        X_train = np.array(X)[train_indices]
        X_test = np.array(X)[test_indices]
        y_train = np.array(y)[train_indices]
        y_test = np.array(y)[test_indices]
        
        # perform grid search for best parameters
        best_params = grid_search_cv(X_train, y_train, h_list, k_list, gamma_list, c_list, gs_fold, n_jobs) 
        best_h, best_k, best_gamma, best_c = best_params
        
        # setup SVM and perform classification using best parameters
        gram = get_gram_mat(X, best_h, best_k, best_gamma)  
        acc_test = perform_svc(gram, X_train, X_test, y_train, y_test, best_c)
        print('best_params (h,k,gamma,C):', best_params, 'acc_test', acc_test)
        accs.append(acc_test)
    
    return accs
       
                    
# 
def grid_search_cv(X, y, h_list, k_list, gamma_list, c_list, gs_fold, n_jobs):
    
    # create list of function calls over all parameter combinations
    job_args = []
    for h in h_list:
        for k in k_list:
            for gamma in gamma_list:
                job_args.append([X, y, h, k, gamma, c_list, gs_fold])
    
    # run jobs in parallel
    with mp.Pool(processes=n_jobs) as pool:
        grid_search_results = pool.starmap(kernel, job_args)
    #print('ret', grid_search_results)
    
    # extract best parameters
    best_params = (None,-1.0)
    for l in grid_search_results:
        for lc in l:
            if lc[1] > best_params[1]:
                best_params = lc
                
    # return best parameter triple (h,k,gamma,c)
    return best_params[0]



def kernel(X, y, h, k, gamma, c_list, gs_fold):
    
    # compute complete gram matrix over X
    gram = get_gram_mat(X, h, k, gamma)
    
    # set indices of submatrix X
    X = np.arange(len(X))
    
    # perform cross validation for each value c
    parameters_meanacc_tuples = []
    for c in c_list:
        accs = []
        skf = StratifiedKFold(n_splits=gs_fold, shuffle=True)
        for train_indices, test_indices in skf.split(X, y):
            
            X_train = np.array(X)[train_indices]
            X_test = np.array(X)[test_indices]
            y_train = np.array(y)[train_indices]
            y_test = np.array(y)[test_indices]
            
            acc_test = perform_svc(gram, X_train, X_test, y_train, y_test, c)
            accs.append(acc_test)
        
        # return cross validation mean accuracy
        mean_acc = np.mean(accs)
        parameters = (h, k, gamma, c)
        parameters_meanacc_tuples.append((parameters, mean_acc))
    
    # 
    return parameters_meanacc_tuples


def get_gram_mat(X, h, k, gamma):
    
    # 
    flat_gram = params2grams[(h,k,gamma)]
    
    # select entries X in flattened matrix
    size = int(np.sqrt(len(flat_gram)*2))
    idcs = [i*size+j-int(i*(i+1)/2) for i in X for j in X if i<=j]
    flat_subgram = flat_gram[idcs]
    
    # unflatten
    n = len(X)
    R,C = np.triu_indices(n)
    mat = np.zeros((n,n),dtype=flat_subgram.dtype)
    mat[R,C] = flat_subgram
    mat[C,R] = flat_subgram
    
    return mat
    

        
def perform_svc(gram, X_train, X_test, y_train, y_test, c):        
    
    # setup SVM and perform classification   
    clf = SVC(C=c, kernel='precomputed', max_iter=1000000)
    gram_train = gram[np.ix_(X_train, X_train)]
    clf.fit(gram_train, y_train)
    
    # predict
    gram_test = gram[np.ix_(X_test, X_train)]
    y_pred = clf.predict(gram_test)
    acc_test = accuracy_score(y_test, y_pred)
    
    return acc_test




        