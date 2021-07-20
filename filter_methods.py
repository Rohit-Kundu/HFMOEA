import numpy as np
from Py_FS.filter._utilities import normalize, Result
from sklearn.feature_selection import *
import pandas as pd
from scipy.stats import spearmanr

def chi_square(data, target):
    data = data.clip(min=0)
    chi,_ = chi2(data, target)
    feature_values = np.array(data)
    result = Result()
    result.features = feature_values
    result.scores = chi
    result.ranks = np.argsort(np.argsort(chi))
    result.ranked_features = feature_values[:, result.ranks]
    return result

def info_gain(data,target):
    importances = mutual_info_classif(data,target)
    feature_values = np.array(data)
    result = Result()
    result.features = feature_values
    result.scores = importances
    result.ranks = np.argsort(np.argsort(importances))
    result.ranked_features = feature_values[:, result.ranks]
    return result

def MAD(data,target):
    #mean absolute deviation
    mean_abs_diff = np.sum(np.abs(data-np.mean(data,axis=0)),axis=0)/data.shape[0]
    feature_values = np.array(data)
    result = Result()
    result.features = feature_values
    result.scores = mean_abs_diff
    result.ranks = np.argsort(np.argsort(mean_abs_diff))
    result.ranked_features = feature_values[:, result.ranks]
    return result

def Dispersion_ratio(data,target):
    data[np.where(data==0)[0]] = 1
    var = np.var(data,axis=0)
    mean = np.mean(data,axis=0)
    disp_ratio = var/mean
    feature_values = np.array(data)
    result = Result()
    result.features = feature_values
    result.scores = disp_ratio
    result.ranks = np.argsort(np.argsort(disp_ratio))
    result.ranked_features = feature_values[:, result.ranks]
    return result

def prob_mass_func(target):
    unique = np.unique(target)
    p = np.zeros(shape=(unique.shape[0]))
    for i,u in enumerate(unique):
        p[i] = target.tolist().count(u)/target.shape[0]
    return p

def feature_selection_sim(in_data, target, measure = 'luca', p = 1):
    d = pd.DataFrame(in_data)
    t = pd.DataFrame(target)
    data = pd.concat([d,t],axis=1)
    
    # Feature selection method using similarity measure and fuzzy entropy 
    # measures based on the article:

    # P. Luukka, (2011) Feature Selection Using Fuzzy Entropy Measures with
    # Similarity Classifier, Expert Systems with Applications, 38, pp. 4600-4607

    # Function call:
    # feature_selection_sim(data, measure, p)

    # OUTPUTS:
    # data_mod      data with removed feature
    # index_rem     index of removed feature in original data

    # INPUTS:
    # data          dataframe, contains class values
    # measure       fuzzy entropy measure, either 'luca' or 'park'              
    #               currently coded
    # p             parameter of Lukasiewicz similarity measure
    #               p in (0, \infty) as default p=1.
    
    # You need to import 'numpy' as 'np' before using this function

    l = int(max(data.iloc[:,-1]))   # -classes in the last column
    m = data.shape[0]               # -samples
    t = data.shape[1]-1             # -features
    
    dataold = data.copy()
    
    idealvec_s = np.zeros((l,t)) 
    for k in range(l):
        idx = data.iloc[:,-1] == k+1
        idealvec_s[k,:] = data[idx].iloc[:,:-1].mean(axis = 0)
    
    # scaling data between [0,1]
    data_v = data.iloc[:,:-1]
    data_c = data.iloc[:,-1] # labels
    mins_v = data_v.min(axis = 0)
    Ones   = np.ones((data_v.shape))
    data_v = data_v + np.dot(Ones,np.diag(abs(mins_v)))
    
    tmp =[]
    for k in range(l):
        tmp.append(abs(mins_v))
    
    idealvec_s = idealvec_s+tmp
    maxs_v     = data_v.max(axis = 0)
    data_v     = np.dot(data_v,np.diag(maxs_v**(-1)))
    tmp2 =[];
    for k in range(l):
        tmp2.append(abs(maxs_v))
        
    idealvec_s = idealvec_s/tmp2
    
    data_vv = pd.DataFrame(data_v) # Convert the array of feature to a dataframe
    data    = pd.concat([data_vv, data_c], axis=1, ignore_index=False)

    # sample data
    datalearn_s = data.iloc[:,:-1]
    
    # similarities
    sim = np.zeros((t,m,l))
    
    for j in range(m):
        for i in range(t):
            for k in range(l):
                sim[i,j,k] = (1-abs(idealvec_s[k,i]**p - datalearn_s.iloc[j,i])**p)**(1/p)
            
    sim = sim.reshape(t,m*l)
    
    # possibility for two different entropy measures
    if measure =='luca':
        # moodifying zero and one values of the similarity values to work with 
        # De Luca's entropy measure
        delta = 1e-10
        sim[sim == 1] = delta
        sim[sim == 0] = 1-delta
        H = (-sim*np.log(sim)-(1-sim)*np.log(1-sim)).sum(axis = 1)
    elif measure == 'park':
        H = (np.sin(np.pi/2*sim)+np.sin(np.pi/2*(1-sim))-1).sum(axis = 1) 
        
    feature_values = np.array(in_data)
    result = Result()
    result.features = feature_values
    result.scores = H
    result.ranks = np.argsort(np.argsort(-H))
    result.ranked_features = feature_values[:, result.ranks]
    return result

def Fisher_score(data,target):
    mean = np.mean(data)
    sigma = np.var(data)
    unique = np.unique(target)
    mu = np.zeros(shape=(data.shape[1],unique.shape[0]))
    n = np.zeros(shape=(unique.shape[0],))
    var = np.zeros(shape=(data.shape[1],unique.shape[0]))
    for j in range(data.shape[1]):
        for i,u in enumerate(unique):
            d = data[np.where(target==u)[0]]
            n[i] = d.shape[0]
            mu[j,i] = np.mean(d[j])
            var[j,i] = np.var(d[j])
    fisher = np.zeros(data.shape[1])
    for j in range(data.shape[1]):
        sum1=0
        sum2=0
        for i,u in enumerate(unique):
            sum1+=n[i]*((mu[j,i]-mean)**2)
            sum2+=n[i]*var[j,i]
        fisher[j] = sum1/sum2
    feature_values = np.array(data)
    result = Result()
    result.features = feature_values
    result.scores = fisher
    result.ranks = np.argsort(np.argsort(fisher))
    result.ranked_features = feature_values[:, result.ranks]
    return result

def Spearman_corr(data,target):
    scores = np.mean(spearmanr(data,axis=0,nan_policy='omit').correlation,axis=1)
    feature_values = np.array(data)
    result = Result()
    result.features = feature_values
    result.scores = scores
    result.ranks = np.argsort(np.argsort(-scores))
    result.ranked_features = feature_values[:, result.ranks]
    return result

def Spearman_pvalue(data,target):
    scores = np.mean(spearmanr(data,axis=0,nan_policy='omit').pvalue,axis=1)
    feature_values = np.array(data)
    result = Result()
    result.features = feature_values
    result.scores = scores
    result.ranks = np.argsort(np.argsort(-scores))
    result.ranked_features = feature_values[:, result.ranks]
    return result
