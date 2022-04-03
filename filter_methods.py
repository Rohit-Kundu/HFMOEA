import numpy as np
from sklearn.feature_selection import *
from sklearn import datasets
import pandas as pd
from scipy.stats import spearmanr
from ReliefF import ReliefF

#-------------------------  utilities  -------------------------------------------#
def normalize(vector, lb=0, ub=1):
    # function to normalize a numpy vector in [lb, ub]
    norm_vector = np.zeros(vector.shape[0])
    maximum = max(vector)
    minimum = min(vector)
    norm_vector = lb + ((vector - minimum)/(maximum - minimum)) * (ub - lb)

    return norm_vector

class Result():
    # structure of the result
    def __init__(self):
        self.ranks = None
        self.scores = None
        self.features = None
        self.ranked_features = None 

#-------------------------  Chi-Square  -------------------------------------------#
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

#-------------------------  Information Gain  -------------------------------------------#
def info_gain(data,target):
    importances = mutual_info_classif(data,target)
    feature_values = np.array(data)
    result = Result()
    result.features = feature_values
    result.scores = importances
    result.ranks = np.argsort(np.argsort(importances))
    result.ranked_features = feature_values[:, result.ranks]
    return result

#-------------------------  Mean Absolute Dispersion  -------------------------------------------#
def MAD(data,target):
    mean_abs_diff = np.sum(np.abs(data-np.mean(data,axis=0)),axis=0)/data.shape[0]
    feature_values = np.array(data)
    result = Result()
    result.features = feature_values
    result.scores = mean_abs_diff
    result.ranks = np.argsort(np.argsort(mean_abs_diff))
    result.ranked_features = feature_values[:, result.ranks]
    return result

#-------------------------  Dispersion Ratio  -------------------------------------------#
def Dispersion_ratio(data,target):
    data[np.where(data==0)[0]]=1
    am = np.mean(data,axis=0)
    gm = np.power(np.prod(data,axis=0),1/data.shape[0])
    disp_ratio = am/gm
    feature_values = np.array(data)
    result = Result()
    result.features = feature_values
    result.scores = disp_ratio
    result.ranks = np.argsort(np.argsort(disp_ratio))
    result.ranked_features = feature_values[:, result.ranks]
    return result

#-------------------------  Pasi Luukka  -------------------------------------------#
def feature_selection_sim(in_data, target, measure = 'luca', p = 1):
    d = pd.DataFrame(in_data)
    t = pd.DataFrame(target)
    data = pd.concat([d,t],axis=1)
    
    # Feature selection method using similarity measure and fuzzy entroropy 
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

#-------------------------  Fisher Score  -------------------------------------------#
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
            mu[j,i] = np.mean(d[:,j])
            var[j,i] = np.var(d[:,j])
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

#-------------------------  Mutual Information  -------------------------------------------#
def MI(data, target):
    # function that assigns scores to features according to Mutual Information (MI)
    # the rankings should be done in increasing order of the MI scores 
    
    # initialize the variables and result structure
    feature_values = np.array(data)
    num_features = feature_values.shape[1]
    MI_mat = np.zeros((num_features, num_features))
    MI_values_feat = np.zeros(num_features)
    MI_values_class = np.zeros(num_features)
    result = Result()
    result.features = feature_values
    weight_feat = 0.3   # weightage provided to feature-feature correlation
    weight_class = 0.7  # weightage provided to feature-class correlation
    
    # generate the information matrix
    for ind_1 in range(num_features):
        for ind_2 in range(num_features):
            MI_mat[ind_1, ind_2] = MI_mat[ind_2, ind_1] = compute_MI(feature_values[:, ind_1], feature_values[:, ind_2])

    for ind in range(num_features):
        MI_values_feat[ind] = -np.sum(abs(MI_mat[ind,:]))
        MI_values_class[ind] = compute_MI(feature_values[:, ind], target)

    # produce scores and ranks from the information matrix
    MI_values_feat = normalize(MI_values_feat)
    MI_values_class = normalize(MI_values_class)
    MI_scores = (weight_class * MI_values_class) + (weight_feat * MI_values_feat)
    MI_ranks = np.argsort(np.argsort(-MI_scores))

    # assign the results to the appropriate fields
    result.scores = MI_scores
    result.ranks = MI_ranks
    result.ranked_features = feature_values[:, np.argsort(-MI_scores)]

    return result  

def compute_MI(x, y):
    # function to compute mutual information between two variables 
    sum_mi = 0.0
    x_value_list = np.unique(x)
    y_value_list = np.unique(y)
    Px = np.array([ len(x[x==xval])/float(len(x)) for xval in x_value_list ]) #P(x)
    Py = np.array([ len(y[y==yval])/float(len(y)) for yval in y_value_list ]) #P(y)
    for i in range(len(x_value_list)):
        if Px[i] ==0.:
            continue
        sy = y[x == x_value_list[i]]
        if len(sy)== 0:
            continue
        pxy = np.array([len(sy[sy==yval])/float(len(y))  for yval in y_value_list]) #p(x,y)
        t = pxy[Py>0.]/Py[Py>0.] /Px[i] # log(P(x,y)/( P(x)*P(y))
        sum_mi += sum(pxy[t>0]*np.log2( t[t>0]) ) # sum ( P(x,y)* log(P(x,y)/( P(x)*P(y)) )

    return sum_mi

#-------------------------  Relief  -------------------------------------------#
def Relief(data, target):
    # function that assigns scores to features according to Relief algorithm
    # the rankings should be done in increasing order of the Relief scores 

    # initialize the variables and result structure
    feature_values = np.array(data)
    num_features = feature_values.shape[1]
    result = Result()
    result.features = feature_values

    # generate the ReliefF scores
    relief = ReliefF(n_neighbors=5, n_features_to_keep=num_features)
    relief.fit_transform(data, target)
    result.scores = normalize(relief.feature_scores)
    result.ranks = np.argsort(np.argsort(-relief.feature_scores))

    # produce scores and ranks from the information matrix
    Relief_scores = normalize(relief.feature_scores)
    Relief_ranks = np.argsort(np.argsort(-relief.feature_scores))

    # assign the results to the appropriate fields
    result.scores = Relief_scores
    result.ranks = Relief_ranks
    result.ranked_features = feature_values[:, Relief_ranks]

    return result

#-------------------------  Spearman's Correlation Coefficient  -------------------------------------------#
def SCC(data, target):
    # function that assigns scores to features according to Spearman's Correlation Coefficient (SCC)
    # the rankings should be done in increasing order of the SCC scores 

    # initialize the variables and result structure
    feature_values = np.array(data)
    num_features = feature_values.shape[1]
    SCC_mat = np.zeros((num_features, num_features))
    SCC_values_feat = np.zeros(num_features)
    SCC_values_class = np.zeros(num_features)
    result = Result()
    result.features = feature_values
    weight_feat = 0.3   # weightage provided to feature-feature correlation
    weight_class = 0.7  # weightage provided to feature-class correlation

    # generate the correlation matrix
    for ind_1 in range(num_features):
        for ind_2 in range(num_features):
            SCC_mat[ind_1, ind_2] = SCC_mat[ind_2, ind_1] = compute_SCC(feature_values[:, ind_1], feature_values[:, ind_2])

    for ind in range(num_features):
        SCC_values_feat[ind] = -np.sum(abs(SCC_mat[ind,:]))
        SCC_values_class[ind] = compute_SCC(feature_values[:, ind], target)

    # produce scores and ranks from the information matrix
    SCC_values_feat = normalize(SCC_values_feat)
    SCC_values_class = normalize(SCC_values_class)
    SCC_scores = (weight_class * SCC_values_class) + (weight_feat * SCC_values_feat)
    SCC_ranks = np.argsort(np.argsort(-SCC_scores))

    # assign the results to the appropriate fields
    result.scores = SCC_scores
    result.ranks = SCC_ranks
    result.ranked_features = feature_values[:, np.argsort(-SCC_scores)]

    return result

def compute_SCC(x, y):
    # function to compute the SCC value for two variables
    x_order = np.argsort(np.argsort(x))
    y_order = np.argsort(np.argsort(y))
    mean_x = np.mean(x_order)
    mean_y = np.mean(y_order)
    numerator = np.sum((x_order - mean_x) * (y_order - mean_y))
    denominator = np.sqrt(np.sum(np.square(x_order - mean_x)) * np.sum(np.square(y_order - mean_y)))
    SCC_val = numerator/denominator

    return SCC_val

#-------------------------  Pearson's Correlation Coefficient  -------------------------------------------#
def PCC(data, target):
    # function that assigns scores to features according to Pearson's Correlation Coefficient (PCC)
    # the rankings should be done in increasing order of the PCC scores 
    
    # initialize the variables and result structure
    feature_values = np.array(data)
    num_features = feature_values.shape[1]
    PCC_mat = np.zeros((num_features, num_features))
    PCC_values_feat = np.zeros(num_features)
    PCC_values_class = np.zeros(num_features)
    PCC_scores = np.zeros(num_features)
    result = Result()
    result.features = feature_values
    weight_feat = 0.3   # weightage provided to feature-feature correlation
    weight_class = 0.7  # weightage provided to feature-class correlation

    # generate the correlation matrix
    mean_values = np.mean(feature_values, axis=0)
    for ind_1 in range(num_features):
        for ind_2 in range(num_features):
            PCC_mat[ind_1, ind_2] = PCC_mat[ind_2, ind_1] = compute_PCC(feature_values[:, ind_1], feature_values[:, ind_2])

    for ind in range(num_features):
        PCC_values_feat[ind] = -np.sum(abs(PCC_mat[ind,:])) # -ve because we want to remove the corralation
        PCC_values_class[ind] = abs(compute_PCC(feature_values[:, ind], target))

    # produce scores and ranks from the information matrix
    PCC_values_feat = normalize(PCC_values_feat)
    PCC_values_class = normalize(PCC_values_class)
    PCC_scores = (weight_class * PCC_values_class) + (weight_feat * PCC_values_feat)
    PCC_ranks = np.argsort(np.argsort(-PCC_scores)) # ranks basically represents the rank of the original features

    # assign the results to the appropriate fields
    result.scores = PCC_scores
    result.ranks = PCC_ranks
    result.ranked_features = feature_values[:, np.argsort(-PCC_scores)]

    return result


def compute_PCC(x, y):
    # function to compute the PCC value for two variables
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum(np.square(x - mean_x)) * np.sum(np.square(y - mean_y)))
    if denominator == 0:
        return 0
    PCC_val = numerator/denominator

    return PCC_val
