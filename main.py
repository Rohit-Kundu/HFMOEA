from filter_methods import *
import time
import csv
import os
import math
import random
import matplotlib.pyplot as plt
from utils import *
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required = True, help='Path to where the csv file of features')
parser.add_argument('--csv_header', type=bool, default=False, help='Does csv file have headers?')
parser.add_argument('--popsize', type=int, default=100, help='Population Size')
parser.add_argument('--generations', type=int, default=100, help='Number of generations')
parser.add_argument('--mutation', type=int, default=6, help='Mutation percentage')
parser.add_argument('--topk', type=int, default=10, help='topk number of features (Refer to the Paper)')
parser.add_argument('--save_fig', type=bool, default=True, help='Save the figure or not')
args = parser.parse_args()

csv_path = args.path
if ".csv" not in csv_path:
    csv_path+=".csv"

if args.csv_header is None:
    df = np.asarray(pd.read_csv(csv_path, header=None))
else:
    df = np.asarray(pd.read_csv(csv_path))

data = df[:,:-1]
target = df[:,-1]
num_feat = df.shape[1]-1

sol = []
sol.append(MI(data, target))
sol.append(SCC(data,target))
sol.append(Relief(data,target))
sol.append(PCC(data,target))
sol.append(chi_square(data,target))
sol.append(info_gain(data,target))
sol.append(MAD(data,target))
sol.append(Dispersion_ratio(data,target))
sol.append(feature_selection_sim(data, target))
sol.append(Fisher_score(data,target))

topk = args.topk

if args.popsize<10:
    pop_size = 10
    print("Population size cannot be less than 10.")
else:
    pop_size = args.popsize

init_size = len(sol)
max_gen = args.generations
initial_chromosome = np.zeros(shape=(pop_size,data.shape[1]))
for i in range(len(sol)):
    initial_chromosome[i,np.where(sol[i].ranks<=topk)[0]]=1

rand_size = pop_size-init_size
rand_sol = np.random.randint(low=0,high=2,size=(rand_size,data.shape[1]))
initial_chromosome[init_size:,:] = rand_sol

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data,target, test_size=0.2, random_state=0)

#pop_shape = (pop_size,num_features)
num_features = data.shape[1]
num_mutations = (int)(pop_size*num_features*args.mutation/100)
solution = initial_chromosome

gen_no=0
while(gen_no<max_gen):
    function1_values = function1(np.array(solution),X_train,y_train,X_test,y_test).tolist()    
    function2_values = [function2(solution[i])for i in range(0,pop_size)]
    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])
    print("Generation number: ",gen_no+1)
    crowding_distance_values=[]
    for i in range(0,len(non_dominated_sorted_solution)):
        crowding_distance_values.append(crowding_distance(function1_values[:],function2_values[:],non_dominated_sorted_solution[i][:]))
    
    #Generating offsprings
    solution2 = crossover(np.array(solution), offspring_size = (pop_size,num_features))
    solution2 = mutation(solution2, num_mutations = num_mutations)
    solution2 = check_sol(solution2)
    function1_values2 = function1(solution2,X_train,y_train,X_test,y_test).tolist()#[function1(solution2[i])for i in range(0,2*pop_size)]
    function2_values2 = [function2(solution2[i])for i in range(0,pop_size)]
    non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:],function2_values2[:])
    crowding_distance_values2=[]
    for i in range(0,len(non_dominated_sorted_solution2)):
        crowding_distance_values2.append(crowding_distance(function1_values2[:],function2_values2[:],non_dominated_sorted_solution2[i][:]))
    new_solution= []
    for i in range(0,len(non_dominated_sorted_solution2)):
        non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
        front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front.reverse()
        for value in front:
            new_solution.append(value)
            if(len(new_solution)==pop_size):
                break
        if (len(new_solution) == pop_size):
            break
    solution = [solution2[i] for i in new_solution]
    gen_no = gen_no + 1

#Lets plot the final front now
plt.figure()
dst = "Saving/"
if not os.path.exists(dst):
    os.makedirs(dst)

func1 = [i for i in function1_values]
func2 = [j*-1 for j in function2_values]
plt.xlabel('No.of Features Selected', fontsize=15)
plt.ylabel('Classification Accuracy', fontsize=15)
plt.scatter(func2, func1)

csv_name = csv_path.split("/")[-1]

if args.save_fig:
    plt.savefig(dst+csv_name.split('.csv')[0]+"_"+'_all solutions.png', dpi=300)

front_f = fast_non_dominated_sort(function1_values, function2_values)
df = np.concatenate(( np.expand_dims(np.asarray(func1),1), np.expand_dims(np.asarray(func2),1) ), axis = 1)
np.savetxt(dst+csv_name.split('.csv')[0]+'_all solutions.csv',df,newline='\n', delimiter=",")

df = df[df[:,1].argsort()]
feat_unique = np.unique(df[:,1])

pareto = np.array([0,1])
pareto = np.expand_dims(pareto, axis=0)
thresh = 0.0
for f in feat_unique:
    acc_li = []
    for i in range(df.shape[0]):
        if df[i,1] == f:
            acc_li.append(df[i,0])
    max_acc = max(acc_li)
    if max_acc>thresh:
        kk = np.expand_dims(np.asarray([max_acc, f]), axis=0)
        pareto = np.concatenate((pareto, kk), axis=0)
        thresh = max_acc

pareto = np.delete(pareto, 0, 0)
np.savetxt(dst+csv_name.split('.csv')[0]+"_pareto.csv", pareto, delimiter=",", newline="\n")

acc = pareto[:,0].astype(float)
fs = pareto[:,1].astype(int)

plt.figure()
plt.plot(fs,acc,"r*-")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("No. of Features Selected")
plt.ylabel("Classification Accuracy")

if args.save_fig:
    plt.savefig(dst+csv_name.split('.csv')[0]+"_pareto.png",dpi=300)
