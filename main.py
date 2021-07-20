import numpy as np
import pandas as pd
from Py_FS.filter import MI,SCC,Relief,PCC
from filter_methods import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default = './', help='Directory where feature files are stored.')
parser.add_argument('--csv_name', type=str, required = True, help='Name of csv file- Example: SpectEW.csv')
parser.add_argument('--csv_header', type=str, default = 'no', help='Does csv file have header?: yes/no')
parser.add_argument('--generations', type=int, default = 20, help='Number of Generations to run the algorithm')
parser.add_argument('--mutation', type=int, default = 6, help='Percentage of mutation in the NSGA-II')
parser.add_argument('--topk', type=int, default = 11, help="Top 'k' ranked features to be taken")
parser.add_argument('--save_fig', type=bool, default = False, help="Save result plot?")
args = parser.parse_args()

root = args.root
if root[-1]!='/':
    root+='/'
csv_path = args.csv_name
if args.csv_header.lower()=="yes":
    df = np.asarray(pd.read_csv(root+csv_path))
else:
    df = np.asarray(pd.read_csv(root+csv_path,header=None))
data = df[:,0:-1]
target = df[:,-1]

#Importing required modules
import math
import random
import matplotlib.pyplot as plt
from utils import *
import pandas as pd
import numpy as np

#Function to find index of list
def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

#Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = math.inf
    return sorted_list

#Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front


#Function to calculate crowding distance
def crowding_distance(values1,values2, front):
    epsilon = 0.00001
    distance = [0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1,len(front)-1):
        kk = (max(values1)-min(values1))
        if kk==0:
          kk=epsilon
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/kk
    for k in range(1,len(front)-1):
        kk = max(values2)-min(values2)
        if kk==0:
          kk=epsilon
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/kk
    return distance

#Function to carry out the mutation operator

def mutation(solution):
    mutation_prob = random.random()
    if mutation_prob <1:
        solution = min_x+(max_x-min_x)*random.random()
    return solution

#First function to optimize
def function1(x,X_train,y_train,X_test,y_test):
    value,_,_ = cal_pop_fitness(x,X_train,y_train,X_test,y_test)
    return value

#Second function to optimize
def function2(x):
    value = np.where(x==1)[0].shape[0]
    return -value

def check_sol(sol):
  for i,s in enumerate(sol):
    if False not in (s==np.zeros(shape=(s.shape))):
      sol[i,:] = np.random.randint(low=0,high=2,size=sol[i].shape)
  return sol

sol = []
sol.append(MI(data, target))
sol.append(SCC(data,target))
sol.append(Relief(data,target))
sol.append(PCC(data,target))
sol.append(chi_square(data,target))
sol.append(info_gain(data,target))
sol.append(MAD(data,target))
sol.append(Dispersion_ratio(data,target))
sol.append(feature_selection_sim(data, target, measure = 'park'))
sol.append(Fisher_score(data,target))

topk = args.topk

pop_size = 100

init_size = len(sol)
max_gen = 20
initial_chromosome = np.zeros(shape=(pop_size,data.shape[1]))
for i in range(len(sol)):
  initial_chromosome[i,np.where(sol[i].ranks<=topk)[0]]=1

rand_size = pop_size-init_size
rand_sol = np.random.randint(low=0,high=2,size=(rand_size,data.shape[1]))
initial_chromosome[init_size:,:] = rand_sol

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data,target, test_size=0.2, random_state=0)

#pop_shape = (pop_size,num_features)
num_mutations = (int)(pop_size*num_features*args.mutation/100)
solution = initial_chromosome

gen_no=0
while(gen_no<max_gen):
    function1_values = function1(np.array(solution),X_train,y_train,X_test,y_test).tolist()#[function1(solution[i],X_train,y_train,X_test,y_test)for i in range(0,pop_size)]
    function2_values = [function2(solution[i])for i in range(0,pop_size)]
    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])
    print("The best front for Generation number ",gen_no, " is")
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

dst = "nsga_res/"
import os
if not os.path.exists(dst):
  os.makedirs(dst)

function1 = [i for i in function1_values]
function2 = [j*-1 for j in function2_values]
plt.xlabel('Number of Features Selected', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.scatter(function2, function1)

if args.save_fig==True:
    plt.savefig(dst+csv_path.split('.')[0]+'.png', dpi=300)
plt.show()
