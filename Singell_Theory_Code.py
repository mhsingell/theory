#!/usr/bin/env python
# coding: utf-8

# # Occam’s Razor is a Theory: Why Cause-and-Effect Performance Links Form Parsimonious Mental Models of Complex Strategic Environments 
# 
# **Author:** Madison Singell
# 
# **Acknowledgement:** Baseline NK-Model Code from the TOM NK-landscape competition
# 
# **Updated as of:** 7/18/2025
# 
# ## Goal: To measure how theories (mental models with cause-and-effect performance links) and mental models with associative performance links accurately and simply represent strategic environments, and the performance consequences of this differential representation. 
# 
# 
# ## Table of Contents
# 
# ### Section 0: Load Libraries, Generate Functions, and Set Parameters
#  * Section 0.1: Load Libraries
#  * Section 0.2: Specify NK-Landscape Functions
#  * Section 0.3: Set Parameters (for TOM presentation, T=100, NS=1000)
# 
# ### Section 1: NK-Landscape, Varying N
# Results of mental models with cause-and-effect vs. associative performance links searching across varying N, where K = 3. This includes the results for Figure 5A, 6A, and 6B. 
# * **FIGURE 4**: Average Proportion of Shared Strategies between Causal and Associative Mental Models
#     * Figure 4 below shows the average proportion of shared strategies selected by a decision-maker using the same M strategic choices in an associative mental model vs. theory across 100 times periods and 1000 simulation runs (for strategic environment, N = 10, K = 3). 
# * **FIGURE 5**: Number of Links by Representation Size and Link Type
#     * Figure 5 visualizes equation 1 and equation 2 across values of M, showing that the number of performance links represented by cause-and-effect mental models ( I_(cause-and-effect)) will always be half that of the number of performance links of the equivalent associative mental models (I_associative).
# * **FIGURE 6A**: Accuracy of Links by Representation Size (M) and Number of Strategic Choices (N) for Mental Models of Varying Link Types
#     * Figure 6a shows the performance of mental models of varying size (M) relative to the number of links represented in the mental model (I) as the number of strategic choices (N) in the environment increases (with K = 3).  
# * **FIGURE 7**: Performance by Link Type and Size (M)
#     * Figure 7 shows the performance of strategies found by decision-makers using associative vs. causal mental model of size (M). (with N = 10 and K = 3).
# * **FIGURE 8**: Performance by Number of Links (I) and Link Type
#     * Figure 8 shows the relationship between the performance of strategies found and the number of links in a mental model (I) by type of representation (associative vs. causal) for a strategic environment with N = 10 and K = 3.
# * **FIGURE 9**: Performance by Number of Links (I) and Size (M)
#     * Figure 9 shows why this is the case, by graphing the performance of strategies selected and the number of performance links in a representation (I) by the number of strategic choices in the mental model (M).
# * **FIGURE 10**: Performance by Accuracy of Links (Acc) and Link Type
#      * Figure 10 shows the relationship between performance and accuracy of representation (Acc) by type of mental model (associative vs. theory) for a strategic environment of N = 10 and K = 3.
# * **FIGURE 11**: Performance by Accuracy of Links (Acc) and Representation Size (M)
#     * Figure 11 shows why accuracy in representation is actually associated with higher performance, by graphing the performance of strategies selected by accuracy of representation (Acc) and by size of the mental model (M). 
# * **APPENDIX B1**: Across N
#     * Figure 7 shows the performance of strategies found by decision-makers using associative vs. causal mental model of size (M). (with N = 10 and K = 3). These results are stable across strategic environments of varying N and K, as shown in Appendix B. 
# * **APPENDIX C1**:, Across N
#     * Because the number of links in a representation isn’t dependent on features of the environment (N or K), these results hold across all strategic environments as shown in Appendix C.
#     
# 
# ### Section 2: NK-Landscape, Varying K
# Results of mental models with cause-and-effect vs. associative performance links searching across varying K, where N = 10.
# * **FIGURE 12**: Performance of Mental Models by Accuracy (Acc) and Link Type Across Environments of Increasing Number of Performance Links Between Strategic Choices (K)
#     * Figure 12, however, shows the relationship between performance and accuracy of representation (Acc) by type of mental model (associative vs. theory) for a strategic environment of N = 10 where K increases from 3 to 7.
# * **FIGURE 13**: Performance of Mental Models by Accuracy (Acc) and Size (M) Across Environments of Increasing Number of Performance Links Between Strategic Choices (K) 
#     * Figure 13, graphs performance and accuracy by size of representation (M) across strategic environments with N = 10 and increasing value of K (from 3 to 7). 
# * **APPENDIX B2**: Across K
#     * Figure 7 shows the performance of strategies found by decision-makers using associative vs. causal mental model of size (M). (with N = 10 and K = 3). These results are stable across strategic environments of varying N and K, as shown in Appendix B. 
# * **APPENDIX C2**:, Across K
#     * Because the number of links in a representation isn’t dependent on features of the environment (N or K), these results hold across all strategic environments as shown in Appendix C.

# ## Section 0: Load Libraries, Generate Functions, and Set Parameters
# 
# In this section I load the necessary libraries for the code to run, specify functions that will generate the NK landscape to search over, and set the parameters for the simulation run.

# ### Section 0.1: Load Libraries

# In[56]:


#For Life
import numpy as np
import random as random
from random import shuffle
import itertools
import pandas as pd
import math

#For Regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from scipy import stats

#For Plotting
import seaborn as sns
import matplotlib.pyplot as plt


# ### Section 0.2: Specify NK-Landscape Functions
# 
# These are the functions to generate a standard NK-landscape.

# In[57]:


"""
===================================================================
Here we define the performance function and the dependencies
"""

class PerfFuncCallError(Exception):
    pass


def perffunc_evenk(ac, k, n, dep, fitn, t):
    ll = np.arange(k + 1)
    binv = 2 ** ll
    # binv = 2^0,2^1,...,2^k
    # this is used to index the different choice vectors
    # each choice vector has a number associated with it
    # using the binary representation

    # Next
    # create a matrix "mat" with n rows and k+1 columns
    # The matrix specifies which dimensions the fitness contribution
    # of a given dimension depends on

    mat = np.zeros((n, k + 1), dtype=int)
    mat[:, 0] = ac[:n]   # the fitness contribution
    mat[:, 1:] = ac[dep[:n, :]]
    # Compute binary numbers for each row using matrix multiplication
    nos = np.dot(mat, binv)
    # this gives each choice vector in the rows a unique number
    # for example 0101 = 5 since
    # (starting from the right): 1*2^0 + 0*2^1 + 1*2^2 + 0*2^3 = 1+0+4+0 = 5
    # then I know that the fitness contribution of 0101 for row i
    # (corresponding to dimension i) is fitn(i,5)

    # Compute performance: just find relevant value in fitn matrix
    perf = np.mean(fitn[np.arange(n), nos])

    return perf


def depend_evenk(n,k):
    '''
    Here select interdependencies: which k others does a given dimension depend on?
    1) np.arange(n) = list of integers 0,1,2,..,n-1
    2) np.delete(np.arange(n), i): we delete the number i (because dimension i should not be selected)
    3) np.random.choice: now we select k values from the resulting list
    '''

    dep = np.zeros((n, k), dtype=int)
    for i in range(n):
        dep[i] = np.random.choice(np.delete(np.arange(n), i), k, replace=False)

    return dep


# ### Section 0.3: Set Parameters
# 
# Set the number of time periods and the number of simulations to run for the NK-landscape functions below. 

# In[58]:


#SET PARAMETERS
number_of_periods  = 100 #Number of time periods to search
number_of_simulations = 1000 # Number of simulations/iterations (you can vary this at your will)
#Set at 100, 1000 for TOM presentation


# ## Section 1: NK-Landscape, Varying N
# 
# Results of mental models with cause-and-effect vs. associative performance links searching across varying N, where K = 3.

# In[59]:


#Set Seed
my_rand = 42
np.random.seed(my_rand)

#SET K to 3
k = 3 # K in NK
#List of Possible Ns
#n_list = list(range(10,15))
n_list = [10]
#List of Possible Simultaneous Representation Sizes
m_list = list(range(3,10))


#Generate Empty Items to Store Results
#Pulling from object max_perf_forM
#Of the form: {3(M): [list of average of max performance by N]}
max_perf_forM = dict.fromkeys(m_list,[])
max_perf_forO = dict.fromkeys(m_list,[])

avg_acc_sim = np.zeros((len(m_list),len(n_list)), dtype=float)
avg_pre_sim = np.zeros((len(m_list),len(n_list)), dtype=float)
avg_rec_sim = np.zeros((len(m_list),len(n_list)), dtype=float)
avg_perf_sim = np.zeros((len(m_list),len(n_list)), dtype=float)

avg_acc_seq = np.zeros((len(m_list),len(n_list)), dtype=float)
avg_pre_seq = np.zeros((len(m_list),len(n_list)), dtype=float)
avg_rec_seq = np.zeros((len(m_list),len(n_list)), dtype=float)
avg_perf_seq = np.zeros((len(m_list),len(n_list)), dtype=float)

shared_bym = np.zeros((len(m_list),number_of_simulations), dtype=float) 

#For each different N
for n_it in range(len(n_list)):
    n = n_list[n_it]
    #Accuracy of interdependencies simultaneous choice
    acc_sim_list = np.zeros((len(m_list),number_of_simulations), dtype=float)
    pre_sim_list = np.zeros((len(m_list),number_of_simulations), dtype=float)
    rec_sim_list = np.zeros((len(m_list),number_of_simulations), dtype=float)
    perf_sim_list = np.zeros((len(m_list),number_of_simulations), dtype=float)
    
    #Accuracy of interdependencies simultaneous choice
    acc_seq_list = np.zeros((len(m_list),number_of_simulations), dtype=float)
    pre_seq_list = np.zeros((len(m_list),number_of_simulations), dtype=float)
    rec_seq_list = np.zeros((len(m_list),number_of_simulations), dtype=float)
    perf_seq_list = np.zeros((len(m_list),number_of_simulations), dtype=float)
    
    # repeat the simulation for many runs
    # array to store final scores (max performance of a single simulation)
    scores = np.zeros(number_of_simulations, dtype=float) 
    scores_int = np.zeros(number_of_simulations, dtype=float) 

    #For Number of simulations
    for ns in range(0, number_of_simulations): # repeated over the number of simulations
        print(ns,n)
        # Here select fitness values and dependencies
        #Equal k
        fitness_values = np.random.rand(n, 2**(k+1))
        dependencies = depend_evenk(n, k)
        
        #Find pairs of dependencies for accuracy calculations
        my_pair_int = []
        for item in range(len(dependencies)):
            my_dep = dependencies[item]
            for ant in my_dep:
                my_pair_int.append((ant, item))
        #pair not in dependencies
        all_pos_pairs = list(itertools.product(list(range(0,n)), repeat=2))
        non_deps = []
        for item in all_pos_pairs: 
            if item not in my_pair_int and item[0] != item[1]:
                non_deps.append(item)
        
        outputs = []
        my_pos_relats = [0,1]
        pos_config = list(itertools.product(my_pos_relats, repeat=n))
        for dat in range(len(pos_config)):
            choice_configuration = np.array(pos_config[dat])
            my_perf = perffunc_evenk(choice_configuration, k, n, dependencies, fitness_values, 0)
            outputs.append(my_perf)
        max_value_tot = max(outputs)
        min_value_tot = min(outputs)
        max_index = outputs.index(max_value_tot)
        best_model_tot = pos_config[max_index]

        #BOTH SIMULTANEOUS AND SEQUENTIAL CHOICE
        #BASED ON SAME M COGNITIVE REPRESENTATION
        #OF SAME ENVIRONMENT
        #WITH SAME RANDOM STARTING GUESS FOR STRATEGY
        for m_it in range(len(m_list)):
            m = m_list[m_it]
            #Random Choice of M Choices for Representation
            my_rep = random.sample(list(range(n)), m)
            not_in_rep = list(set(list(range(n))) - set(list(my_rep)))
            
            #SIMULTANEOUS CHOICE
            #using representation
            choices_int = np.zeros((number_of_periods,n), dtype=int)
            performances_int = np.zeros(number_of_periods, dtype=float)
            
            
            #dependencies for accuracy of M
            my_m_deps_int = list(itertools.product(my_rep, repeat=2))
            my_m_deps = my_m_deps_int.copy()
            for item in my_m_deps_int: 
                if item[0] == item[1]:
                    my_m_deps.remove(item)
            TP = 0
            FP = 0
            for item in my_m_deps: #represented
                if item in my_pair_int: #true representation TP
                    TP += 1
                else: #false representation FP
                    FP += 1
            TN = 0
            FN = 0
            not_reps = list(set(list(all_pos_pairs)) - set(list(my_m_deps)))
            for item in not_reps: #not represented
                if item in non_deps: #if not true dependency TN
                    TN += 1
                else: #if true dependency FN
                    FN += 1
            accuracy = (TP + TN) / (TP+FP+TN+FN)
            accuracy = (TP + TN) / (n*(n-1))
            precision = (TP) / (TP+FP)
            recall = (TP) / (TP + FN)
            acc_sim_list[m_it][ns] = accuracy
            pre_sim_list[m_it][ns] = precision
            rec_sim_list[m_it][ns] = recall
            
            for t in range(1, number_of_periods):
                #Initial Random Choice
                if t == 1:
                    my_choice_config_init = list(np.random.randint(0, 2, n))
                    choice_configuration = np.array(my_choice_config_init)
                    bestch = choice_configuration
                else:
                    my_choices = choices_int.tolist()[:t-1]
                    perfy = performances_int.tolist()[:t-1]
                    max_index = np.argmax(performances_int) 
                    bestch = choices_int[max_index,:] 
                    my_choice_config = list(bestch)
                    #For all choice not in representation, choose max performing value
                    if len(not_in_rep) > 0:
                        for my_it in not_in_rep:
                            my_0s_perf = []
                            my_1s_perf = []
                            for itr in range(len(my_choices)):
                                if my_choices[itr][my_it] == 0:
                                    my_0s_perf.append(perfy[itr])
                                else:
                                    my_1s_perf.append(perfy[itr])
                            x_0 = np.mean(my_0s_perf)
                            x_1 = np.mean(my_1s_perf)
                            #print(x_0, x_1)
                            if math.isnan(x_0) and math.isnan(x_1):
                                my_choice_config[my_it] = my_choice_config[my_it]
                            elif math.isnan(x_0):
                                my_choice_config[my_it] = 0
                            elif math.isnan(x_1):
                                my_choice_config[my_it] = 1
                            elif x_0 > x_1:
                                my_choice_config[my_it] = 0
                            else:
                                my_choice_config[my_it] = 1
                    #Select one random strategic choice in M to change
                    my_change = random.choice(my_rep)
                    rep_index = my_rep.index(my_change)
                    #Change value from the best performing strategy thus far
                    my_choice_config[my_change] = 1- my_choice_config[my_change]
                    #Find most performant combo of all M choices in representation
                    #Conditional on choice in M being set to explore value
                    pos_config_m_all = list(itertools.product(my_pos_relats, repeat=m))
                    pos_config_m = []
                    for item in pos_config_m_all: 
                        if item[rep_index] == my_choice_config[my_change]:
                            pos_config_m.append(item)
                    avg_perf_config = []
                    for config in range(len(pos_config_m)):
                        my_config = pos_config_m[config]
                        my_perf_list = []
                        for chc in range(len(my_choices)):
                            count = 0
                            for iterl in range(len(my_rep)):
                                my_filter = my_rep[iterl]
                                if my_choices[chc][my_filter] == my_config[iterl]:
                                    count += 1
                            if count == m:
                                my_perf_list.append(perfy[chc])
                        if len(my_perf_list) > 0:
                            my_avg = np.mean(my_perf_list)
                        else:
                            my_avg = 0
                        avg_perf_config.append(my_avg)
                    max_index_config = avg_perf_config.index(max(avg_perf_config))
                    my_max_config = pos_config_m[max_index_config]
                    for item in range(len(my_rep)):
                        my_choice_config[my_rep[item]] = my_max_config[item]
                    #If this exact combo has been tried before
                    #Randomly change one item
                    choice_configuration = np.array(my_choice_config)
                    if any((choices_int[:]==choice_configuration).all(1)): 
                        ch = random.choice(list(range(n)))
                        choice_configuration[ch] = 1- choice_configuration[ch]
                current_performance=perffunc_evenk(choice_configuration, k, n, dependencies, fitness_values, t)
                # we save performance and choices to the respective vectors
                choices_int[t-1] = choice_configuration  # this function records your current choice
                performances_int[t-1] = (current_performance - min_value_tot) / (max_value_tot - min_value_tot)   # and performance associated with it
            #SEQUENTIAL CHOICE
            choices = np.zeros((number_of_periods,n), dtype=int)
            performances = np.zeros(number_of_periods, dtype=float)
            
            #dependencies for accuracy of sequential
            my_deps = []
            for item in range(len(my_rep)):
                if item == 0:
                    woohoo = 1
                else:
                    my_rep_before = my_rep[:item]
                    for item2 in range(len(my_rep_before)):
                        my_deps.append((my_rep[item2], my_rep[item]))
            TP = 0
            FP = 0
            for item in my_deps: 
                if item in my_pair_int:
                    TP += 1
                else:
                    FP += 1
            TN = 0
            FN = 0
            not_reps = list(set(list(all_pos_pairs)) - set(list(my_deps)))
            for item in not_reps: #not represented
                if item in non_deps: #if not true dependency TN
                    TN += 1
                else: #if true dependency
                    FN += 1
            accuracy = (TP + TN) / (TP+FP+TN+FN)
            accuracy = (TP + TN) / (n*(n-1))
            precision = (TP) / (TP+FP)
            recall = (TP)/(TP+FN)
            #print(TP, TN, accuracy)
            acc_seq_list[m_it][ns] = accuracy
            pre_seq_list[m_it][ns] = precision
            rec_seq_list[m_it][ns] = recall
            
            
            for t in range(1, number_of_periods):
                #Initial Random Choice
                if t == 1:
                    choice_configuration = np.array(my_choice_config_init)
                    bestch = choice_configuration
                else:
                    my_choices = choices.tolist()[:t-1]
                    perfy = performances.tolist()[:t-1]
                    max_index = np.argmax(performances) # index of the highest performance
                    bestch = choices[max_index,:] 
                    my_choice_config = list(bestch)
                    #my_theory = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]
                    #0->1->2->3->4->5->6->7->8->9
                    #choose a random choice to change
                    my_rando = random.choice(my_rep)
                    my_index = my_rep.index(my_rando)
                    #Change consequents
                    to_change = my_rep[my_index:]
                    to_consider = my_rep[:my_index]
                    my_choice_config[to_change[0]] = 1-my_choice_config[to_change[0]]
                    my_choicy = my_choices.copy()
                    perfyty = perfy.copy()
                    my_dat_x = []
                    my_dat_y = []
                    new_choices = []
                    new_perfy = [] 
                    if len(to_consider) > 0:
                        for item in range(len(to_consider)):
                            my_filter = to_consider[item]
                            valley = my_choice_config[to_consider[0]]
                            for chc in range(len(my_choicy)):
                                if my_choicy[chc][my_filter] == valley:
                                    new_choices.append(my_choicy[chc])
                                    new_perfy.append(perfyty[chc])
                                    my_dat_x.append(my_choicy[chc][my_test])
                                    my_dat_y.append(perfyty[chc])
                    val_to_filt = my_choice_config[to_change[0]]
                    if len(my_dat_x) > 0 or len(to_consider) == 0:
                        for item in range(len(to_change)-1):
                            my_dat_x = []
                            my_dat_y = []
                            new_choices = []
                            new_perfy = []
                            my_test = to_change[item+1]
                            my_filter = to_change[item]
                            for chc in range(len(my_choicy)):
                                if my_choicy[chc][my_filter] == val_to_filt:
                                    new_choices.append(my_choicy[chc])
                                    new_perfy.append(perfyty[chc])
                                    my_dat_x.append(my_choicy[chc][my_test])
                                    my_dat_y.append(perfyty[chc])
                            my_dat["x"] = my_dat_x
                            my_dat["y"] = my_dat_y
                            my_data_frame = pd.DataFrame(my_dat)
                            avg_x_0 = my_data_frame[my_data_frame['x'] == 0]
                            x_0 = avg_x_0["y"].mean()
                            avg_x_1 = my_data_frame[my_data_frame['x'] == 1]
                            x_1 = avg_x_1["y"].mean()
                            #print(x_0, x_1)
                            if math.isnan(x_0) and math.isnan(x_1):
                                my_choice_config[my_test] = my_choice_config[my_test]
                            elif math.isnan(x_0):
                                my_choice_config[my_test] = 0
                            elif math.isnan(x_1):
                                my_choice_config[my_test] = 1
                            elif x_0 > x_1:
                                my_choice_config[my_test] = 0
                            elif x_1 > x_0:
                                my_choice_config[my_test] = 1
                            val_to_filt = my_choice_config[my_test]
                            my_choicy = new_choices.copy()
                            perfyty = new_perfy.copy()
                    if len(not_in_rep) > 0:
                        for my_it in not_in_rep:
                            my_0s_perf = []
                            my_1s_perf = []
                            for itr in range(len(my_choices)):
                                if my_choices[itr][my_it] == 0:
                                    my_0s_perf.append(perfy[itr])
                                else:
                                    my_1s_perf.append(perfy[itr])
                            x_0 = np.mean(my_0s_perf)
                            x_1 = np.mean(my_1s_perf)
                            #print(x_0, x_1)
                            if math.isnan(x_0) and math.isnan(x_1):
                                my_choice_config[my_it] = my_choice_config[my_it]
                            elif math.isnan(x_0):
                                my_choice_config[my_it] = 0
                            elif math.isnan(x_1):
                                my_choice_config[my_it] = 1
                            elif x_0 > x_1:
                                my_choice_config[my_it] = 0
                            else:
                                my_choice_config[my_it] = 1
                #If this exact combo has been tried before
                #Randomly change one item
                choice_configuration = np.array(my_choice_config)
                if any((choices[:]==choice_configuration).all(1)):
                    ch = random.choice(list(range(n)))
                    choice_configuration[ch] = 1- choice_configuration[ch]
                # we calculate the performance of this choice vector
                current_performance=perffunc_evenk(choice_configuration, k, n, dependencies, fitness_values, t)
                # we save performance and choices to the respective vectors
                choices[t-1] = choice_configuration  # this function records your current choice
                performances[t-1] = (current_performance - min_value_tot) / (max_value_tot - min_value_tot)   # and performance associated with it             # after all time periods (t) we calculate the score
            scores_int[ns] = np.max(performances_int)   # The score is the max of this run
            max_perf_forM[m] = max_perf_forM[m] + [scores_int[ns]]
            scores[ns] = np.max(performances)   # The score is the max of this run, seq
            max_perf_forO[m] = max_perf_forO[m] + [scores[ns]]
            #Shared Proportion of Strategies by Link Type Calculation
            if n_list[n_it] == 10:
                shared = 0
                for item in choices:
                    my_item = list(item)
                    in_list = 0
                    for item2 in choices_int: 
                        my_item2 = list(item2)
                        if my_item == my_item2:
                            in_list = 1
                    if in_list == 1:
                        shared += 1
                shared_bym[m_it][ns] = shared
        #Accuracy Calcs
        for mer in range(len(m_list)): 
            avg_acc_sim[mer][n_it] = np.mean(acc_sim_list[mer])
            avg_pre_sim[mer][n_it] = np.mean(pre_sim_list[mer])
            avg_rec_sim[mer][n_it] = np.mean(rec_sim_list[mer])
            avg_perf_sim[mer][n_it] = np.mean(perf_sim_list[mer])
            avg_acc_seq[mer][n_it] = np.mean(acc_seq_list[mer])
            avg_pre_seq[mer][n_it] = np.mean(pre_seq_list[mer])
            avg_rec_seq[mer][n_it] = np.mean(rec_seq_list[mer])
            avg_perf_seq[mer][n_it] = np.mean(perf_seq_list[mer])
        


# ## FIGURE 4: Average Proportion of Shared Strategies between Causal and Associative Mental Models
# Figure 4 below shows the average proportion of shared strategies selected by a decision-maker using the same M strategic choices in an associative mental model vs. theory across 100 times periods and 1000 simulation runs (for strategic environment, N = 10, K = 3). 

# In[60]:


avg_shared = []
for m in range(len(m_list)): 
    avg_shared.append(np.mean(shared_bym[m])/number_of_periods)
    
my_dict = {"Representation Size (M)": m_list, "Avg Prop of Shared Strategies": avg_shared}
my_df = pd.DataFrame(my_dict)

my_title = "Average Proportion of Shared Strategies Between Representation Types"

#plot sales of each store as a line
# Set figure size (width, height) in inches
plt.figure(figsize = ( 7 , 5 ))
# Plot lineplot
sns.lineplot(data=my_df, x="Representation Size (M)", y="Avg Prop of Shared Strategies")
# Set label for x-axis
plt.xlabel( "Representation Size (M)" , size = 12 )

# Set label for y-axis
plt.ylabel("Avg Prop of Shared Strategies", size = 12 )

# Set title for figure
plt.title( my_title , size = 12 )

# Save the plot to a file
plt.savefig('Fig4_AvgProp.png') 


# ## FIGURE 5: Number of Links by Representation Size and Link Type
# 
# Figure 5 visualizes equation 1 and equation 2 across values of M, showing that the number of performance links represented by cause-and-effect mental models ( I_(cause-and-effect)) will always be half that of the number of performance links of the equivalent associative mental models (I_associative).

# In[61]:


M_list = list(range(0,20))
links = []
typer = []
my_m = []
for M in M_list:
    assoc_links = M*(M-1) #equation 1
    links.append(assoc_links)
    #print(assoc_links)
    typer.append("Associative")
    my_m.append(M)
    cause_links = (M*(M-1))/2 #equation 2
    #cause_links_test = 0
    #for ms in range(1,M+1): #test sum is the same
        #cause_links_test += ms-1
    #if cause_links == cause_links_test:
        #print("yay")
    links.append(cause_links)
    typer.append("Cause-and-Effect")
    my_m.append(M)
    
my_dict = {"Representation Size (M)": my_m, "Link Type": typer, "Number of Links": links}
my_df = pd.DataFrame(my_dict)

my_title = "Simplicity of Representation by Size (M) and Link Type"

#plot sales of each store as a line
# Set figure size (width, height) in inches
plt.figure(figsize = ( 7 , 5 ))
# Plot lineplot
sns.lineplot(data=my_df, x="Representation Size (M)", y="Number of Links", hue = "Link Type")
# Set label for x-axis
plt.xlabel( "Representation Size (M)" , size = 12 )

# Set label for y-axis
plt.ylabel(  "Number of Links", size = 12 )

# Set title for figure
plt.title( my_title , size = 12 )

# Save the plot to a file
plt.savefig('Fig4_Simp.png') 


# ## FIGURE 6A: Accuracy of Links by Representation Size (M) and Number of Strategic Choices (N) for Mental Models of Varying Link Types
# 
# Figure 6a shows the performance of mental models of varying size (M) relative to the number of links represented in the mental model (I) as the number of strategic choices (N) in the environment increases (with K = 3). 

# In[62]:


#Accuracy of Interdependencies and Performance by N
avg_max_bym_n = []
my_perfer = []
my_num_int = []
my_m = []
my_type = []
my_n = []


for iti in range(len(max_perf_forO)):
    #All N runs for a given Ms
    M = m_list[iti]
    my_m_ns_sim = avg_acc_sim[iti]
    my_m_ns_seq= avg_acc_seq[iti]
    my_m_ns_sim_perf = max_perf_forM[M]
    my_m_ns_seq_perf = max_perf_forO[M]
    for ner in range(len(n_list)):
        if n_list[ner] == 10:
            my_type.append("Associative")
            my_avg_assoc = np.mean(my_m_ns_sim_perf[(ner)*number_of_simulations:(ner+1)*number_of_simulations])
            my_perfer.append(my_avg_assoc)
            avg_max_bym_n.append(my_m_ns_sim[ner])
            my_n.append(n_list[ner])
            my_m.append("Size = " + str(M))
            my_type.append("Cause-and-Effect")
            my_avg_ce = np.mean(my_m_ns_seq_perf[(ner)*number_of_simulations:(ner+1)*number_of_simulations])
            my_perfer.append(my_avg_ce)
            avg_max_bym_n.append(my_m_ns_seq[ner])
            my_n.append(n_list[ner])
            my_m.append("Size = " + str(M))
   

my_dict = {"Number of Choices (N)":my_n, "Representation Size (M)": my_m, "Performance": my_perfer, "Link Type": my_type, "Accuracy of Links (Acc)": avg_max_bym_n}
my_df = pd.DataFrame(my_dict)

my_title = "Performance by Accuracy of Links (Acc) and Link Type"

#plot sales of each store as a line
# Set figure size (width, height) in inches
plt.figure(figsize = ( 7 , 5 ))
# Plot lineplot
sns.lineplot(data=my_df, x="Accuracy of Links (Acc)", y="Performance", hue = "Link Type")
# Set label for x-axis
plt.xlabel( "Accuracy of Links (Acc)" , size = 12 )

# Set label for y-axis
plt.ylabel( "Performance", size = 12 )

# Set title for figure
plt.title( my_title , size = 12 )


# Save the plot to a file
plt.savefig('Fig6A_Perf_Acc_LT.png')


# ## FIGURE 7: Performance by Link Type and Size (M)
# 
# Figure 7 shows the performance of strategies found by decision-makers using associative vs. causal mental model of size (M). (with N = 10 and K = 3).

# In[63]:


#Performance by Link Type and Size(M)
avg_max_bym_n = []
my_perfer = []
my_num_int = []
my_m = []
my_type = []
my_n = []


for iti in range(len(max_perf_forO)):
    #All N runs for a given Ms
    M = m_list[iti]
    my_m_ns_sim_perf = max_perf_forM[M]
    my_m_ns_seq_perf = max_perf_forO[M]
    for ner in range(len(n_list)):
        if n_list[ner] == 10:
            my_type.append("Associative")
            my_avg_assoc = np.mean(my_m_ns_sim_perf[(ner)*number_of_simulations:(ner+1)*number_of_simulations])
            my_perfer.append(my_avg_assoc)
            avg_max_bym_n.append(M*(M-1))
            my_n.append(n_list[ner])
            my_m.append(M)
            my_type.append("Cause-and-Effect")
            my_avg_ce = np.mean(my_m_ns_seq_perf[(ner)*number_of_simulations:(ner+1)*number_of_simulations])
            my_perfer.append(my_avg_ce)
            avg_max_bym_n.append((M*(M-1))/2)
            my_n.append(n_list[ner])
            my_m.append(M)
   

my_dict = {"Number of Choices (N)":my_n, "Representation Size (M)": my_m, "Performance": my_perfer, "Link Type": my_type, "Number of Links (I)": avg_max_bym_n}
my_df = pd.DataFrame(my_dict)

my_title = "Performance by Representation Size (M) and Link Type"

#plot sales of each store as a line
# Set figure size (width, height) in inches
plt.figure(figsize = ( 7 , 5 ))
# Plot lineplot
sns.lineplot(data=my_df, x="Representation Size (M)", y="Performance", hue = "Link Type")
# Set label for x-axis
plt.xlabel( "Representation Size (M)" , size = 12 )

# Set label for y-axis
plt.ylabel( "Performance", size = 12 )

# Set title for figure
plt.title( my_title , size = 12 )


# Save the plot to a file
plt.savefig('Fig7_Perf_M_LT.png')


# ## FIGURE 8: Performance by Number of Links (I) and Link Type
# Figure 8 shows the relationship between the performance of strategies found and the number of links in a mental model (I) by type of representation (associative vs. causal) for a strategic environment with N = 10 and K = 3.

# In[64]:


my_dict = {"Number of Choices (N)":my_n, "Representation Size (M)": my_m, "Performance": my_perfer, "Link Type": my_type, "Number of Links (I)": avg_max_bym_n}
my_df = pd.DataFrame(my_dict)

my_title = "Performance by Number of Links (I) and Link Type"

#plot sales of each store as a line
# Set figure size (width, height) in inches
plt.figure(figsize = ( 7 , 5 ))
# Plot lineplot
sns.lineplot(data=my_df, x="Number of Links (I)", y="Performance", hue = "Link Type")
# Set label for x-axis
plt.xlabel( "Number of Links (I)" , size = 12 )

# Set label for y-axis
plt.ylabel( "Performance", size = 12 )

# Set title for figure
plt.title( my_title , size = 12 )


# Save the plot to a file
plt.savefig('Fig8_Perf_I_LT.png')


# ## FIGURE 9: Performance by Number of Links (I) and Size (M)
# Figure 9 shows why this is the case, by graphing the performance of strategies selected and the number of performance links in a representation (I) by the number of strategic choices in the mental model (M).

# In[65]:


#Number of Links and Performance by N
avg_max_bym_n = []
my_perfer = []
my_num_int = []
my_m = []
my_type = []
my_n = []


for iti in range(len(max_perf_forO)):
    #All N runs for a given Ms
    M = m_list[iti]
    my_m_ns_sim_perf = max_perf_forM[M]
    my_m_ns_seq_perf = max_perf_forO[M]
    for ner in range(len(n_list)):
        if n_list[ner] == 10:
            my_type.append("Associative")
            my_avg_assoc = np.mean(my_m_ns_sim_perf[(ner)*number_of_simulations:(ner+1)*number_of_simulations])
            my_perfer.append(my_avg_assoc)
            avg_max_bym_n.append(M*(M-1))
            my_n.append(n_list[ner])
            my_m.append("Size = " + str(M))
            my_type.append("Cause-and-Effect")
            my_avg_ce = np.mean(my_m_ns_seq_perf[(ner)*number_of_simulations:(ner+1)*number_of_simulations])
            my_perfer.append(my_avg_ce)
            avg_max_bym_n.append((M*(M+1))/2)
            my_n.append(n_list[ner])
            my_m.append("Size = " + str(M))


my_dict = {"Number of Choices (N)":my_n, "Representation Size (M)": my_m, "Performance": my_perfer, "Link Type": my_type, "Number of Links (I)": avg_max_bym_n}
my_df = pd.DataFrame(my_dict)

my_title = "Performance by Number of Links (I) and Representation Size (M)"

#plot sales of each store as a line
# Set figure size (width, height) in inches
plt.figure(figsize = ( 7 , 5 ))
# Plot lineplot
sns.lineplot(data=my_df, x="Number of Links (I)", y="Performance", hue = "Representation Size (M)")
# Set label for x-axis
plt.xlabel( "Number of Links (I)" , size = 12 )

# Set label for y-axis
plt.ylabel( "Performance", size = 12 )

# Set title for figure
plt.title( my_title , size = 12 )


# Save the plot to a file
plt.savefig('Fig9_Perf_I_M.png')


# ## FIGURE 10: Performance by Accuracy of Links (Acc) and Link Type
# 
# Figure 10 shows the relationship between performance and accuracy of representation (Acc) by type of mental model (associative vs. theory) for a strategic environment of N = 10 and K = 3.

# In[66]:


#Accuracy of Interdependencies and Performance by N
avg_max_bym_n = []
my_perfer = []
my_num_int = []
my_m = []
my_type = []
my_n = []


for iti in range(len(max_perf_forO)):
    #All N runs for a given Ms
    M = m_list[iti]
    my_m_ns_sim = avg_acc_sim[iti]
    my_m_ns_seq= avg_acc_seq[iti]
    my_m_ns_sim_perf = max_perf_forM[M]
    my_m_ns_seq_perf = max_perf_forO[M]
    for ner in range(len(n_list)):
        if n_list[ner] == 10:
            my_type.append("Associative")
            my_avg_assoc = np.mean(my_m_ns_sim_perf[(ner)*number_of_simulations:(ner+1)*number_of_simulations])
            my_perfer.append(my_avg_assoc)
            avg_max_bym_n.append(my_m_ns_sim[ner])
            my_n.append(n_list[ner])
            my_m.append("Size = " + str(M))
            my_type.append("Cause-and-Effect")
            my_avg_ce = np.mean(my_m_ns_seq_perf[(ner)*number_of_simulations:(ner+1)*number_of_simulations])
            my_perfer.append(my_avg_ce)
            avg_max_bym_n.append(my_m_ns_seq[ner])
            my_n.append(n_list[ner])
            my_m.append("Size = " + str(M))
   

my_dict = {"Number of Choices (N)":my_n, "Representation Size (M)": my_m, "Performance": my_perfer, "Link Type": my_type, "Accuracy of Links (Acc)": avg_max_bym_n}
my_df = pd.DataFrame(my_dict)

my_title = "Performance by Accuracy of Links (Acc) and Link Type"

#plot sales of each store as a line
# Set figure size (width, height) in inches
plt.figure(figsize = ( 7 , 5 ))
# Plot lineplot
sns.lineplot(data=my_df, x="Accuracy of Links (Acc)", y="Performance", hue = "Link Type")
# Set label for x-axis
plt.xlabel( "Accuracy of Links (Acc)" , size = 12 )

# Set label for y-axis
plt.ylabel( "Performance", size = 12 )

# Set title for figure
plt.title( my_title , size = 12 )


# Save the plot to a file
plt.savefig('Fig10_Perf_Acc_LT.png')


# ## FIGURE 11: Performance by Accuracy of Links (Acc) and Representation Size (M)
# Figure 11 shows why accuracy in representation is actually associated with higher performance, by graphing the performance of strategies selected by accuracy of representation (Acc) and by size of the mental model (M). 

# In[67]:


my_title = "Performance by Accuracy of Links (Acc) and Representation Size (M)"

#plot sales of each store as a line
# Set figure size (width, height) in inches
plt.figure(figsize = ( 7 , 5 ))
# Plot lineplot
sns.lineplot(data=my_df, x="Accuracy of Links (Acc)", y="Performance", hue = "Representation Size (M)")
# Set label for x-axis
plt.xlabel( "Accuracy of Links (Acc)" , size = 12 )

# Set label for y-axis
plt.ylabel( "Performance", size = 12 )

# Set title for figure
plt.title( my_title , size = 12 )


# Save the plot to a file
plt.savefig('Fig11_Perf_Acc_M.png')


# ## Appendix B, Across N
# Figure 7 shows the performance of strategies found by decision-makers using associative vs. causal mental model of size (M). (with N = 10 and K = 3). These results are stable across strategic environments of varying N and K, as shown in Appendix B. 

# In[ ]:


#Accuracy of Interdependencies and Performance by N
avg_max_bym_n = []
my_perfer = []
my_num_int = []
my_m = []
my_type = []
my_n = []


for iti in range(len(max_perf_forO)):
    #All N runs for a given Ms
    M = m_list[iti]
    my_m_ns_sim = avg_acc_sim[iti]
    my_m_ns_seq= avg_acc_seq[iti]
    my_m_ns_sim_perf = max_perf_forM[M]
    my_m_ns_seq_perf = max_perf_forO[M]
    for ner in range(len(n_list)):
        my_type.append("Associative")
        my_avg_assoc = np.mean(my_m_ns_sim_perf[(ner)*number_of_simulations:(ner+1)*number_of_simulations])
        my_perfer.append(my_avg_assoc)
        avg_max_bym_n.append(my_m_ns_sim[ner])
        my_n.append(n_list[ner])
        my_m.append("Size = " + str(M))
        my_type.append("Cause-and-Effect")
        my_avg_ce = np.mean(my_m_ns_seq_perf[(ner)*number_of_simulations:(ner+1)*number_of_simulations])
        my_perfer.append(my_avg_ce)
        avg_max_bym_n.append(my_m_ns_seq[ner])
        my_n.append(n_list[ner])
        my_m.append("Size = " + str(M))
   

my_dict = {"Number of Choices (N)":my_n, "Representation Size (M)": my_m, "Performance": my_perfer, "Link Type": my_type, "Accuracy of Links": avg_max_bym_n}
my_df = pd.DataFrame(my_dict)

my_title = "Performance by Representation Size (M) Across N"

#
#plot sales of each store as a line
# Set figure size (width, height) in inches
plt.figure(figsize = ( 7 , 5 ))
g = sns.FacetGrid(my_df, col="Number of Choices (N)", hue = "Link Type", col_wrap=5)
g.map(sns.lineplot, "Representation Size (M)","Performance")
g.add_legend()

# Save the plot to a file
g.savefig('AppendixB_Perf_M_N.png')


# ## Appendix C, Across N
# 
# Because the number of links in a representation isn’t dependent on features of the environment (N or K), these results hold across all strategic environments as shown in Appendix C.

# In[43]:


#Number of Links and Performance by N
avg_max_bym_n = []
my_perfer = []
my_num_int = []
my_m = []
my_type = []
my_n = []


for iti in range(len(max_perf_forO)):
    #All N runs for a given Ms
    M = m_list[iti]
    my_m_ns_sim_perf = max_perf_forM[M]
    my_m_ns_seq_perf = max_perf_forO[M]
    for ner in range(len(n_list)):
        my_type.append("Associative")
        my_avg_assoc = np.mean(my_m_ns_sim_perf[(ner)*number_of_simulations:(ner+1)*number_of_simulations])
        my_perfer.append(my_avg_assoc)
        avg_max_bym_n.append(M*(M-1))
        my_n.append(n_list[ner])
        my_m.append("Size = " + str(M))
        my_type.append("Cause-and-Effect")
        my_avg_ce = np.mean(my_m_ns_seq_perf[(ner)*number_of_simulations:(ner+1)*number_of_simulations])
        my_perfer.append(my_avg_ce)
        avg_max_bym_n.append((M*(M+1))/2)
        my_n.append(n_list[ner])
        my_m.append("Size = " + str(M))
   

my_dict = {"Number of Choices (N)":my_n, "Representation Size (M)": my_m, "Performance": my_perfer, "Link Type": my_type, "Number of Links (I)": avg_max_bym_n}
my_df = pd.DataFrame(my_dict)

my_title = "Performance by Number of Links (I) and Representation Size (M) Across N"

#
#plot sales of each store as a line
# Set figure size (width, height) in inches
plt.figure(figsize = ( 7 , 5 ))
g = sns.FacetGrid(my_df, col="Number of Choices (N)", hue = "Link Type", col_wrap=5)
g.map(sns.lineplot, "Number of Links (I)","Performance")
g.add_legend()


# Save the plot to a file
g.savefig('AppendixC2_Perf_Acc_N.png')


# In[44]:


#Number of Links and Performance by N
avg_max_bym_n = []
my_perfer = []
my_num_int = []
my_m = []
my_type = []
my_n = []


for iti in range(len(max_perf_forO)):
    #All N runs for a given Ms
    M = m_list[iti]
    my_m_ns_sim_perf = max_perf_forM[M]
    my_m_ns_seq_perf = max_perf_forO[M]
    for ner in range(len(n_list)):
        my_type.append("Associative")
        my_avg_assoc = np.mean(my_m_ns_sim_perf[(ner)*number_of_simulations:(ner+1)*number_of_simulations])
        my_perfer.append(my_avg_assoc)
        avg_max_bym_n.append(M*(M-1))
        my_n.append(n_list[ner])
        my_m.append("Size = " + str(M))
        my_type.append("Cause-and-Effect")
        my_avg_ce = np.mean(my_m_ns_seq_perf[(ner)*number_of_simulations:(ner+1)*number_of_simulations])
        my_perfer.append(my_avg_ce)
        avg_max_bym_n.append((M*(M+1))/2)
        my_n.append(n_list[ner])
        my_m.append("Size = " + str(M))
   

my_dict = {"Number of Choices (N)":my_n, "Representation Size (M)": my_m, "Performance": my_perfer, "Link Type": my_type, "Number of Links (I)": avg_max_bym_n}
my_df = pd.DataFrame(my_dict)

my_title = "Performance by Number of Links (I) and Representation Size (M) Across N"

#
#plot sales of each store as a line
# Set figure size (width, height) in inches
plt.figure(figsize = ( 7 , 5 ))
g = sns.FacetGrid(my_df, col="Number of Choices (N)", hue = "Representation Size (M)", col_wrap=5)
g.map(sns.lineplot, "Number of Links (I)","Performance")
g.add_legend()

# Save the plot to a file
g.savefig('AppendixC1_Perf_Acc_N.png')


# ## Section 2: NK-Landscape, Varying K
# 
# Results of mental models with cause-and-effect vs. associative performance links searching across varying K, where N = 10. 

# In[45]:


#Set Seed
my_rand = 42
np.random.seed(my_rand)

#SET N to 10
n = 10 # K in NK
#List of Possible ks
k_list = list(range(3,8))
#List of Possible Non-Ordered Cognitions Sizes
m_list = list(range(3,n))


#Generate Empty Items to Store Results
#Pulling from object max_perf_forM
#Of the form: {3(M): [list of average of max performance by K]}
max_perf_forM = dict.fromkeys(m_list,[])
max_perf_forO = dict.fromkeys(m_list,[])

avg_acc_sim = np.zeros((len(m_list),len(k_list)), dtype=float)
avg_pre_sim = np.zeros((len(m_list),len(k_list)), dtype=float)
avg_rec_sim = np.zeros((len(m_list),len(k_list)), dtype=float)
avg_perf_sim = np.zeros((len(m_list),len(k_list)), dtype=float)

avg_acc_seq = np.zeros((len(m_list),len(k_list)), dtype=float)
avg_pre_seq = np.zeros((len(m_list),len(k_list)), dtype=float)
avg_rec_seq = np.zeros((len(m_list),len(k_list)), dtype=float)
avg_perf_seq = np.zeros((len(m_list),len(k_list)), dtype=float)

#For each different N
for k_it in range(len(k_list)):
    k = k_list[k_it]
    
    # repeat the simulation for many runs
    # array to store final scores (max performance of a single simulation)
    scores = np.zeros(number_of_simulations, dtype=float) 
    scores_int = np.zeros(number_of_simulations, dtype=float) 
    
    #Accuracy of interdependencies simultaneous choice
    acc_sim_list = np.zeros((len(m_list),number_of_simulations), dtype=float)
    pre_sim_list = np.zeros((len(m_list),number_of_simulations), dtype=float)
    rec_sim_list = np.zeros((len(m_list),number_of_simulations), dtype=float)
    perf_sim_list = np.zeros((len(m_list),number_of_simulations), dtype=float)
    
    #Accuracy of interdependencies simultaneous choice
    acc_seq_list = np.zeros((len(m_list),number_of_simulations), dtype=float)
    pre_seq_list = np.zeros((len(m_list),number_of_simulations), dtype=float)
    rec_seq_list = np.zeros((len(m_list),number_of_simulations), dtype=float)
    perf_seq_list = np.zeros((len(m_list),number_of_simulations), dtype=float)
    
    #For Number of simulations
    for ns in range(0, number_of_simulations): # repeated over the number of simulations
        print(k, ns)
        # Here select fitness values and dependencies
        #Equal k
        fitness_values = np.random.rand(n, 2**(k+1))
        dependencies = depend_evenk(n, k)

        #Find pairs of dependencies for accuracy calculations
        my_pair_int = []
        for item in range(len(dependencies)):
            my_dep = dependencies[item]
            for ant in my_dep:
                my_pair_int.append((ant, item))
        #pair not in dependencies
        all_pos_pairs = list(itertools.product(list(range(0,n)), repeat=2))
        non_deps = []
        for item in all_pos_pairs: 
            if item not in my_pair_int and item[0] != item[1]:
                non_deps.append(item)
        
        outputs = []
        my_pos_relats = [0,1]
        pos_config = list(itertools.product(my_pos_relats, repeat=n))
        for dat in range(len(pos_config)):
            choice_configuration = np.array(pos_config[dat])
            my_perf = perffunc_evenk(choice_configuration, k, n, dependencies, fitness_values, 0)
            outputs.append(my_perf)
        max_value_tot = max(outputs)
        min_value_tot = min(outputs)
        max_index = outputs.index(max_value_tot)
        best_model_tot = pos_config[max_index]


        #BOTH SIMULTANEOUS AND SEQUENTIAL CHOICE
        #BASED ON SAME M COGNITIVE REPRESENTATION
        #OF SAME ENVIRONMENT
        for m_it in range(len(m_list)):
            m = m_list[m_it]
            #Random Choice of M Choices for Representation
            my_rep = random.sample(list(range(n)), m)
            not_in_rep = list(set(list(range(n))) - set(list(my_rep)))
            
            #SIMULTANEOUS CHOICE
            #using representation
            choices_int = np.zeros((number_of_periods,n), dtype=int)
            performances_int = np.zeros(number_of_periods, dtype=float)
            
            #dependencies for accuracy of M
            my_m_deps_int = list(itertools.product(my_rep, repeat=2))
            my_m_deps = my_m_deps_int.copy()
            for item in my_m_deps_int: 
                if item[0] == item[1]:
                    my_m_deps.remove(item)
            TP = 0
            FP = 0
            for item in my_m_deps: 
                if item in my_pair_int:
                    TP += 1
                else:
                    FP += 1
            TN = 0
            FN = 0
            not_reps = list(set(list(all_pos_pairs)) - set(list(my_m_deps)))
            for item in not_reps: #not represented
                if item in non_deps: #if not true dependency TN
                    TN += 1
                else: #if true dependency
                    FN += 1

            accuracy = (TP + TN) / (TP+FP+TN+FN)
            accuracy = (TP + TN) / (n*(n-1))
            precision = (TP) / (TP+FP)
            recall = (TP) / (TP + FN)
            acc_sim_list[m_it][ns] = accuracy
            pre_sim_list[m_it][ns] = precision
            rec_sim_list[m_it][ns] = recall
            
            
            for t in range(1, number_of_periods):
                #Initial Random Choice
                if t == 1:
                    my_choice_config_init = list(np.random.randint(0, 2, n))
                    choice_configuration = np.array(my_choice_config_init)
                    bestch = choice_configuration
                else:
                    my_choices = choices_int.tolist()[:t-1]
                    perfy = performances_int.tolist()[:t-1]
                    max_index = np.argmax(performances_int) 
                    bestch = choices_int[max_index,:] 
                    my_choice_config = list(bestch)
                    #For all choice not in representation, choose max performing value
                    if len(not_in_rep) > 0:
                        for my_it in not_in_rep:
                            my_0s_perf = []
                            my_1s_perf = []
                            for itr in range(len(my_choices)):
                                if my_choices[itr][my_it] == 0:
                                    my_0s_perf.append(perfy[itr])
                                else:
                                    my_1s_perf.append(perfy[itr])
                            x_0 = np.mean(my_0s_perf)
                            x_1 = np.mean(my_1s_perf)
                            #print(x_0, x_1)
                            if math.isnan(x_0) and math.isnan(x_1):
                                my_choice_config[my_it] = my_choice_config[my_it]
                            elif math.isnan(x_0):
                                my_choice_config[my_it] = 0
                            elif math.isnan(x_1):
                                my_choice_config[my_it] = 1
                            elif x_0 > x_1:
                                my_choice_config[my_it] = 0
                            else:
                                my_choice_config[my_it] = 1
                    #Select one random strategic choice in M to change
                    my_change = random.choice(my_rep)
                    rep_index = my_rep.index(my_change)
                    #Change value from the best performing strategy thus far
                    my_choice_config[my_change] = 1- my_choice_config[my_change]
                    #Find most performant combo of all M choices in representation
                    #Conditional on choice in M being set to explore value
                    pos_config_m_all = list(itertools.product(my_pos_relats, repeat=m))
                    pos_config_m = []
                    for item in pos_config_m_all: 
                        if item[rep_index] == my_choice_config[my_change]:
                            pos_config_m.append(item)
                    avg_perf_config = []
                    for config in range(len(pos_config_m)):
                        my_config = pos_config_m[config]
                        my_perf_list = []
                        for chc in range(len(my_choices)):
                            count = 0
                            for iterl in range(len(my_rep)):
                                my_filter = my_rep[iterl]
                                if my_choices[chc][my_filter] == my_config[iterl]:
                                    count += 1
                            if count == m:
                                my_perf_list.append(perfy[chc])
                        if len(my_perf_list) > 0:
                            my_avg = np.mean(my_perf_list)
                        else:
                            my_avg = 0
                        avg_perf_config.append(my_avg)
                    max_index_config = avg_perf_config.index(max(avg_perf_config))
                    my_max_config = pos_config_m[max_index_config]
                    for item in range(len(my_rep)):
                        my_choice_config[my_rep[item]] = my_max_config[item]
                    #If this exact combo has been tried before
                    #Randomly change one item
                    choice_configuration = np.array(my_choice_config)
                    if any((choices_int[:]==choice_configuration).all(1)): 
                        ch = random.choice(list(range(n)))
                        choice_configuration[ch] = 1- choice_configuration[ch]
                current_performance=perffunc_evenk(choice_configuration, k, n, dependencies, fitness_values, t)
                # we save performance and choices to the respective vectors
                choices_int[t-1] = choice_configuration  # this function records your current choice
                performances_int[t-1] = (current_performance - min_value_tot) / (max_value_tot - min_value_tot)   # and performance associated with it
            
            #SEQUENTIAL CHOICE
            choices = np.zeros((number_of_periods,n), dtype=int)
            performances = np.zeros(number_of_periods, dtype=float)
            
            #dependencies for accuracy of sequential
            my_deps = []
            for item in range(len(my_rep)):
                if item == 0:
                    woohoo = 1
                else:
                    my_rep_before = my_rep[:item]
                    for item2 in range(len(my_rep_before)):
                        my_deps.append((my_rep[item2], my_rep[item]))
            TP = 0
            FP = 0
            for item in my_deps: 
                if item in my_pair_int:
                    TP += 1
                else:
                    FP += 1
            TN = 0
            FN = 0
            not_reps = list(set(list(all_pos_pairs)) - set(list(my_deps)))
            for item in not_reps: #not represented
                if item in non_deps: #if not true dependency TN
                    TN += 1
                else: #if true dependency
                    FN += 1
            
            accuracy = (TP + TN) / (TP+FP+TN+FN)
            accuracy = (TP + TN) / (n*(n-1))
            precision = (TP) / (TP+FP)
            recall = (TP)/(TP+FN)
            #print(TP, TN, accuracy)
            acc_seq_list[m_it][ns] = accuracy
            pre_seq_list[m_it][ns] = precision
            rec_seq_list[m_it][ns] = recall
            
            for t in range(1, number_of_periods):
                #Initial Random Choice
                if t == 1:
                    choice_configuration = np.array(my_choice_config_init)
                    bestch = choice_configuration
                else:
                    my_choices = choices.tolist()[:t-1]
                    perfy = performances.tolist()[:t-1]
                    max_index = np.argmax(performances) # index of the highest performance
                    bestch = choices[max_index,:] 
                    my_choice_config = list(bestch)
                    #my_theory = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]
                    #0->1->2->3->4->5->6->7->8->9
                    #choose a random choice to change
                    my_rando = random.choice(my_rep)
                    my_index = my_rep.index(my_rando)
                    #Change consequents
                    to_change = my_rep[my_index:]
                    to_consider = my_rep[:my_index]
                    my_choice_config[to_change[0]] = 1-my_choice_config[to_change[0]]
                    my_choicy = my_choices.copy()
                    perfyty = perfy.copy()
                    my_dat_x = []
                    my_dat_y = []
                    new_choices = []
                    new_perfy = [] 
                    if len(to_consider) > 0:
                        for item in range(len(to_consider)):
                            my_filter = to_consider[item]
                            valley = my_choice_config[to_consider[0]]
                            for chc in range(len(my_choicy)):
                                if my_choicy[chc][my_filter] == valley:
                                    new_choices.append(my_choicy[chc])
                                    new_perfy.append(perfyty[chc])
                                    my_dat_x.append(my_choicy[chc][my_test])
                                    my_dat_y.append(perfyty[chc])
                    val_to_filt = my_choice_config[to_change[0]]
                    if len(my_dat_x) > 0 or len(to_consider) == 0:
                        for item in range(len(to_change)-1):
                            my_dat_x = []
                            my_dat_y = []
                            new_choices = []
                            new_perfy = []
                            my_test = to_change[item+1]
                            my_filter = to_change[item]
                            for chc in range(len(my_choicy)):
                                if my_choicy[chc][my_filter] == val_to_filt:
                                    new_choices.append(my_choicy[chc])
                                    new_perfy.append(perfyty[chc])
                                    my_dat_x.append(my_choicy[chc][my_test])
                                    my_dat_y.append(perfyty[chc])
                            my_dat["x"] = my_dat_x
                            my_dat["y"] = my_dat_y
                            my_data_frame = pd.DataFrame(my_dat)
                            avg_x_0 = my_data_frame[my_data_frame['x'] == 0]
                            x_0 = avg_x_0["y"].mean()
                            avg_x_1 = my_data_frame[my_data_frame['x'] == 1]
                            x_1 = avg_x_1["y"].mean()
                            #print(x_0, x_1)
                            if math.isnan(x_0) and math.isnan(x_1):
                                my_choice_config[my_test] = my_choice_config[my_test]
                            elif math.isnan(x_0):
                                my_choice_config[my_test] = 0
                            elif math.isnan(x_1):
                                my_choice_config[my_test] = 1
                            elif x_0 > x_1:
                                my_choice_config[my_test] = 0
                            elif x_1 > x_0:
                                my_choice_config[my_test] = 1
                            val_to_filt = my_choice_config[my_test]
                            my_choicy = new_choices.copy()
                            perfyty = new_perfy.copy()
                    if len(not_in_rep) > 0:
                        for my_it in not_in_rep:
                            my_0s_perf = []
                            my_1s_perf = []
                            for itr in range(len(my_choices)):
                                if my_choices[itr][my_it] == 0:
                                    my_0s_perf.append(perfy[itr])
                                else:
                                    my_1s_perf.append(perfy[itr])
                            x_0 = np.mean(my_0s_perf)
                            x_1 = np.mean(my_1s_perf)
                            #print(x_0, x_1)
                            if math.isnan(x_0) and math.isnan(x_1):
                                my_choice_config[my_it] = my_choice_config[my_it]
                            elif math.isnan(x_0):
                                my_choice_config[my_it] = 0
                            elif math.isnan(x_1):
                                my_choice_config[my_it] = 1
                            elif x_0 > x_1:
                                my_choice_config[my_it] = 0
                            else:
                                my_choice_config[my_it] = 1
                #If this exact combo has been tried before
                #Randomly change one item
                choice_configuration = np.array(my_choice_config)
                if any((choices[:]==choice_configuration).all(1)):
                    ch = random.choice(list(range(n)))
                    choice_configuration[ch] = 1- choice_configuration[ch]
                # we calculate the performance of this choice vector
                current_performance=perffunc_evenk(choice_configuration, k, n, dependencies, fitness_values, t)
                # we save performance and choices to the respective vectors
                choices[t-1] = choice_configuration  # this function records your current choice
                performances[t-1] = (current_performance - min_value_tot) / (max_value_tot - min_value_tot)   # and performance associated with it 
            # after all time periods (t) we calculate the score
            # after all time periods (t) we calculate the score
            scores_int[ns] = np.max(performances_int)   # The score is the max of this run
            max_perf_forM[m] = max_perf_forM[m] + [scores_int[ns]]
            scores[ns] = np.max(performances)   # The score is the max of this run, seq
            max_perf_forO[m] = max_perf_forO[m] + [scores[ns]]

        #Accuracy Calcs
        for mer in range(len(m_list)): 
            avg_acc_sim[mer][k_it] = np.mean(acc_sim_list[mer])
            avg_pre_sim[mer][k_it] = np.mean(pre_sim_list[mer])
            avg_rec_sim[mer][k_it] = np.mean(rec_sim_list[mer])
            avg_perf_sim[mer][k_it] = np.mean(perf_sim_list[mer])
            avg_acc_seq[mer][k_it] = np.mean(acc_seq_list[mer])
            avg_pre_seq[mer][k_it] = np.mean(pre_seq_list[mer])
            avg_rec_seq[mer][k_it] = np.mean(rec_seq_list[mer])
            avg_perf_seq[mer][k_it] = np.mean(perf_seq_list[mer])


# ## FIGURE 12: Performance of Mental Models by Accuracy (Acc) and Link Type Across Environments of Increasing Number of Performance Links Between Strategic Choices (K)
# 
# Figure 12, however, shows the relationship between performance and accuracy of representation (Acc) by type of mental model (associative vs. theory) for a strategic environment of N = 10 where K increases from 3 to 7.

# In[ ]:


#Accuracy of Interdependencies and Link Type by K
avg_max_bym_k = []
my_perfer = []
my_num_int = []
my_m = []
my_type = []
my_k = []


for iti in range(len(max_perf_forO)):
    #All N runs for a given Ms
    M = m_list[iti]
    my_m_ks_sim = avg_acc_sim[iti]
    my_m_ks_seq= avg_acc_seq[iti]
    my_m_ks_sim_perf = max_perf_forM[M]
    my_m_ks_seq_perf = max_perf_forO[M]
    for il in range(len(k_list)):
        my_type.append("Associative")
        my_avg_assoc = np.mean(my_m_ks_sim_perf[(il)*number_of_simulations:(il+1)*number_of_simulations])
        my_perfer.append(my_avg_assoc)
        avg_max_bym_k.append(my_m_ks_sim[il])
        my_k.append(k_list[il])
        my_m.append("Size = " + str(M))
        my_type.append("Cause-and-Effect")
        my_avg_ce = np.mean(my_m_ks_seq_perf[(il)*number_of_simulations:(il+1)*number_of_simulations])
        my_perfer.append(my_avg_ce)
        avg_max_bym_k.append(my_m_ks_seq[il])
        my_k.append(k_list[il])
        my_m.append("Size = " + str(M))
   

my_dict = {"Number of Links per Choice (K)":my_k, "Representation Size (M)": my_m, "Performance": my_perfer, "Link Type": my_type, "Accuracy of Links": avg_max_bym_k}
my_df = pd.DataFrame(my_dict)


my_title = "Performance by Accuracy of Links and Link Type Across K"

#
#plot sales of each store as a line
# Set figure size (width, height) in inches
plt.figure(figsize = ( 7 , 5 ))
g = sns.FacetGrid(my_df, col="Number of Links per Choice (K)", hue = "Link Type", col_wrap=5)
g.map(sns.lineplot, "Accuracy of Links","Performance")
g.add_legend()

# Save the plot to a file
g.savefig('Fig12_Acc_LI_K.png')


# ## FIGURE 13: Performance of Mental Models by Accuracy (Acc) and Size (M) Across Environments of Increasing Number of Performance Links Between Strategic Choices (K) 
# 
# Figure 13, graphs performance and accuracy by size of representation (M) across strategic environments with N = 10 and increasing value of K (from 3 to 7). 

# In[ ]:


#Accuracy of Interdependencies and Performance by K
avg_max_bym_k = []
my_perfer = []
my_num_int = []
my_m = []
my_type = []
my_k = []


for iti in range(len(max_perf_forO)):
    #All N runs for a given Ms
    M = m_list[iti]
    my_m_ks_sim = avg_acc_sim[iti]
    my_m_ks_seq= avg_acc_seq[iti]
    my_m_ks_sim_perf = max_perf_forM[M]
    my_m_ks_seq_perf = max_perf_forO[M]
    for il in range(len(k_list)):
        my_type.append("Associative")
        my_avg_assoc = np.mean(my_m_ks_sim_perf[(il)*number_of_simulations:(il+1)*number_of_simulations])
        my_perfer.append(my_avg_assoc)
        avg_max_bym_k.append(my_m_ks_sim[il])
        my_k.append(k_list[il])
        my_m.append("Size = " + str(M))
        my_type.append("Cause-and-Effect")
        my_avg_ce = np.mean(my_m_ks_seq_perf[(il)*number_of_simulations:(il+1)*number_of_simulations])
        my_perfer.append(my_avg_ce)
        avg_max_bym_k.append(my_m_ks_seq[il])
        my_k.append(k_list[il])
        my_m.append("Size = " + str(M))
   

my_dict = {"Number of Links per Choice (K)":my_k, "Representation Size (M)": my_m, "Performance": my_perfer, "Link Type": my_type, "Accuracy of Links": avg_max_bym_k}
my_df = pd.DataFrame(my_dict)

my_title = "Performance by Accuracy of Links and Representation Size (M) Across K"

#
#plot sales of each store as a line
# Set figure size (width, height) in inches
plt.figure(figsize = ( 7 , 5 ))
g = sns.FacetGrid(my_df, col="Number of Links per Choice (K)", hue = "Representation Size (M)", col_wrap=5)
g.map(sns.lineplot, "Accuracy of Links","Performance")
g.add_legend()

# Save the plot to a file
g.savefig('Fig13_Perf_Acc_K.png')


# ## APPENDIX B2: Across K
# Figure 7 shows the performance of strategies found by decision-makers using associative vs. causal mental model of size (M). (with N = 10 and K = 3). These results are stable across strategic environments of varying N and K, as shown in Appendix B. 

# In[ ]:


#Number of Links and Performance by K
avg_max_bym_k = []
my_perfer = []
my_num_int = []
my_m = []
my_type = []
my_k = []


for iti in range(len(max_perf_forO)):
    #All N runs for a given Ms
    M = m_list[iti]
    my_m_ks_sim_perf = max_perf_forM[M]
    my_m_ks_seq_perf = max_perf_forO[M]
    for il in range(len(k_list)):
        my_type.append("Associative")
        my_avg_assoc = np.mean(my_m_ks_sim_perf[(il)*number_of_simulations:(il+1)*number_of_simulations])
        my_perfer.append(my_avg_assoc)
        avg_max_bym_k.append(M*(M-1))
        my_k.append(k_list[il])
        my_m.append(M)
        my_type.append("Cause-and-Effect")
        my_avg_ce = np.mean(my_m_ks_seq_perf[(il)*number_of_simulations:(il+1)*number_of_simulations])
        my_perfer.append(my_avg_ce)
        avg_max_bym_k.append((M*(M+1))/2)
        my_k.append(k_list[il])
        my_m.append(M)
   

my_dict = {"Number of Links per Choice (K)":my_k, "Representation Size (M)": my_m, "Performance": my_perfer, "Link Type": my_type, "Number of Links (I)": avg_max_bym_k}
my_df = pd.DataFrame(my_dict)

my_title = "Performance by Number of Links (I) and Representation Size (M) Across K"

#
#plot sales of each store as a line
# Set figure size (width, height) in inches
plt.figure(figsize = ( 7 , 5 ))
g = sns.FacetGrid(my_df, col="Number of Links per Choice (K)", hue = "Link Type", col_wrap=5)
g.map(sns.lineplot, "Representation Size (M)","Performance")
g.add_legend()

# Save the plot to a file
g.savefig('AppendixB2_Perf_I_K.png')


# ## APPENDIX C2: Across K
# Because the number of links in a representation isn’t dependent on features of the environment (N or K), these results hold across all strategic environments as shown in Appendix C.

# In[ ]:


#Number of Links and Performance by K
avg_max_bym_k = []
my_perfer = []
my_num_int = []
my_m = []
my_type = []
my_k = []


for iti in range(len(max_perf_forO)):
    #All N runs for a given Ms
    M = m_list[iti]
    my_m_ks_sim_perf = max_perf_forM[M]
    my_m_ks_seq_perf = max_perf_forO[M]
    for il in range(len(k_list)):
        my_type.append("Associative")
        my_avg_assoc = np.mean(my_m_ks_sim_perf[(il)*number_of_simulations:(il+1)*number_of_simulations])
        my_perfer.append(my_avg_assoc)
        avg_max_bym_k.append(M*(M-1))
        my_k.append(k_list[il])
        my_m.append(M)
        my_type.append("Cause-and-Effect")
        my_avg_ce = np.mean(my_m_ks_seq_perf[(il)*number_of_simulations:(il+1)*number_of_simulations])
        my_perfer.append(my_avg_ce)
        avg_max_bym_k.append((M*(M+1))/2)
        my_k.append(k_list[il])
        my_m.append(M)
   

my_dict = {"Number of Links per Choice (K)":my_k, "Representation Size (M)": my_m, "Performance": my_perfer, "Link Type": my_type, "Number of Links (I)": avg_max_bym_k}
my_df = pd.DataFrame(my_dict)

my_title = "Performance by Number of Links (I) and Representation Size (M) Across K"

#
#plot sales of each store as a line
# Set figure size (width, height) in inches
plt.figure(figsize = ( 7 , 5 ))
g = sns.FacetGrid(my_df, col="Number of Links per Choice (K)", hue = "Link Type", col_wrap=5)
g.map(sns.lineplot, "Number of Links (I)","Performance")
g.add_legend()

# Save the plot to a file
g.savefig('AppendixC1_Perf_I_K.png')


# In[ ]:


#Number of Links and Performance by K
avg_max_bym_k = []
my_perfer = []
my_num_int = []
my_m = []
my_type = []
my_k = []


for iti in range(len(max_perf_forO)):
    #All N runs for a given Ms
    M = m_list[iti]
    my_m_ks_sim_perf = max_perf_forM[M]
    my_m_ks_seq_perf = max_perf_forO[M]
    for il in range(len(k_list)):
        my_type.append("Associative")
        my_avg_assoc = np.mean(my_m_ks_sim_perf[(il)*number_of_simulations:(il+1)*number_of_simulations])
        my_perfer.append(my_avg_assoc)
        avg_max_bym_k.append(M*(M-1))
        my_k.append(k_list[il])
        my_m.append("Size = " + str(M))
        my_type.append("Cause-and-Effect")
        my_avg_ce = np.mean(my_m_ks_seq_perf[(il)*number_of_simulations:(il+1)*number_of_simulations])
        my_perfer.append(my_avg_ce)
        avg_max_bym_k.append((M*(M+1))/2)
        my_k.append(k_list[il])
        my_m.append("Size = " + str(M))
   

my_dict = {"Number of Links per Choice (K)":my_k, "Representation Size (M)": my_m, "Performance": my_perfer, "Link Type": my_type, "Number of Links (I)": avg_max_bym_k}
my_df = pd.DataFrame(my_dict)

my_title = "Performance by Number of Links (I) and Representation Size (M) Across K"

#
#plot sales of each store as a line
# Set figure size (width, height) in inches
plt.figure(figsize = ( 7 , 5 ))
g = sns.FacetGrid(my_df, col="Number of Links per Choice (K)", hue = "Representation Size (M)", col_wrap=5)
g.map(sns.lineplot, "Number of Links (I)","Performance")
g.add_legend()

# Save the plot to a file
g.savefig('AppendixC1_Perf_I_K.png')

