#!/usr/bin/env python
# coding: utf-8

# # CODE: Any Old Theory Will Do: Why Cause-and-Effect Performance Links Form Parsimonious Mental Models of Complex Strategic Environments
# 
# **Author:** Madison Singell
# 
# **Acknowledgement:** Baseline NK-Model Code from the TOM NK-landscape competition
# 
# **Updated as of:** 1/27/2026
# 
# ## Goal: To measure how theories (mental models with cause-and-effect performance links) and mental models with associative performance links search for strategies across an NK-landscape, and to understand why the difference in performance link type of these mental models generates different strategy selection and performance. 
# 
# 
# ## Table of Contents
# 
# ### Section 0: Load Libraries, Generate Functions, and Set Parameters
#  * Section 0.1: Load Libraries
#  * Section 0.2: Specify NK-Landscape Functions
#  * Section 0.3: Set Parameters (T = 100, S = 10,000)
# 
# ### Section 1: NK-Landscape, N = 10, K = 5
# Results of mental models with cause-and-effect vs. associative performance links searching across an NK-landscape where N = 10 and K = 5. This includes the results for Figure 4, Figure 5, Figure 6, Figure 10, and Figure 11.
# * **FIGURE 4**: Average Proportion of Shared Strategies between Mental Model Types
#     * Figure 4 below shows the average proportion of shared strategies selected by a decision-maker using the same M strategic choices in an associative mental model vs. theory across 100 times periods and 10,000 simulation runs (for strategic environment, N = 10, K = 5).
# * **FIGURE 5**: Performance by Mental Model Size (M) and Type
#     * Figure 5 shows the performance of strategies found by decision-makers using associative vs. causal mental model of size (M). (with N = 10 and K = 5).
# * **FIGURE 6**: Number of Links by Mental Model Size (M) and Type
#     * Figure 6 visualizes the number of performance dependencies considered in associative mental models and theories across varying sizes of mental models (M) (for N=10, K=5).
# * **FIGURE 10**: Maximum Perfomance Attribution by Mental Model Size (M) and Type
#      * Figure 10 visualizes the maximum performance attribution based on strategic search using associative mental models and theories across varying sizes of mental models (M) (for N=10, K=5). 
# * **FIGURE 11**: Average Perfomance Attribution by Mental Model Size (M) and Type
#     * Figure 11 visualizes the average performance attribution based on strategic search using associative mental models and theories across varying sizes of mental models (M) (for N=10, K=5).   
#     
# ### Section 2: NK-Landscape, Varying N
# Results of mental models with cause-and-effect vs. associative performance links searching across an NK-landscape where K=5 and N varies. Parameters are set to T = 100, S = 1,000. This includes the results for Figure 7a, as well as Appendix results B1.
# * **FIGURE 7a**: Accuracy of Links by Mental Model Size (M) and Type Across Number of Strategic Choices (N)
#     * Figure 7a compares model link accuracy as the number of strategic choices (N) increases (with K = 5).
# * **FIGURE B1**:  Performance by Mental Model Size (M) and Type Across N
#     * Figure B1 shows the performance of strategies found by decision-makers using associative vs. causal mental model of size (M) across N (with K = 5).
# 
# ### Section 3: NK-Landscape, Varying K
# Results of mental models with cause-and-effect vs. associative performance links searching across an NK-landscape where N=10 and K varies. Parameters are set to T = 100, S = 1,000. This includes the results for Figure 7b, as well as Appendix results B2.
# * **FIGURE 7b**: Accuracy of Links by Mental Model Size (M) and Type, Across Interdependence of Strategic Choices (K)
#     * Figure 7b compares link accuracy as the number of performance links (K) increases (with N = 10).
# * **FIGURE B2**:  Performance by Mental Model Size (M) and Type Across K
#     * Figure B2 shows the performance of strategies found by decision-makers using associative vs. causal mental model of size (M) across K (with N = 10).  
# 

# ## Section 0: Load Libraries, Generate Functions, and Set Parameters
# 
# In this section I load the necessary libraries for the code to run, specify functions that will generate the NK landscape to search over, and set the parameters for the simulation run.

# ### Section 0.1: Load Libraries

# In[ ]:


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

# In[ ]:


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

# In[ ]:


#SET PARAMETERS
number_of_periods  = 100 #Number of time periods to search
number_of_simulations = 10000 # Number of simulations/iterations 
#Set at 100, 10,000 for paper results


# ## Section 1: NK-Landscape, N = 10, K = 5
# 
# Results of mental models with cause-and-effect vs. associative performance links searching across an NK-landscape where N = 10 and K = 5. This includes the results for Figure 4, Figure 5, Figure 6, Figure 10, and Figure 11.

# In[ ]:


#Set Seed
my_rand = 42
np.random.seed(my_rand)

#SET K to 5
k = 5 # K in NK
#Set N to 10
n = 10 # N in NK
#List of Possible Mental Model Sizes
m_list = list(range(3,10))


#Generate Empty Items to Store Results
#Associative model, simultaneous choice
simp_sim_list = np.zeros((len(m_list),number_of_simulations), dtype=float) #Simplicity of representation
acc_sim_list = np.zeros((len(m_list),number_of_simulations), dtype=float) #Accuracy of interdependencies
perf_sim_list = np.zeros((len(m_list),number_of_simulations), dtype=float) #Average Performance
max_perf_forsim = np.zeros((len(m_list),number_of_simulations), dtype=float) #Maximum Performance
acc_perf_at_sim = np.zeros((len(m_list),number_of_simulations), dtype=float) #Max Performance Attribution
acc_difmeans_at_sim = np.zeros((len(m_list),number_of_simulations), dtype=float) #Average Performance Attribution

#Accuracy of interdependencies, causal model, sequential choice
simp_seq_list = np.zeros((len(m_list),number_of_simulations), dtype=float) #Simplicity of representation
acc_seq_list = np.zeros((len(m_list),number_of_simulations), dtype=float) #Accuracy of interdependencies
perf_seq_list = np.zeros((len(m_list),number_of_simulations), dtype=float) #Average Performance
max_perf_forseq = np.zeros((len(m_list),number_of_simulations), dtype=float) #Maximum Performance
acc_perf_at_seq = np.zeros((len(m_list),number_of_simulations), dtype=float) #Max Performance Attribution
acc_difmeans_at_seq = np.zeros((len(m_list),number_of_simulations), dtype=float) #Average Performance Attribution

#Proportion of strategies shared, simulatneous vs. sequential
shared_bym = np.zeros((len(m_list),number_of_simulations), dtype=float) #Proportion of shared strategies

#Overall Performance
scores_sim = np.zeros(number_of_simulations, dtype=float) 
scores_seq = np.zeros(number_of_simulations, dtype=float) 

#For Number of simulations
for ns in range(0, number_of_simulations):
    print(ns,n)
    
    ## STEP 1: INITIALIZE NK-LANDSCAPE WITH N AND K.
    # Here select fitness values and dependencies
    fitness_values = np.random.rand(n, 2**(k+1))
    dependencies = depend_evenk(n, k)

    #Find pairs of dependencies contained in landscape for accuracy calculations
    my_pair_int = []
    for item in range(len(dependencies)):
        my_dep = dependencies[item]
        for ant in my_dep:
            my_pair_int.append((ant, item))
    #Pair not in dependencies
    all_pos_pairs = list(itertools.product(list(range(0,n)), repeat=2))
    non_deps = []
    for item in all_pos_pairs: 
        if item not in my_pair_int and item[0] != item[1]:
            non_deps.append(item)
    
    #Calculate actual average performance in environment by strategic choice (for average performance attribution)
    #And identify global maximum strategy (for maximum performance attribution)
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
    dif_means_perf_actual = [] 
    for my_nery in range(n):
        my_0s_perf = []
        my_1s_perf = []
        for itr in range(len(pos_config)):
            if pos_config[itr][my_nery] == 0:
                my_0s_perf.append(outputs[itr])
            else:
                my_1s_perf.append(outputs[itr])
        x_0 = np.mean(my_0s_perf)
        x_1 = np.mean(my_1s_perf)
        dif_means_perf_actual.append(x_0 - x_1)

    for m_it in range(len(m_list)):
        ## STEP 2: AT TIME T=0, RANDOMLY SELECT M STRATEGIC CHOICES FOR THE MENTAL MODEL
        ## AN ORDER (OR) FOR THE THEORY AND AN INITIAL RANDOM STRATEGY.
        m = m_list[m_it]
        my_rep = random.sample(list(range(n)), m)
        not_in_rep = list(set(list(range(n))) - set(list(my_rep)))
        my_choice_config_init = list(np.random.randint(0, 2, n))

        #Find Dependencies in Associative Model for accuracy calculation
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
        accuracy = (TP + TN) / (n*(n-1))
        acc_sim_list[m_it][ns] = accuracy
        simp_sim_list[m_it][ns] = len(my_m_deps_int)

      
        #ASSOCIATIVE MENTAL MODEL, SIMULTANEOUS CHOICE
        choices_sim = np.zeros((number_of_periods,n), dtype=int)
        performances_sim = np.zeros(number_of_periods, dtype=float)
    
        for t in range(1, number_of_periods):
            #Initial Random Choice
            if t == 1:
                choice_configuration = np.array(my_choice_config_init)
                bestch = choice_configuration
            ## STEP 3 SIMULTANEOUS: FOR EACH SUBSEQUENT T USE THE ASSOCIATIVE MENTAL MODEL TO SELECT A NEW STRATEGY 
            else:
                
                ## STEP 3.1 START WITH THE BEST PRIOR PERFORMING STRATEGY.
                my_choices = choices_sim.tolist()[:t-1]
                perfy = performances_sim.tolist()[:t-1]
                max_index = np.argmax(performances_sim) 
                bestch = choices_sim[max_index,:] 
                my_choice_config = list(bestch)
                
                ## STEP 3.2 SELECT A RANDOM STRATEGIC CHOICE A TO CHANGE FROM BEST.
                my_change = random.choice(my_rep)
                rep_index = my_rep.index(my_change)
                my_choice_config[my_change] = 1- my_choice_config[my_change]
                
                ## STEP 3.3 FOR MENTAL MODEL RUNS CALCULATE BEST AVERAGE PERFORMING STRATEGY
                ## CONDITIONAL ON THE VALUE OF CHOICE A SELECTED IN 3.2 FOR ALL M CHOICES. 
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
                
                ## STEP 3.5: FOR ALL N-M STRATEGIC CHOICES, SELECT BEST CHOICE BASED ON
                ## INDEPENDENTLY CONSIDERED PERFORMANCE.
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
                
                ## STEP 4: IF STRATEGY IN STEP 3 HAS ALREADY BEEN TRIED
                ## SELECT A RANDOM STRATEGIC CHOICE IN N TO CHANGE. 
                choice_configuration = np.array(my_choice_config)
                if any((choices_sim[:]==choice_configuration).all(1)): 
                    ch = random.choice(list(range(n)))
                    choice_configuration[ch] = 1- choice_configuration[ch]
                       
            #Record Strategy and Normalized Performance at T
            current_performance=perffunc_evenk(choice_configuration, k, n, dependencies, fitness_values, t)
            choices_sim[t-1] = choice_configuration  
            performances_sim[t-1] = (current_performance - min_value_tot) / (max_value_tot - min_value_tot)  
        
        #Calculate Maximum and Average Performance Attribution for Associative Mental Model
        my_choices = choices_sim.tolist()[:t-1]
        perfy = performances_sim.tolist()[:t-1]
        my_choicy_best_sim = []
        dif_means_perf_sim = []
        for my_nery in range(n):
            my_0s_perf = []
            my_1s_perf = []
            for itr in range(len(my_choices)):
                if my_choices[itr][my_nery] == 0:
                    my_0s_perf.append(perfy[itr])
                else:
                    my_1s_perf.append(perfy[itr])
            x_0 = np.mean(my_0s_perf)
            x_1 = np.mean(my_1s_perf)
            dif_means_perf_sim.append(x_0-x_1)
            if math.isnan(x_0):
                my_choicy_best_sim.append(1)
            elif math.isnan(x_1):
                my_choicy_best_sim.append(0)
            elif x_0 > x_1:
                my_choicy_best_sim.append(0)
            elif x_1 > x_0:
                my_choicy_best_sim.append(1)
            else:
                my_choicy_best_sim.append(1)
        my_differ = []
        cor_dif = np.corrcoef(dif_means_perf_actual,dif_means_perf_sim)
        my_corr = cor_dif[0][1]
        acc_difmeans_at_sim[m_it][ns] = my_corr
        best_model_list = list(best_model_tot)
        accuracy_sum = 0
        best_model_sim_share = []
        for item in range(len(my_choicy_best_sim)):
            if my_choicy_best_sim[item] == best_model_list[item]:
                accuracy_sum += 1
                best_model_sim_share.append(item)
        acc_perf_at_sim[m_it][ns] = accuracy_sum/(n)
        
        #Record Best Strategy Performance of Run for Associative Mental Model Search
        scores_sim[ns] = np.max(performances_sim)
        max_perf_forsim[m_it][ns] = scores_sim[ns]
        
        
        #THEORY SEARCH, SEQUENTIAL CHOICE
        choices = np.zeros((number_of_periods,n), dtype=int)
        performances = np.zeros(number_of_periods, dtype=float)

        #Find dependencies in theory for accuracy calculation
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
        accuracy = (TP + TN) / (n*(n-1))
        acc_seq_list[m_it][ns] = accuracy
        simp_seq_list[m_it][ns] = len(my_deps)


        for t in range(1, number_of_periods):
            #Initial Random Choice
            if t == 1:
                choice_configuration = np.array(my_choice_config_init)
                bestch = choice_configuration
            ## STEP 3 SEQUENTIAL: FOR EACH SUBSEQUENT T USE THEORY TO SELECT A NEW STRATEGY 
            else:
                ## STEP 3.1 START WITH THE BEST PRIOR PERFORMING STRATEGY.
                my_choices = choices.tolist()[:t-1]
                perfy = performances.tolist()[:t-1]
                max_index = np.argmax(performances)
                bestch = choices[max_index,:] 
                my_choice_config = list(bestch)
                
                ## STEP 3.2 SELECT A RANDOM STRATEGIC CHOICE A TO CHANGE FROM BEST.
                my_rando = random.choice(my_rep)
                my_index = my_rep.index(my_rando)
                
                ## STEP 3.4 FOR THEORY RUNS CALCULATE THE BEST AVERAGE PERFORMING STRATEGY 
                ## FOR PERFORMANCE CONSEQUENTS OF A ONLY CONDITIONAL ON THE VALUE OF CHOICE A
                ## AND ITS ANTECEDENTS SELECTED IN 2. USING ORDER (OR).
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
                                my_dat_x.append(my_choicy[chc][my_filter])
                                my_dat_y.append(perfyty[chc])
                val_to_filt = my_choice_config[to_change[0]]
                if len(my_dat_x) > 0 or len(to_consider) == 0:
                    for item in range(len(to_change)-1):
                        my_dat = {} 
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
                
                ## STEP 3.5: FOR ALL N-M STRATEGIC CHOICES, SELECT BEST CHOICE BASED ON
                ## INDEPENDENTLY CONSIDERED PERFORMANCE.
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
            
            ## STEP 4: IF STRATEGY IN STEP 3 HAS ALREADY BEEN TRIED
            ## SELECT A RANDOM STRATEGIC CHOICE IN N TO CHANGE. 
            choice_configuration = np.array(my_choice_config)
            if any((choices[:]==choice_configuration).all(1)):
                ch = random.choice(list(range(n)))
                choice_configuration[ch] = 1- choice_configuration[ch]
            
            #Record Strategy and Normalized Performance at T
            current_performance=perffunc_evenk(choice_configuration, k, n, dependencies, fitness_values, t)
            choices[t-1] = choice_configuration
            performances[t-1] = (current_performance - min_value_tot) / (max_value_tot - min_value_tot)     
        
        #Calculate Maximum and Average Performance Attribution for Theory
        my_choices = choices.tolist()[:t-1]
        perfy = performances.tolist()[:t-1]
        my_choicy_best_seq = []
        dif_means_perf_seq = [] 
        for my_nery in range(n):
            my_0s_perf = []
            my_1s_perf = []
            for itr in range(len(my_choices)):
                if my_choices[itr][my_nery] == 0:
                    my_0s_perf.append(perfy[itr])
                else:
                    my_1s_perf.append(perfy[itr])
            x_0 = np.mean(my_0s_perf)
            x_1 = np.mean(my_1s_perf)
            dif_means_perf_seq.append(x_0 - x_1)
            if math.isnan(x_0):
                my_choicy_best_seq.append(1)
            elif math.isnan(x_1):
                my_choicy_best_seq.append(0)
            elif x_0 > x_1:
                my_choicy_best_seq.append(0)
            elif x_1 > x_0:
                my_choicy_best_seq.append(1)
            else:
                my_choicy_best_seq.append(1)
        my_differ = []
        cor_dif = np.corrcoef(dif_means_perf_actual,dif_means_perf_seq)
        my_corr = cor_dif[0][1]
        acc_difmeans_at_seq[m_it][ns] = my_corr
        best_model_list = list(best_model_tot)
        accuracy_sum = 0
        best_model_seq_share = []
        for item in range(len(my_choicy_best_seq)):
            if my_choicy_best_seq[item] == best_model_list[item]:
                accuracy_sum += 1
                best_model_seq_share.append(item)
        acc_perf_at_seq[m_it][ns] = accuracy_sum/(n)
        
        #Record Best Strategy Performance of Run for Theory Search
        scores_seq[ns] = np.max(performances)   
        max_perf_forseq[m_it][ns] = scores_seq[ns]
        

        #Shared Proportion of Strategies by Link Type Calculation
        shared = 0
        for item in choices:
            my_item = list(item)
            in_list = 0
            for item2 in choices_sim: 
                my_item2 = list(item2)
                if my_item == my_item2:
                    in_list = 1
            if in_list == 1:
                shared += 1
        shared_bym[m_it][ns] = shared

#Average Across Runs
#Associative model, simultaneous choice
avg_simp_sim = np.zeros((len(m_list),1), dtype=float) #Simplicity of representation
avg_acc_sim = np.zeros((len(m_list),1), dtype=float) #Accuracy of interdependencies
avg_perf_sim = np.zeros((len(m_list),1), dtype=float) #Performance
avg_perf_acc_sim = np.zeros((len(m_list),1), dtype=float) #Max Performance Attribution
avg_difmeans_acc_sim = np.zeros((len(m_list),1), dtype=float) #Average Performance Attribution

#Accuracy of interdependencies, causal model, sequential choice
avg_simp_seq = np.zeros((len(m_list),1), dtype=float) #Simplicity of representation
avg_acc_seq = np.zeros((len(m_list),1), dtype=float) #Accuracy of interdependencies
avg_perf_seq = np.zeros((len(m_list),1), dtype=float) #Performance
avg_perf_acc_seq = np.zeros((len(m_list),1), dtype=float) #Max Performance Attribution
avg_difmeans_acc_seq = np.zeros((len(m_list),1), dtype=float) #Average Performance Attribution

#Proportion of strategies shared and better, simulatneous vs. sequential
avg_my_prop_shared = np.zeros((len(m_list),1), dtype=float) #Proportion of shared strategies


for mer in range(len(m_list)): 
    #Associative, Simultaneous
    avg_simp_sim[mer][0] = np.mean(simp_sim_list[mer])
    avg_acc_sim[mer][0] = np.mean(acc_sim_list[mer])
    avg_perf_sim[mer][0] = np.mean(perf_sim_list[mer])
    avg_perf_acc_sim[mer][0] = np.mean(acc_perf_at_sim[mer])
    avg_difmeans_acc_sim[mer][0] = np.mean(acc_difmeans_at_sim[mer])
    #Theory, Sequential
    avg_simp_seq[mer][0] = np.mean(simp_seq_list[mer])
    avg_acc_seq[mer][0] = np.mean(acc_seq_list[mer])
    avg_perf_seq[mer][0] = np.mean(perf_seq_list[mer])
    avg_perf_acc_seq[mer][0] = np.mean(acc_perf_at_seq[mer])
    avg_difmeans_acc_seq[mer][0] = np.mean(acc_difmeans_at_seq[mer])
    #Proportion Shared
    avg_my_prop_shared[mer][0] = np.mean(shared_bym[mer])


# ### FIGURE 4: Average Proportion of Shared Strategies between Mental Model Types
# 
# Figure 4 below shows the average proportion of shared strategies selected by a decision-maker using the same M strategic choices in an associative mental model vs. theory across 100 times periods and 10,000 simulation runs (for strategic environment, N = 10, K = 5).

# In[ ]:


avg_shared = []
for m in range(len(m_list)): 
    avg_shared.append(np.mean(shared_bym[m])/number_of_periods)
    
my_dict = {"Mental Model Size (M)": m_list, "Avg Prop of Shared Strategies": avg_shared}
my_df = pd.DataFrame(my_dict)

my_title = "Average Proportion of Shared Strategies Between Mental Model Types"

#plot sales of each store as a line
# Set figure size (width, height) in inches
plt.figure(figsize = ( 7 , 5 ))
# Plot lineplot
sns.lineplot(data=my_df, x="Mental Model Size (M)", y="Avg Prop of Shared Strategies")
# Set label for x-axis
plt.xlabel( "Mental Model Size (M)" , size = 12 )

# Set label for y-axis
plt.ylabel("Avg Prop of Shared Strategies", size = 12 )

# Set title for figure
plt.title( my_title , size = 12 )

# Save the plot to a file
plt.savefig('Fig4_AvgProp.png') 


# ### FIGURE 5: Performance by Mental Model Size (M) and Type
# Figure 5 shows the performance of strategies found by decision-makers using associative vs. causal mental model of size (M). (with N = 10 and K = 5).

# In[ ]:


my_perfer = []
my_m = []
my_type = []


for iti in range(len(m_list)):
    M = m_list[iti]
    #Associative Mental Models
    my_type.append("Associative")
    my_perfer.append(np.mean(max_perf_forsim[iti]))
    my_m.append(M)
    
    #Theory
    my_type.append("Theory")
    my_perfer.append(np.mean(max_perf_forseq[iti]))
    my_m.append(M)
   

my_dict = {"Mental Model Size (M)": my_m, "Performance": my_perfer, "Mental Model Type": my_type}
my_df = pd.DataFrame(my_dict)

my_title = "Performance by Mental Model Size (M) and Type"

#plot sales of each store as a line
# Set figure size (width, height) in inches
plt.figure(figsize = ( 7 , 5 ))
# Plot lineplot
sns.lineplot(data=my_df, x="Mental Model Size (M)", y="Performance", hue = "Mental Model Type")
# Set label for x-axis
plt.xlabel( "Mental Model Size (M)" , size = 12 )

# Set label for y-axis
plt.ylabel( "Performance", size = 12 )

# Set title for figure
plt.title( my_title , size = 12 )


# Save the plot to a file
plt.savefig('Fig5_Perf_M_LT.png')


# ### FIGURE 6: Number of Links by Mental Model Size (M) and Type
# Figure 6 visualizes the number of performance dependencies considered in associative mental models and theories across varying sizes of mental models (M) (for N=10, K=5).

# In[ ]:


links = []
typer = []
my_m = []

for M in range(len(m_list)):
    #Associative Mental Models
    typer.append("Associative")
    links.append(avg_simp_sim[M][0])
    my_m.append(str(m_list[M]))
    
    #Theories
    typer.append("Theory")
    links.append(avg_simp_seq[M][0])
    my_m.append(str(m_list[M]))
    
my_dict = {"Mental Model Size (M)": my_m, "Mental Model Type": typer, "Number of Links": links}
my_df = pd.DataFrame(my_dict)

my_title = "Simplicity of Representation by Mental Model Size (M) and Type"

#plot sales of each store as a line
# Set figure size (width, height) in inches
plt.figure(figsize = ( 7 , 5 ))
# Plot lineplot
sns.lineplot(data=my_df, x="Mental Model Size (M)", y="Number of Links", hue = "Mental Model Type")
# Set label for x-axis
plt.xlabel( "Mental Model Size (M)" , size = 12 )

# Set label for y-axis
plt.ylabel(  "Number of Links", size = 12 )

# Set title for figure
plt.title( my_title , size = 12 )

# Save the plot to a file
plt.savefig('Fig6_Simp.png') 


# ### FIGURE 10: Maximum Perfomance Attribution by Mental Model Size (M) and Type
# Figure 10 visualizes the maximum performance attribution based on strategic search using associative mental models and theories across varying sizes of mental models (M) (for N=10, K=5). 

# In[ ]:


max_perf_attr = []
my_m = []
my_type = []


for iti in range(len(m_list)):
    M = m_list[iti]
    
    #Associative Mental Models
    my_type.append("Associative")
    my_m.append(str(M))
    max_perf_attr.append(avg_perf_acc_sim[iti][0])
    
    #Theories
    my_type.append("Theory")
    my_m.append(str(M))
    max_perf_attr.append(avg_perf_acc_seq[iti][0])
    
           

my_dict = {"Mental Model Size (M)": my_m, "Mental Model Type": my_type, "Max Performance Attribution": max_perf_attr}
my_df = pd.DataFrame(my_dict)

my_title = "Max Perfomance Attribution by Mental Model Size (M) and Type"

#plot sales of each store as a line
# Set figure size (width, height) in inches
plt.figure(figsize = ( 7 , 5 ))
# Plot lineplot
sns.lineplot(data=my_df, x="Mental Model Size (M)", y="Max Performance Attribution", hue = "Mental Model Type")
# Set label for x-axis
plt.xlabel( "Mental Model Size (M)" , size = 12 )

# Set label for y-axis
plt.ylabel( "Max Performance Attribution", size = 12 )

# Set title for figure
plt.title( my_title , size = 12 )


# Save the plot to a file
plt.savefig('Fig10_MaxPerfA.png')


# ### FIGURE 11: Average Perfomance Attribution by Mental Model Size (M) and Type
# Figure 11 visualizes the average performance attribution based on strategic search using associative mental models and theories across varying sizes of mental models (M) (for N=10, K=5).   

# In[ ]:


avg_perf_attr = []
my_m = []
my_type = []


for iti in range(len(m_list)):
    M = m_list[iti]
    
    #Associative Mental Models
    my_type.append("Associative")
    my_m.append(str(M))
    avg_perf_attr.append(avg_difmeans_acc_sim[iti][0])
    
    #Theories
    my_type.append("Theory")
    my_m.append(str(M))
    avg_perf_attr.append(avg_difmeans_acc_seq[iti][0])

my_dict = {"Mental Model Size (M)": my_m, "Mental Model Type": my_type, "Avg Performance Attribution": avg_perf_attr}
my_df = pd.DataFrame(my_dict)

my_title = "Average Perfomance Attribution by Mental Model Size (M) and Type"

#plot sales of each store as a line
# Set figure size (width, height) in inches
plt.figure(figsize = ( 7 , 5 ))
# Plot lineplot
sns.lineplot(data=my_df, x="Mental Model Size (M)", y="Avg Performance Attribution", hue = "Mental Model Type")
# Set label for x-axis
plt.xlabel( "Mental Model Size (M)" , size = 12 )

# Set label for y-axis
plt.ylabel( "Avg Performance Attribution", size = 12 )

# Set title for figure
plt.title( my_title , size = 12 )


# Save the plot to a file
plt.savefig('Fig11_AvgPerfA.png')


# ## Section 2: NK-Landscape, Varying N
# Results of mental models with cause-and-effect vs. associative performance links searching across an NK-landscape where K=5 and N varies. Parameters are set to T = 100, S = 1,000. This includes the results for Figure 7a, as well as Appendix results B1.

# In[ ]:


#Set Parameters to T = 100, S = 1,000
number_of_periods  = 100 #Number of time periods to search
number_of_simulations = 1000 # Number of simulations/iterations 


# In[ ]:


#Set Seed
my_rand = 42
np.random.seed(my_rand)

#SET K to 5
k = 5 # K in NK
#List of Possible Ns
n_list = list(range(10,15))
#List of Possible Simultaneous Representation Sizes
m_list = list(range(3,10))

#Generate Empty Items to Store Results
#Associative model, simultaneous choice
avg_simp_sim = np.zeros((len(m_list),len(n_list)), dtype=float) #Simplicity of representation
avg_acc_sim = np.zeros((len(m_list),len(n_list)), dtype=float) #Accuracy of interdependencies
avg_perf_sim = np.zeros((len(m_list),len(n_list)), dtype=float) # Average Performance
avg_max_perf_forsim = np.zeros((len(m_list),len(n_list)), dtype=float)  #Maximum Performance
avg_perf_acc_sim = np.zeros((len(m_list),len(n_list)), dtype=float) #Max Performance Attribution
avg_difmeans_acc_sim = np.zeros((len(m_list),len(n_list)), dtype=float) #Average Performance Attribution

#Accuracy of interdependencies, causal model, sequential choice
avg_simp_seq = np.zeros((len(m_list),len(n_list)), dtype=float) #Simplicity of representation
avg_acc_seq = np.zeros((len(m_list),len(n_list)), dtype=float) #Accuracy of interdependencies
avg_perf_seq = np.zeros((len(m_list),len(n_list)), dtype=float) #Average Performance
avg_max_perf_forseq = np.zeros((len(m_list),len(n_list)), dtype=float) #Maximum Performance
avg_perf_acc_seq = np.zeros((len(m_list),len(n_list)), dtype=float) #Max Performance Attribution
avg_difmeans_acc_seq = np.zeros((len(m_list),len(n_list)), dtype=float) #Average Performance Attribution


#For each different N sized strategic environment
for n_it in range(len(n_list)):
    n = n_list[n_it]

    #Associative model, simultaneous choice
    simp_sim_list = np.zeros((len(m_list),number_of_simulations), dtype=float) #Simplicity of representation
    acc_sim_list = np.zeros((len(m_list),number_of_simulations), dtype=float) #Accuracy of interdependencies
    perf_sim_list = np.zeros((len(m_list),number_of_simulations), dtype=float) #Average Performance
    max_perf_forsim = np.zeros((len(m_list),number_of_simulations), dtype=float) #Maximum Performance
    acc_perf_at_sim = np.zeros((len(m_list),number_of_simulations), dtype=float) #Max Performance Attribution
    acc_difmeans_at_sim = np.zeros((len(m_list),number_of_simulations), dtype=float) #Average Performance Attribution

    #Accuracy of interdependencies, causal model, sequential choice
    simp_seq_list = np.zeros((len(m_list),number_of_simulations), dtype=float) #Simplicity of representation
    acc_seq_list = np.zeros((len(m_list),number_of_simulations), dtype=float) #Accuracy of interdependencies
    perf_seq_list = np.zeros((len(m_list),number_of_simulations), dtype=float) #Average Performance
    max_perf_forseq = np.zeros((len(m_list),number_of_simulations), dtype=float) #Maximum Performance
    acc_perf_at_seq = np.zeros((len(m_list),number_of_simulations), dtype=float) #Max Performance Attribution
    acc_difmeans_at_seq = np.zeros((len(m_list),number_of_simulations), dtype=float) #Average Performance Attribution

    #Overall Performance
    scores_sim = np.zeros(number_of_simulations, dtype=float) 
    scores_seq = np.zeros(number_of_simulations, dtype=float) 

    #For Number of simulations
    for ns in range(0, number_of_simulations): # repeated over the number of simulations
        print(ns,n)
        
        ## STEP 1: INITIALIZE NK-LANDSCAPE WITH N AND K.
        # Here select fitness values and dependencies
        fitness_values = np.random.rand(n, 2**(k+1))
        dependencies = depend_evenk(n, k)
        
        #Find pairs of dependencies for accuracy calculations
        my_pair_int = []
        for item in range(len(dependencies)):
            my_dep = dependencies[item]
            for ant in my_dep:
                my_pair_int.append((ant, item))
        #Pair not in dependencies
        all_pos_pairs = list(itertools.product(list(range(0,n)), repeat=2))
        non_deps = []
        for item in all_pos_pairs: 
            if item not in my_pair_int and item[0] != item[1]:
                non_deps.append(item)
        
        #Calculate actual average performance in environment by strategic choice (for average performance attribution)
        #And identify global maximum strategy (for maximum performance attribution)
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
        dif_means_perf_actual = [] 
        for my_nery in range(n):
            my_0s_perf = []
            my_1s_perf = []
            for itr in range(len(pos_config)):
                if pos_config[itr][my_nery] == 0:
                    my_0s_perf.append(outputs[itr])
                else:
                    my_1s_perf.append(outputs[itr])
            x_0 = np.mean(my_0s_perf)
            x_1 = np.mean(my_1s_perf)
            dif_means_perf_actual.append(x_0 - x_1)


        for m_it in range(len(m_list)):
            ## STEP 2: AT TIME T=0, RANDOMLY SELECT M STRATEGIC CHOICES FOR THE MENTAL MODEL
            ## AN ORDER (OR) FOR THE THEORY AND AN INITIAL RANDOM STRATEGY.
            m = m_list[m_it]
            my_rep = random.sample(list(range(n)), m)
            not_in_rep = list(set(list(range(n))) - set(list(my_rep)))
            my_choice_config_init = list(np.random.randint(0, 2, n))
            
            #ASSOCIAITIVE MENTAL MODEL, SIMULTANEOUS CHOICE
            choices_sim = np.zeros((number_of_periods,n), dtype=int)
            performances_sim = np.zeros(number_of_periods, dtype=float)
            
            
            #Calculate dependencies for accuracy and simplicity of mental model
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
            accuracy = (TP + TN) / (n*(n-1))
            acc_sim_list[m_it][ns] = accuracy
            simp_sim_list[m_it][ns] = len(my_m_deps_int)
            
            for t in range(1, number_of_periods):
                #Initial Random Choice
                if t == 1:
                    choice_configuration = np.array(my_choice_config_init)
                    bestch = choice_configuration
                
                ## STEP 3 SIMULTANEOUS: FOR EACH SUBSEQUENT T USE THE ASSOCIATIVE MENTAL MODEL TO SELECT A NEW STRATEGY 
                else:
                    ## STEP 3.1 START WITH THE BEST PRIOR PERFORMING STRATEGY.
                    my_choices = choices_sim.tolist()[:t-1]
                    perfy = performances_sim.tolist()[:t-1]
                    max_index = np.argmax(performances_sim) 
                    bestch = choices_sim[max_index,:] 
                    my_choice_config = list(bestch)
                    
                    ## STEP 3.2 SELECT A RANDOM STRATEGIC CHOICE A TO CHANGE FROM BEST.
                    my_change = random.choice(my_rep)
                    rep_index = my_rep.index(my_change)
                    my_choice_config[my_change] = 1- my_choice_config[my_change]
                    
                    ## STEP 3.3 FOR MENTAL MODEL RUNS CALCULATE BEST AVERAGE PERFORMING STRATEGY
                    ## CONDITIONAL ON THE VALUE OF CHOICE A SELECTED IN 3.2 FOR ALL M CHOICES. 
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
                    
                    ## STEP 3.5: FOR ALL N-M STRATEGIC CHOICES, SELECT BEST CHOICE BASED ON
                    ## INDEPENDENTLY CONSIDERED PERFORMANCE.
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
                    
                    ## STEP 4: IF STRATEGY IN STEP 3 HAS ALREADY BEEN TRIED
                    ## SELECT A RANDOM STRATEGIC CHOICE IN N TO CHANGE. 
                    choice_configuration = np.array(my_choice_config)
                    if any((choices_sim[:]==choice_configuration).all(1)): 
                        ch = random.choice(list(range(n)))
                        choice_configuration[ch] = 1- choice_configuration[ch]
                
                #Record Strategy and Normalized Performance at T
                current_performance=perffunc_evenk(choice_configuration, k, n, dependencies, fitness_values, t)
                choices_sim[t-1] = choice_configuration 
                performances_sim[t-1] = (current_performance - min_value_tot) / (max_value_tot - min_value_tot)
            
            #Calculate Maximum and Average Performance Attribution for Associative Mental Model
            my_choices = choices_sim.tolist()[:t-1]
            perfy = performances_sim.tolist()[:t-1]
            my_choicy_best_sim = []
            dif_means_perf_sim = []
            for my_nery in range(n):
                my_0s_perf = []
                my_1s_perf = []
                for itr in range(len(my_choices)):
                    if my_choices[itr][my_nery] == 0:
                        my_0s_perf.append(perfy[itr])
                    else:
                        my_1s_perf.append(perfy[itr])
                x_0 = np.mean(my_0s_perf)
                x_1 = np.mean(my_1s_perf)
                dif_means_perf_sim.append(x_0-x_1)
                if math.isnan(x_0):
                    my_choicy_best_sim.append(1)
                elif math.isnan(x_1):
                    my_choicy_best_sim.append(0)
                elif x_0 > x_1:
                    my_choicy_best_sim.append(0)
                elif x_1 > x_0:
                    my_choicy_best_sim.append(1)
                else:
                    my_choicy_best_sim.append(1)
            my_differ = []
            cor_dif = np.corrcoef(dif_means_perf_actual,dif_means_perf_sim)
            my_corr = cor_dif[0][1]
            acc_difmeans_at_sim[m_it][ns] = my_corr
            best_model_list = list(best_model_tot)
            accuracy_sum = 0
            best_model_sim_share = []
            for item in range(len(my_choicy_best_sim)):
                if my_choicy_best_sim[item] == best_model_list[item]:
                    accuracy_sum += 1
                    best_model_sim_share.append(item)
            acc_perf_at_sim[m_it][ns] = accuracy_sum/(n)
            
            #Record Best Strategy Performance of Run for Associative Mental Model Search
            scores_sim[ns] = np.max(performances_sim)
            max_perf_forsim[m_it][ns] = scores_sim[ns]
            
            
            #THEORY SEARCH, SEQUENTIAL CHOICE
            choices = np.zeros((number_of_periods,n), dtype=int)
            performances = np.zeros(number_of_periods, dtype=float)
            
            #Find dependencies for accuracy and simplicity of theory
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
            accuracy = (TP + TN) / (n*(n-1))
            acc_seq_list[m_it][ns] = accuracy
            simp_seq_list[m_it][ns] = len(my_deps)
            
            
            for t in range(1, number_of_periods):
                #Initial Random Choice
                if t == 1:
                    choice_configuration = np.array(my_choice_config_init)
                    bestch = choice_configuration
                ## STEP 3 SIMULTANEOUS: FOR EACH SUBSEQUENT T USE THE THEORY TO SELECT A NEW STRATEGY 
                else:
                    ## STEP 3.1 START WITH THE BEST PRIOR PERFORMING STRATEGY
                    my_choices = choices.tolist()[:t-1]
                    perfy = performances.tolist()[:t-1]
                    max_index = np.argmax(performances) 
                    bestch = choices[max_index,:] 
                    my_choice_config = list(bestch)
                    
                    ## STEP 3.2 SELECT A RANDOM STRATEGIC CHOICE A TO CHANGE FROM BEST.
                    my_rando = random.choice(my_rep)
                    my_index = my_rep.index(my_rando)
                    
                    ## STEP 3.4 FOR THEORY RUNS CALCULATE THE BEST AVERAGE PERFORMING STRATEGY 
                    ## FOR PERFORMANCE CONSEQUENTS OF A ONLY CONDITIONAL ON THE VALUE OF CHOICE A
                    ## AND ITS ANTECEDENTS SELECTED IN 2. USING ORDER (OR).
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
                                    my_dat_x.append(my_choicy[chc][my_filter])
                                    my_dat_y.append(perfyty[chc])
                    val_to_filt = my_choice_config[to_change[0]]
                    if len(my_dat_x) > 0 or len(to_consider) == 0:
                        for item in range(len(to_change)-1):
                            my_dat = {} 
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
                    
                    ## STEP 3.5: FOR ALL N-M STRATEGIC CHOICES, SELECT BEST CHOICE BASED ON
                    ## INDEPENDENTLY CONSIDERED PERFORMANCE.
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
                
                ## STEP 4: IF STRATEGY IN STEP 3 HAS ALREADY BEEN TRIED
                ## SELECT A RANDOM STRATEGIC CHOICE IN N TO CHANGE. 
                choice_configuration = np.array(my_choice_config)
                if any((choices[:]==choice_configuration).all(1)):
                    ch = random.choice(list(range(n)))
                    choice_configuration[ch] = 1- choice_configuration[ch]
                
                #Record Strategy and Normalized Performance at T
                current_performance=perffunc_evenk(choice_configuration, k, n, dependencies, fitness_values, t)
                choices[t-1] = choice_configuration
                performances[t-1] = (current_performance - min_value_tot) / (max_value_tot - min_value_tot)
            
            #Calculate Maximum and Average Performance Attribution for Theory
            my_choices = choices.tolist()[:t-1]
            perfy = performances.tolist()[:t-1]
            my_choicy_best_seq = []
            dif_means_perf_seq = [] 
            for my_nery in range(n):
                my_0s_perf = []
                my_1s_perf = []
                for itr in range(len(my_choices)):
                    if my_choices[itr][my_nery] == 0:
                        my_0s_perf.append(perfy[itr])
                    else:
                        my_1s_perf.append(perfy[itr])
                x_0 = np.mean(my_0s_perf)
                x_1 = np.mean(my_1s_perf)
                dif_means_perf_seq.append(x_0 - x_1)
                if math.isnan(x_0):
                    my_choicy_best_seq.append(1)
                elif math.isnan(x_1):
                    my_choicy_best_seq.append(0)
                elif x_0 > x_1:
                    my_choicy_best_seq.append(0)
                elif x_1 > x_0:
                    my_choicy_best_seq.append(1)
                else:
                    my_choicy_best_seq.append(1)
            my_differ = []
            cor_dif = np.corrcoef(dif_means_perf_actual,dif_means_perf_seq)
            my_corr = cor_dif[0][1]
            acc_difmeans_at_seq[m_it][ns] = my_corr
            best_model_list = list(best_model_tot)
            accuracy_sum = 0
            best_model_seq_share = []
            for item in range(len(my_choicy_best_seq)):
                if my_choicy_best_seq[item] == best_model_list[item]:
                    accuracy_sum += 1
                    best_model_seq_share.append(item)
            acc_perf_at_seq[m_it][ns] = accuracy_sum/(n)

            #Record Best Strategy Performance of Run for Theory Search
            scores_seq[ns] = np.max(performances)
            max_perf_forseq[m_it][ns] = scores_seq[ns]
                
    #Average Across Runs
    for mer in range(len(m_list)): 
        #Associative, Simultaneous
        avg_acc_sim[mer][n_it] = np.mean(acc_sim_list[mer])
        avg_perf_sim[mer][n_it] = np.mean(perf_sim_list[mer])
        avg_max_perf_forsim[mer][n_it] = np.mean(max_perf_forsim[mer])
        avg_perf_acc_sim[mer][n_it] = np.mean(acc_perf_at_sim[mer])
        avg_difmeans_acc_sim[mer][n_it] = np.mean(acc_difmeans_at_sim[mer])

        #Theory, Sequential
        avg_acc_seq[mer][n_it] = np.mean(acc_seq_list[mer])
        avg_perf_seq[mer][n_it] = np.mean(perf_seq_list[mer])
        avg_max_perf_forseq[mer][n_it] = np.mean(max_perf_forseq[mer])
        avg_perf_acc_seq[mer][n_it] = np.mean(acc_perf_at_seq[mer])
        avg_difmeans_acc_seq[mer][n_it] = np.mean(acc_difmeans_at_seq[mer])

        


# ### FIGURE 7a: Accuracy of Links by Mental Model Size (M) and Type Across Number of Strategic Choices (N)
# Figure 7a compares model link accuracy as the number of strategic choices (N) increases (with K = 5).

# In[ ]:


acc_bym_n = []
my_m = []
my_type = []
my_n = []


for iti in range(len(m_list)):
    M = m_list[iti]
    for ner in range(len(n_list)):
        
        #Associative Mental Model
        my_type.append("Associative")
        my_n.append(n_list[ner])
        my_m.append(str(M))
        acc_bym_n.append(avg_acc_sim[iti][ner])
        
        #Theory
        my_type.append("Theory")
        my_n.append(n_list[ner])
        my_m.append(str(M))
        acc_bym_n.append(avg_acc_seq[iti][ner])
   

my_dict = {"Number of Choices (N)":my_n, "Mental Model Size (M)": my_m, "Mental Model Type": my_type, "Accuracy of Links": acc_bym_n}
my_df = pd.DataFrame(my_dict)

my_title = "Accuracy of Links by Mental Model Size (M) and Type Across Number of Strategic Choices (N)"

#
#plot sales of each store as a line
# Set figure size (width, height) in inches
plt.figure(figsize = ( 7 , 5 ))
g = sns.FacetGrid(my_df, col="Number of Choices (N)", hue = "Mental Model Type", col_wrap=5)
g.map(sns.lineplot, "Mental Model Size (M)", "Accuracy of Links")
g.add_legend()


# Save the plot to a file
g.savefig('Fig7a_Acc_N.png')


# ### FIGURE B1:  Performance by Mental Model Size (M) and Type Across Number of Strategic Choices (N)
# Figure B1 shows the performance of strategies found by decision-makers using associative vs. causal mental model of size (M) across N (with K = 5).
# 

# In[ ]:


perf_bym_n = []
my_m = []
my_type = []
my_n = []


for iti in range(len(m_list)):
    M = m_list[iti]
    for ner in range(len(n_list)):
        
        #Associative Mental Model
        my_type.append("Associative")
        my_n.append(n_list[ner])
        my_m.append(str(M))
        perf_bym_n.append(avg_max_perf_forsim[iti][ner])
        
        #Theory
        my_type.append("Theory")
        my_n.append(n_list[ner])
        my_m.append(str(M))
        perf_bym_n.append(avg_max_perf_forseq[iti][ner])
   

my_dict = {"Number of Choices (N)":my_n, "Mental Model Size (M)": my_m, "Mental Model Type": my_type, "Performance": perf_bym_n}
my_df = pd.DataFrame(my_dict)

my_title = "Performance by Mental Model Size (M) and Type Across Number of Strategic Choices (N)"

#
#plot sales of each store as a line
# Set figure size (width, height) in inches
plt.figure(figsize = ( 7 , 5 ))
g = sns.FacetGrid(my_df, col="Number of Choices (N)", hue = "Mental Model Type", col_wrap=5)
g.map(sns.lineplot, "Mental Model Size (M)", "Performance")
g.add_legend()


# Save the plot to a file
g.savefig('FigB1_Perf_N.png')


# ## Section 3: NK-Landscape, Varying K
# Results of mental models with cause-and-effect vs. associative performance links searching across an NK-landscape where N=10 and K varies. Parameters are set to T = 100, S = 1,000. This includes the results for Figure 7b, as well as Appendix results B2.

# In[ ]:


#Set Parameters to T = 100, S = 1,000
number_of_periods  = 100 #Number of time periods to search
number_of_simulations = 1000 # Number of simulations/iterations 


# In[ ]:


#Set Seed
my_rand = 42
np.random.seed(my_rand)

#Set N to 10
n = 10
#List of possible Ks
k_list = list(range(3,8))

#List of Possible Simultaneous Representation Sizes
m_list = list(range(3,10))

#Generate Empty Items to Store Results
#Associative model, simultaneous choice
avg_simp_sim = np.zeros((len(m_list),len(k_list)), dtype=float) #Simplicity of representation
avg_acc_sim = np.zeros((len(m_list),len(k_list)), dtype=float) #Accuracy of interdependencies
avg_perf_sim = np.zeros((len(m_list),len(k_list)), dtype=float) # Average Performance
avg_max_perf_forsim = np.zeros((len(m_list),len(k_list)), dtype=float)  #Maximum Performance
avg_perf_acc_sim = np.zeros((len(m_list),len(k_list)), dtype=float) #Max Performance Attribution
avg_difmeans_acc_sim = np.zeros((len(m_list),len(k_list)), dtype=float) #Average Performance Attribution

#Accuracy of interdependencies, causal model, sequential choice
avg_simp_seq = np.zeros((len(m_list),len(k_list)), dtype=float) #Simplicity of representation
avg_acc_seq = np.zeros((len(m_list),len(k_list)), dtype=float) #Accuracy of interdependencies
avg_perf_seq = np.zeros((len(m_list),len(k_list)), dtype=float) #Average Performance
avg_max_perf_forseq = np.zeros((len(m_list),len(k_list)), dtype=float) #Maximum Performance
avg_perf_acc_seq = np.zeros((len(m_list),len(k_list)), dtype=float) #Max Performance Attribution
avg_difmeans_acc_seq = np.zeros((len(m_list),len(k_list)), dtype=float) #Average Performance Attribution


#For each different K interdependent strategic environment
for k_it in range(len(k_list)):
    k = k_list[k_it]

    #Associative model, simultaneous choice
    simp_sim_list = np.zeros((len(m_list),number_of_simulations), dtype=float) #Simplicity of representation
    acc_sim_list = np.zeros((len(m_list),number_of_simulations), dtype=float) #Accuracy of interdependencies
    perf_sim_list = np.zeros((len(m_list),number_of_simulations), dtype=float) #Average Performance
    max_perf_forsim = np.zeros((len(m_list),number_of_simulations), dtype=float) #Maximum Performance
    acc_perf_at_sim = np.zeros((len(m_list),number_of_simulations), dtype=float) #Max Performance Attribution
    acc_difmeans_at_sim = np.zeros((len(m_list),number_of_simulations), dtype=float) #Average Performance Attribution

    #Accuracy of interdependencies, causal model, sequential choice
    simp_seq_list = np.zeros((len(m_list),number_of_simulations), dtype=float) #Simplicity of representation
    acc_seq_list = np.zeros((len(m_list),number_of_simulations), dtype=float) #Accuracy of interdependencies
    perf_seq_list = np.zeros((len(m_list),number_of_simulations), dtype=float) #Average Performance
    max_perf_forseq = np.zeros((len(m_list),number_of_simulations), dtype=float) #Maximum Performance
    acc_perf_at_seq = np.zeros((len(m_list),number_of_simulations), dtype=float) #Max Performance Attribution
    acc_difmeans_at_seq = np.zeros((len(m_list),number_of_simulations), dtype=float) #Average Performance Attribution

    #Overall Performance
    scores_sim = np.zeros(number_of_simulations, dtype=float) 
    scores_seq = np.zeros(number_of_simulations, dtype=float) 

    #For Number of simulations
    for ns in range(0, number_of_simulations): # repeated over the number of simulations
        print(ns,k)
        
        ## STEP 1: INITIALIZE NK-LANDSCAPE WITH N AND K.
        # Here select fitness values and dependencies
        fitness_values = np.random.rand(n, 2**(k+1))
        dependencies = depend_evenk(n, k)
        
        #Find pairs of dependencies for accuracy calculations
        my_pair_int = []
        for item in range(len(dependencies)):
            my_dep = dependencies[item]
            for ant in my_dep:
                my_pair_int.append((ant, item))
        #Pair not in dependencies
        all_pos_pairs = list(itertools.product(list(range(0,n)), repeat=2))
        non_deps = []
        for item in all_pos_pairs: 
            if item not in my_pair_int and item[0] != item[1]:
                non_deps.append(item)
        
        #Calculate actual average performance in environment by strategic choice (for average performance attribution)
        #And identify global maximum strategy (for maximum performance attribution)
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
        dif_means_perf_actual = [] 
        for my_nery in range(n):
            my_0s_perf = []
            my_1s_perf = []
            for itr in range(len(pos_config)):
                if pos_config[itr][my_nery] == 0:
                    my_0s_perf.append(outputs[itr])
                else:
                    my_1s_perf.append(outputs[itr])
            x_0 = np.mean(my_0s_perf)
            x_1 = np.mean(my_1s_perf)
            dif_means_perf_actual.append(x_0 - x_1)


        for m_it in range(len(m_list)):
            ## STEP 2: AT TIME T=0, RANDOMLY SELECT M STRATEGIC CHOICES FOR THE MENTAL MODEL
            ## AN ORDER (OR) FOR THE THEORY AND AN INITIAL RANDOM STRATEGY.
            m = m_list[m_it]
            my_rep = random.sample(list(range(n)), m)
            not_in_rep = list(set(list(range(n))) - set(list(my_rep)))
            my_choice_config_init = list(np.random.randint(0, 2, n))
            
            #ASSOCIAITIVE MENTAL MODEL, SIMULTANEOUS CHOICE
            choices_sim = np.zeros((number_of_periods,n), dtype=int)
            performances_sim = np.zeros(number_of_periods, dtype=float)
            
            
            #Calculate dependencies for accuracy and simplicity of mental model
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
            accuracy = (TP + TN) / (n*(n-1))
            acc_sim_list[m_it][ns] = accuracy
            simp_sim_list[m_it][ns] = len(my_m_deps_int)
            
            for t in range(1, number_of_periods):
                #Initial Random Choice
                if t == 1:
                    choice_configuration = np.array(my_choice_config_init)
                    bestch = choice_configuration
                
                ## STEP 3 SIMULTANEOUS: FOR EACH SUBSEQUENT T USE THE ASSOCIATIVE MENTAL MODEL TO SELECT A NEW STRATEGY 
                else:
                    ## STEP 3.1 START WITH THE BEST PRIOR PERFORMING STRATEGY.
                    my_choices = choices_sim.tolist()[:t-1]
                    perfy = performances_sim.tolist()[:t-1]
                    max_index = np.argmax(performances_sim) 
                    bestch = choices_sim[max_index,:] 
                    my_choice_config = list(bestch)
                    
                    ## STEP 3.2 SELECT A RANDOM STRATEGIC CHOICE A TO CHANGE FROM BEST.
                    my_change = random.choice(my_rep)
                    rep_index = my_rep.index(my_change)
                    my_choice_config[my_change] = 1- my_choice_config[my_change]
                    
                    ## STEP 3.3 FOR MENTAL MODEL RUNS CALCULATE BEST AVERAGE PERFORMING STRATEGY
                    ## CONDITIONAL ON THE VALUE OF CHOICE A SELECTED IN 3.2 FOR ALL M CHOICES. 
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
                    
                    ## STEP 3.5: FOR ALL N-M STRATEGIC CHOICES, SELECT BEST CHOICE BASED ON
                    ## INDEPENDENTLY CONSIDERED PERFORMANCE.
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
                    
                    ## STEP 4: IF STRATEGY IN STEP 3 HAS ALREADY BEEN TRIED
                    ## SELECT A RANDOM STRATEGIC CHOICE IN N TO CHANGE. 
                    choice_configuration = np.array(my_choice_config)
                    if any((choices_sim[:]==choice_configuration).all(1)): 
                        ch = random.choice(list(range(n)))
                        choice_configuration[ch] = 1- choice_configuration[ch]
                
                #Record Strategy and Normalized Performance at T
                current_performance=perffunc_evenk(choice_configuration, k, n, dependencies, fitness_values, t)
                choices_sim[t-1] = choice_configuration 
                performances_sim[t-1] = (current_performance - min_value_tot) / (max_value_tot - min_value_tot)
            
            #Calculate Maximum and Average Performance Attribution for Associative Mental Model
            my_choices = choices_sim.tolist()[:t-1]
            perfy = performances_sim.tolist()[:t-1]
            my_choicy_best_sim = []
            dif_means_perf_sim = []
            for my_nery in range(n):
                my_0s_perf = []
                my_1s_perf = []
                for itr in range(len(my_choices)):
                    if my_choices[itr][my_nery] == 0:
                        my_0s_perf.append(perfy[itr])
                    else:
                        my_1s_perf.append(perfy[itr])
                x_0 = np.mean(my_0s_perf)
                x_1 = np.mean(my_1s_perf)
                dif_means_perf_sim.append(x_0-x_1)
                if math.isnan(x_0):
                    my_choicy_best_sim.append(1)
                elif math.isnan(x_1):
                    my_choicy_best_sim.append(0)
                elif x_0 > x_1:
                    my_choicy_best_sim.append(0)
                elif x_1 > x_0:
                    my_choicy_best_sim.append(1)
                else:
                    my_choicy_best_sim.append(1)
            my_differ = []
            cor_dif = np.corrcoef(dif_means_perf_actual,dif_means_perf_sim)
            my_corr = cor_dif[0][1]
            acc_difmeans_at_sim[m_it][ns] = my_corr
            best_model_list = list(best_model_tot)
            accuracy_sum = 0
            best_model_sim_share = []
            for item in range(len(my_choicy_best_sim)):
                if my_choicy_best_sim[item] == best_model_list[item]:
                    accuracy_sum += 1
                    best_model_sim_share.append(item)
            acc_perf_at_sim[m_it][ns] = accuracy_sum/(n)
            
            #Record Best Strategy Performance of Run for Associative Mental Model Search
            scores_sim[ns] = np.max(performances_sim)
            max_perf_forsim[m_it][ns] = scores_sim[ns]
            
            
            #THEORY SEARCH, SEQUENTIAL CHOICE
            choices = np.zeros((number_of_periods,n), dtype=int)
            performances = np.zeros(number_of_periods, dtype=float)
            
            #Find dependencies for accuracy and simplicity of theory
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
            accuracy = (TP + TN) / (n*(n-1))
            acc_seq_list[m_it][ns] = accuracy
            simp_seq_list[m_it][ns] = len(my_deps)
            
            
            for t in range(1, number_of_periods):
                #Initial Random Choice
                if t == 1:
                    choice_configuration = np.array(my_choice_config_init)
                    bestch = choice_configuration
                ## STEP 3 SIMULTANEOUS: FOR EACH SUBSEQUENT T USE THE THEORY TO SELECT A NEW STRATEGY 
                else:
                    ## STEP 3.1 START WITH THE BEST PRIOR PERFORMING STRATEGY
                    my_choices = choices.tolist()[:t-1]
                    perfy = performances.tolist()[:t-1]
                    max_index = np.argmax(performances) 
                    bestch = choices[max_index,:] 
                    my_choice_config = list(bestch)
                    
                    ## STEP 3.2 SELECT A RANDOM STRATEGIC CHOICE A TO CHANGE FROM BEST.
                    my_rando = random.choice(my_rep)
                    my_index = my_rep.index(my_rando)
                    
                    ## STEP 3.4 FOR THEORY RUNS CALCULATE THE BEST AVERAGE PERFORMING STRATEGY 
                    ## FOR PERFORMANCE CONSEQUENTS OF A ONLY CONDITIONAL ON THE VALUE OF CHOICE A
                    ## AND ITS ANTECEDENTS SELECTED IN 2. USING ORDER (OR).
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
                                    my_dat_x.append(my_choicy[chc][my_filter])
                                    my_dat_y.append(perfyty[chc])
                    val_to_filt = my_choice_config[to_change[0]]
                    if len(my_dat_x) > 0 or len(to_consider) == 0:
                        for item in range(len(to_change)-1):
                            my_dat = {} 
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
                    
                    ## STEP 3.5: FOR ALL N-M STRATEGIC CHOICES, SELECT BEST CHOICE BASED ON
                    ## INDEPENDENTLY CONSIDERED PERFORMANCE.
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
                
                ## STEP 4: IF STRATEGY IN STEP 3 HAS ALREADY BEEN TRIED
                ## SELECT A RANDOM STRATEGIC CHOICE IN N TO CHANGE. 
                choice_configuration = np.array(my_choice_config)
                if any((choices[:]==choice_configuration).all(1)):
                    ch = random.choice(list(range(n)))
                    choice_configuration[ch] = 1- choice_configuration[ch]
                
                #Record Strategy and Normalized Performance at T
                current_performance=perffunc_evenk(choice_configuration, k, n, dependencies, fitness_values, t)
                choices[t-1] = choice_configuration
                performances[t-1] = (current_performance - min_value_tot) / (max_value_tot - min_value_tot)
            
            #Calculate Maximum and Average Performance Attribution for Theory
            my_choices = choices.tolist()[:t-1]
            perfy = performances.tolist()[:t-1]
            my_choicy_best_seq = []
            dif_means_perf_seq = [] 
            for my_nery in range(n):
                my_0s_perf = []
                my_1s_perf = []
                for itr in range(len(my_choices)):
                    if my_choices[itr][my_nery] == 0:
                        my_0s_perf.append(perfy[itr])
                    else:
                        my_1s_perf.append(perfy[itr])
                x_0 = np.mean(my_0s_perf)
                x_1 = np.mean(my_1s_perf)
                dif_means_perf_seq.append(x_0 - x_1)
                if math.isnan(x_0):
                    my_choicy_best_seq.append(1)
                elif math.isnan(x_1):
                    my_choicy_best_seq.append(0)
                elif x_0 > x_1:
                    my_choicy_best_seq.append(0)
                elif x_1 > x_0:
                    my_choicy_best_seq.append(1)
                else:
                    my_choicy_best_seq.append(1)
            my_differ = []
            cor_dif = np.corrcoef(dif_means_perf_actual,dif_means_perf_seq)
            my_corr = cor_dif[0][1]
            acc_difmeans_at_seq[m_it][ns] = my_corr
            best_model_list = list(best_model_tot)
            accuracy_sum = 0
            best_model_seq_share = []
            for item in range(len(my_choicy_best_seq)):
                if my_choicy_best_seq[item] == best_model_list[item]:
                    accuracy_sum += 1
                    best_model_seq_share.append(item)
            acc_perf_at_seq[m_it][ns] = accuracy_sum/(n)

            #Record Best Strategy Performance of Run for Theory Search
            scores_seq[ns] = np.max(performances)
            max_perf_forseq[m_it][ns] = scores_seq[ns]

            
                
    #Average Across Runs 
    for mer in range(len(m_list)): 
        #Associative, Simultaneous
        avg_acc_sim[mer][k_it] = np.mean(acc_sim_list[mer])
        avg_perf_sim[mer][k_it] = np.mean(perf_sim_list[mer])
        avg_max_perf_forsim[mer][k_it] = np.mean(max_perf_forsim[mer])
        avg_perf_acc_sim[mer][k_it] = np.mean(acc_perf_at_sim[mer])
        avg_difmeans_acc_sim[mer][k_it] = np.mean(acc_difmeans_at_sim[mer])

        #Theory, Sequential
        avg_acc_seq[mer][k_it] = np.mean(acc_seq_list[mer])
        avg_perf_seq[mer][k_it] = np.mean(perf_seq_list[mer])
        avg_max_perf_forseq[mer][k_it] = np.mean(max_perf_forseq[mer])
        avg_perf_acc_seq[mer][k_it] = np.mean(acc_perf_at_seq[mer])
        avg_difmeans_acc_seq[mer][k_it] = np.mean(acc_difmeans_at_seq[mer])
        


# ### FIGURE 7b: Accuracy of Links by Mental Model Size (M) and Type, Across Interdependence of Strategic Choices (K)
# Figure 7b compares link accuracy as the number of performance links (K) increases (with N = 10).

# In[ ]:


acc_bym_k = []
my_m = []
my_type = []
my_k = []


for iti in range(len(m_list)):
    M = m_list[iti]
    for ker in range(len(k_list)):
        
        #Associative Mental Model
        my_type.append("Associative")
        my_k.append(k_list[ker])
        my_m.append(str(M))
        acc_bym_k.append(avg_acc_sim[iti][ker])
        
        #Theory
        my_type.append("Theory")
        my_k.append(k_list[ker])
        my_m.append(str(M))
        acc_bym_k.append(avg_acc_seq[iti][ker])
   

my_dict = {"Number of Interdependencies (K)":my_k, "Mental Model Size (M)": my_m, "Mental Model Type": my_type, "Accuracy of Links": acc_bym_k}
my_df = pd.DataFrame(my_dict)

my_title = "Accuracy of Links by Mental Model Size (M) and Type Across Number of Interdependencies (K)"

#
#plot sales of each store as a line
# Set figure size (width, height) in inches
plt.figure(figsize = ( 7 , 5 ))
g = sns.FacetGrid(my_df, col="Number of Interdependencies (K)", hue = "Mental Model Type", col_wrap=5)
g.map(sns.lineplot, "Mental Model Size (M)", "Accuracy of Links")
g.add_legend()


# Save the plot to a file
g.savefig('Fig7b_Acc_K.png')


# ### FIGURE B2:  Performance by Mental Model Size (M) and Type Across Interdependence of Strategic Choices (K)
# Figure B2 shows the performance of strategies found by decision-makers using associative vs. causal mental model of size (M) across K (with N = 10).  

# In[ ]:


perf_bym_k = []
my_m = []
my_type = []
my_k = []


for iti in range(len(m_list)):
    M = m_list[iti]
    for ker in range(len(k_list)):
        
        #Associative Mental Model
        my_type.append("Associative")
        my_k.append(k_list[ker])
        my_m.append(str(M))
        perf_bym_k.append(avg_max_perf_forsim[iti][ker])
        
        #Theory
        my_type.append("Theory")
        my_k.append(k_list[ker])
        my_m.append(str(M))
        perf_bym_k.append(avg_max_perf_forseq[iti][ker])
   

my_dict = {"Number of Interdependencies (K)":my_k, "Mental Model Size (M)": my_m, "Mental Model Type": my_type, "Performance": perf_bym_k}
my_df = pd.DataFrame(my_dict)

my_title = "Performance by Mental Model Size (M) and Type Across Number of Interdependencies (K)"

#
#plot sales of each store as a line
# Set figure size (width, height) in inches
plt.figure(figsize = ( 7 , 5 ))
g = sns.FacetGrid(my_df, col="Number of Interdependencies (K)", hue = "Mental Model Type", col_wrap=5)
g.map(sns.lineplot, "Mental Model Size (M)", "Performance")
g.add_legend()


# Save the plot to a file
g.savefig('FigB2_Perf_K.png')

