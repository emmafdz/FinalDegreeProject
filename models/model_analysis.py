
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from PIL import Image
from sklearn.preprocessing import normalize
import math
from scipy.stats import norm


def accuracy(decision_value_list,mu):
    error_list = []
    for decision_value in decision_value_list:
        right_decision =  decision_value[len(decision_value)-1] >= 0
        right_mu = mu >= 0
        left_decision =  decision_value[len(decision_value)-1] < 0
        left_mu = mu < 0
        if right_decision == right_mu or left_decision == left_mu:
            error_list.append(0)
        else:
            error_list.append(1)
    per_error = 1-(sum(error_list)/len(error_list))
    return per_error

def prob_right(decision_value_list,mu):
    error_list = []
    for decision_value in decision_value_list:
        right_decision =  decision_value[len(decision_value)-1] >= 0
#         right_mu = mu >= 0
        if right_decision == True:
            error_list.append(0)
        else:
            error_list.append(1)
    per_error = 1-(sum(error_list)/len(error_list))
    return per_error

def kernel(decision_value_list, stimulus_list):
    binary_decision = []
    for decision_value in decision_value_list:
        if decision_value[len(decision_value)-1] >= 0:
            binary_decision.append(1)
        else:
            binary_decision.append(0)
    logit_mod = sm.Logit(binary_decision, stimulus_list)
    result = logit_mod.fit()
    pars =result.params
    return pars

def FM(L,mu,dt,tau,alfa,sigma_i,sigma_s):
    forward_matrix = np.zeros((int(500*(L+L)),int(500*(L+L))))
    y = 0
    for x in np.arange(-L,L,0.002): 
        # The mean of the distribution will be computed
        mean = x-(dt/tau)*(-mu-alfa*x*2+4*x**3)
        # The standard deviation of the distribution will be computed
        stdr_dev = np.sqrt(dt/tau)*np.sqrt(sigma_i**2+sigma_s**2)

        limits = np.linspace(-L,L,1000)
        cumulative = norm.cdf(limits, loc = mean, scale = stdr_dev)
        #plt.plot(np.arange(0,len(cumulative),1),cumulative)
        d = np.diff(cumulative)
        #plt.plot(np.arange(0,len(d),1),d)
        counter = 0
        for element in d:
            forward_matrix[counter][y]= element
            counter += 1
        if y <10:
            infinity = 1-sum(d)
            forward_matrix[0][y]+=infinity
        if y >= 10:
            infinity = 1-sum(d)
            forward_matrix[999][y]+= infinity
        y += 1 
    return(forward_matrix)


def FM1(L,mu,dt,tau,alfa,sigma_i,sigma_s):
    forward_matrix = np.zeros((int(1000*(L+L)),int(1000*(L+L))))
    y = 0
    for x in np.arange(-L,L,0.001): 
        # The mean of the distribution will be computed
        mean = x-(dt/tau)*(-mu-alfa*x*2+4*x**3)
        # The standard deviation of the distribution will be computed
        stdr_dev = np.sqrt(dt/tau)*np.sqrt(sigma_i**2+sigma_s**2)

        limits = np.linspace(-L,L,2000)
        cumulative = norm.cdf(limits, loc = mean, scale = stdr_dev)
        #plt.plot(np.arange(0,len(cumulative),1),cumulative)
        d = np.diff(cumulative)
        #plt.plot(np.arange(0,len(d),1),d)
        counter = 0
        for element in d:
            forward_matrix[counter][y]= element
            counter += 1
        if y <10:
            infinity = 1-sum(d)
            forward_matrix[0][y]+=infinity
        if y >= 10:
            infinity = 1-sum(d)
            forward_matrix[1999][y]+= infinity
        y += 1 
    return(forward_matrix)