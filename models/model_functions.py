""" In this file, the functions which generate the double well and drift diffusion model can be found.
      """
# Imprting the libraries needed:
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from PIL import Image
from sklearn.preprocessing import normalize
import math
from scipy.stats import norm


def perfect_integrator(x,stimulus,dt_in_frame,dt,tau,mu,sigma_i,sigma_s):
    # The starting decision value will be 0:
    decision_value = [x]
    # The process will be repeated a two hundred times (1 second stimulus)
    for xi_s in stimulus:
        for i in range(int(dt_in_frame)):
            xi_i = np.random.randn(1)
            # Solved differential diffusion equation:
            x = x - (dt/tau)*(-mu)+np.sqrt(dt/tau)*(sigma_i*xi_i+sigma_s*xi_s)
            # All x will be stored in the deccision value list:
            decision_value.append(x[0])
    return decision_value

# Function of the drift diffusion model equation
def perfect_integrator_data(x, dt, tau, mu, sigma_i, xi_i, sigma_s, xi_s, n_trials, n_frames, dt_in_frame):
    stimulus_list = np.random.randn(n_trials, int(n_frames/dt_in_frame))
    decision_value_list = []
    for stimulus in stimulus_list:
        decision_value = perfect_integrator(x,stimulus,dt_in_frame,dt,tau,mu,sigma_i,sigma_s)
        decision_value_list.append(decision_value)
    return decision_value_list, stimulus_list

def DDM_absorbing(x,stimulus,dt_in_frame,n_frames,dt,tau,mu,sigma_i,sigma_s,bound):
    decision_value = [x]
    # The process will be repeated a two hundred times (1 second stimulus)
    for xi_s in stimulus:
        for i in range(int(dt_in_frame)):
            xi_i = np.random.randn(1)
            # Solved differential diffusion equation:
            x = x - (dt/tau)*(-mu)+np.sqrt(dt/tau)*(sigma_i*xi_i+sigma_s*xi_s)
            if x >= bound:
                if len(decision_value)<= n_frames:
                    reached_bound = True
                    for i in range(int(n_frames-len(decision_value))):
                        decision_value.append(bound)
            if x <= -bound:
                if len(decision_value) <= n_frames:
                    reached_bound = True
                    for i in range(int(n_frames-len(decision_value))):
                        decision_value.append(-bound)
            else:
                if len(decision_value) < n_frames:
                    # All x before reaching the bound value will be stored in the deccision value list:
                    decision_value.append(x[0])
    return decision_value

# Function of the DDM absorbing model
def DDM_absorbing_data(x, dt, tau, mu, sigma_i, xi_i, sigma_s, xi_s, n_trials, n_frames, dt_in_frame,bound):
    stimulus_list = np.random.randn(n_trials, int(n_frames/dt_in_frame))
    decision_value_list = []
    for stimulus in stimulus_list:
        # The starting decision value will be 0:
        decision_value = DDM_absorbing(x,stimulus,dt_in_frame,n_frames,dt,tau,mu,sigma_i,sigma_s,bound)
        # All x will be stored in the deccision value list:
        decision_value_list.append(decision_value)
    return decision_value_list, stimulus_list


def DDM_reflecting(x,stimulus,dt_in_frame,dt,tau,mu,sigma_i,sigma_s,bound):
    # The starting decision value will be 0:
    decision_value = [x]
    # The process will be repeated a two hundred times (1 second stimulus)
    for xi_s in stimulus:
        for i in range(int(dt_in_frame)):
            xi_i = np.random.randn(1)
            # Solved differential diffusion equation:
            x = x - (dt/tau)*(-mu)+np.sqrt(dt/tau)*(sigma_i*xi_i+sigma_s*xi_s)
            x = x[0]
            if x >= bound:
                x = bound
            elif x <= -bound:
                x = -bound
            # All x will be stored in the deccision value list:
            decision_value.append(x)
    return decision_value


# Function of the DDM reflecting model
def DDM_reflecting_data(x, dt, tau, mu, sigma_i, xi_i, sigma_s, xi_s, n_trials, n_frames, dt_in_frame,bound):
    stimulus_list = np.random.randn(n_trials, int(n_frames/dt_in_frame))
    decision_value_list = []
    for stimulus in stimulus_list:
        decision_value  = DDM_reflecting(x,stimulus,dt_in_frame,dt,tau,mu,sigma_i,sigma_s,bound)
        decision_value_list.append(decision_value)
    return decision_value_list, stimulus_list

def DW(x,stimulus,dt_in_frame,dt,tau,mu,alfa,sigma_i,sigma_s):
    decision_value = [x]
    # The process will be repeated a two hundred times (1 second stimulus)
    for xi_s in stimulus:
        for i in range(int(dt_in_frame)):
            xi_i = np.random.randn(1)
            # Solved differential diffusion equation:
            x = x -(dt/tau)*(-mu-alfa*x*2+4*x**3)+np.sqrt(dt/tau)*(sigma_i*xi_i[0]+sigma_s*xi_s)
            # All x will be stored in the deccision value list:
            decision_value.append(x)
    return decision_value
                       
# Function of the double well model
def DW_data(x, dt, tau, mu, sigma_i, sigma_s, alfa, n_trials, n_frames, dt_in_frame):
    stimulus_list = np.random.randn(n_trials, int(n_frames/dt_in_frame))
    decision_value_list = []
    for stimulus in stimulus_list:
            decision_value = DW(x,stimulus,dt_in_frame,dt,tau,mu,alfa,sigma_i,sigma_s)
            decision_value_list.append(decision_value)
    return decision_value_list, stimulus_list

