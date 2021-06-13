#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Modifiable variables when changing computer location:
# Directory where the images will be stored:
directory_images = '/home/emma/github/TFG/results/fit_model/'
directory_functions = '/home/emma/github/TFG/functions'
directory_data = '/home/emma/github/TFG/data/'
directory_results = '/home/emma/github/TFG/results/'

# Importing all the libraries that will be used
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json 
from sklearn.linear_model import LinearRegression
import sys
import datetime
import pandas as pd

# Insert path where the functions are stored
sys.path.insert(1, directory_functions)

# Importing the functions that will be used
import rat_functions1 as rf
import help_plot as hp


# In[ ]:


fig, axs = plt.subplots(2,4,figsize =(15,7.5) )
PR_total = []
param_fits = []
i = 0
titles = []
for rat in [24,25,36,37]:
# for rat in [24]:
    # Opening JSON file 
    titles.append("Rat "+str(rat)+" 05 sec")
    f = open(directory_results+'PRs/check_PR_fit_PR_simStLinPi_subject'+str(rat)+'_05sec.json',) 
#     print('/check_PR_fit_PR_simStLinPi_subject'+str(rat)+'_05sec.json')
    # returns JSON object as  
    # a dictionary 
    data = json.load(f) 
    f.close()
    print(data.keys())
#     fig,axes = plt.subplots()
#     plt.plot(data["PR_sim"],data["PR_fit"][0:100],".")
    
    axs[0][i].plot([0,1],[0,1],"k--")
    axs[0][i].plot(data["PR_sim"][30:80],data["PR_fit"][30:80],".",color = "cornflowerblue")
    axs[0][i].tick_params(axis='y', labelsize=16 )
    axs[0][i].tick_params(axis='x', labelsize=16 )
    axs[0][i].set_title("Rat "+str(rat)+" 05 sec",fontsize = 18)

    f = open(directory_results+'PRs/synthetic_data_PI.json',) 
#     print('/check_PR_fit_PR_simStLinPi_subject'+str(rat)+'_05sec.json')
    # returns JSON object as  
    # a dictionary 
    data = json.load(f) 
    f.close()
    i+=1
i = 0
PR_total = []
param_fits = []
# for rat in [24,25,36,37]:
for rat in [25,35,37]:
    titles.append("Rat "+str(rat)+" 1 sec")
    # Opening JSON file 
    f = open(directory_results+'PRs/check_PR_fit_PR_simStLinPi_subject'+str(rat)+'_1sec.json',) 
    # returns JSON object as  
    # a dictionary 
    data = json.load(f) 
    f.close()
    print(data.keys())
#     fig,axes = plt.subplots()
#     plt.plot(data["PR_sim"],data["PR_fit"][0:100],".")
    
    axs[1][i].plot([0,1],[0,1],"k--")
    axs[1][i].plot(data["PR_sim"][50:100],data["PR_fit"][50:100],".",color = "cornflowerblue")
    axs[1][i].tick_params(axis='y', labelsize=16 )
    axs[1][i].tick_params(axis='x', labelsize=16)
    axs[1][i].set_title("Rat "+str(rat)+" 1 sec",fontsize = 18)

    i +=1
fig.tight_layout(pad=3.0)
fig.text(0.5, 0, "PR sim", ha='center',fontsize =24)
fig.text(0., 0.5, "PR fit", va='center', rotation='vertical',fontsize = 24)
title = "Double Well with logistic stimulus transformation"# \n with time adaptation"
fig.text(0.5, 0.97,title, ha ="center",fontsize = 24)
fig.savefig("/home/emma/github/TFG/figure_PR/StLogDw_1.png",bbox_inches='tight')


# In[ ]:


PR_total = []
param_fits = []
# for rat in [24,25,36,37]:
for rat in [25,35,37]:
    # Opening JSON file 
    f = open(directory_results+'PRs/check_PR_fit_PR_simStLinPi_subject'+str(rat)+'_1sec.json',) 
    # returns JSON object as  
    # a dictionary 
    data = json.load(f) 
    f.close()
    print(data.keys())
    fig,axes = plt.subplots()
#     plt.plot(data["PR_sim"],data["PR_fit"][0:100],".")
    
    axes.plot([0,1],[0,1],"k--")
    axes.plot(data["PR_sim"][0:50],data["PR_fit"][0:50],".")
    plt.xlabel("PR sim")
    plt.ylabel("PR fit")


# In[ ]:


#plt.plot([x[100] for x in xfinal])
plt.plot(xfinal[50])


# In[ ]:


fig, ax = plt.subplots(figsize=(50, 30))
ax.imshow(xfinal)


# In[ ]:



# Opening JSON file 
f = open(directory_results+'PRs/one_pdf.json',) 
# returns JSON object as  
# a dictionary 
data = json.load(f) 
f.close()
print(data.keys())
x = data["x"]
bins = data["bins"]
fig, ax = plt.subplots(figsize=(50, 30))
ax.imshow(x)


# In[ ]:


last_line = [x_1[49] for x_1 in x]
plt.plot(bins,[x_1[49]*500 for x_1 in x])
bi, xf =np.histogram(xfinal[50],bins=50)
print(len(bi))
plt.plot(xf[0:50],bi)
print("PR left:",sum(last_line[0:25]))
print("PR right:",sum(last_line[26:51]))


# In[ ]:



# Opening JSON file 
f = open(directory_results+'PRs/one_pdf_sim_1.json',) 
# returns JSON object as  
# a dictionary 
data = json.load(f) 
f.close()
print(data.keys())
x_n = data["sim"]
plt.plot(x_n[4])


# In[ ]:



# Opening JSON file 
f = open(directory_results+'PRs/check_best_syntheticStLinearDw.json',) 
# returns JSON object as  
# a dictionary 
data = json.load(f) 
f.close()
print(data.keys())
LL = data["LL"]
PR_sim = data["PR_sim"]
PR = data["PR"]
LL2 = data["LL2"]

Nstim=len(PR_sim)
plt.plot(PR[0:Nstim],PR_sim,".")
plt.plot([0,1],[0,1])

