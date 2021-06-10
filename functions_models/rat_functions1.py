
# coding: utf-8

# In[ ]:


# Importing all the libraries that will be used
import os
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import curve_fit
from PIL import Image


# In[ ]:


""" Function that will read all the files in a directory and extract the variables necessary for the analysis of the data"""
def read_files(directory):
    file_names = []
    directory = os.path.join("c:\\",directory)
    for root,dirs,files in os.walk(directory):
        for file in files:
            if file.endswith(".mat"):
                file_names.append(file)
    stim = []
    coh = []
    reward = []
    decision = []
    performance = []
    sigma = []

    for file_name in file_names:
        Protocol = "" 
        for i in np.arange(6,30,1): 
            Protocol = Protocol + file_name[i]
            if i == 29:
                if file_name[i+1] != "_":
                    Protocol = Protocol+file_name[i+1]
        annots = loadmat(file_name)
        nTrial = int(annots['saved']['ProtocolsSection_n_completed_trials'])
        HitHistory = annots['saved'][Protocol+'_HitHistory'][0][0][1:nTrial+1]
        RewardSideList = annots['saved'][Protocol+'_RewardSideList'][0][0][0][0][1][0:]
        StimulusList = annots['saved'][Protocol+'_StimulusList'][0][0][0][0][0][0:]
        CoherenceVec_rat = annots['saved_history'][Protocol+'_TargetCoherenceVec'][0][0][0:]
        TargetCoherence = annots['saved_history'][Protocol+'_TargetCoherence'][0][0][0:]
        TargetSigma = annots['saved_history'][Protocol+'_TargetSigma'][0][0][0:]
        
        hit_history = [i[0] for i in HitHistory]
        reward_side_list = [i[0] for i in RewardSideList]
        stimulus_list = [i[0] for i in StimulusList]
        stimulus = [i[0][0] for i in CoherenceVec_rat]
        target_coherence = [i[0][0][0] for i in TargetCoherence]
        target_sigma = [i[0][0][0] for i in TargetSigma]
        rat_decision = []
        rat_performance = []
        valid_trials = []
        index = 0
        invalid_trials = 0
        reward_side_list = [reward_side_list[i-1] for i in stimulus_list]
        for i in hit_history:
            if i == 0:
                if reward_side_list[index] == 1:
                    rat_decision.append(2)
                elif reward_side_list[index] == 2:
                    rat_decision.append(1)
                else:
                    print("Reward_side_list no és ni 1 ni 2:  és ", reward_side_list[index])    
                rat_performance.append(0)
                valid_trials.append(index)
            elif i == 1:
                rat_decision.append(reward_side_list[index])
                rat_performance.append(1)
                valid_trials.append(index)
            else:
                invalid_trials += 1
            index += 1
        stimulus_list = [stimulus_list[i] for i in valid_trials]
        reward_side_list = [reward_side_list[i] for i in valid_trials]
        target_coherence = [target_coherence[i-1] for i in stimulus_list]
        target_sigma = [target_sigma[i-1] for i in stimulus_list]
        stimulus = [stimulus[i-1] for i in stimulus_list]
        
        stim_2 = []
        for i in stimulus:
            stim_1 = []
            for a in i:
                stim_1.append(a)
            stim_2.append(stim_1)
        decision_value_list = []

        for dec in rat_decision:
            if dec == 2:
                decision_value_list.append(1)
            else:
                decision_value_list.append(0)
        stim.append(stim_2)
        coh.append(target_coherence)
        reward.append(reward_side_list)
        decision.append(decision_value_list)
        performance.append(rat_performance)
        sigma.append(target_sigma)
    return(stim,coh,reward,decision,performance,sigma)


# In[ ]:


"""Function that will keep only one of the elements repeated on a list"""
def count(mylist):
    newlist = []
    for i in mylist:
        if i not in newlist:
            newlist.append(i)
    return newlist


# In[ ]:


"""Function that will return the coherences vector"""
def return_coherences_vector(coh):
    coherences = []
    for i in count(coh):
            coherences.append(i)
    coherences = count(coherences)
    coh_vec = sorted(coherences)
    return(coh_vec)

def return_vector(coh):
    coherences = []
    for a in range(len(coh)):
        for i in count(coh[a]):
            coherences.append(i)
    coherences = count(coherences)
    coh_vec = sorted(coherences)
    return(coh_vec)


# In[ ]:


"""Function that will return the data of the coherence it is given"""
def divide_coh(coh,reward,decision,performance,stim,target_sigma,coherence):
    reward_1 = []
    decision_1 = []
    performance_1 = []
    stim_1 = []
    target_sigma_1 = []
    indexes = np.where(np.asarray(coh) == coherence)[0]
    for i in indexes:
        reward_1.append(reward[i])
        decision_1.append(decision[i])
        performance_1.append(performance[i])
        stim_1.append(stim[i])
        target_sigma_1.append(target_sigma[i])
    return reward_1,decision_1,performance_1,stim_1,target_sigma_1


# In[ ]:


"""Function that will return the data sorted into one second and half second data"""
def divide_time(stim,decision_value_list):
    sec_stim = []
    decision_half =[]
    decision_sec = []
    half_sec_stim = []
    index = 0
    for st in stim:
        if len(st) == 20:
            sec_stim.append(st)
            decision_sec.append(decision_value_list[index])
        else:
            half_sec_stim.append(st)
            decision_half.append(decision_value_list[index])
        index += 1
    return sec_stim, decision_sec, half_sec_stim, decision_half


# In[ ]:


"""Function that will compute the PK"""
def PK(decision,stim,axs,col):
    logit_mod = sm.Logit(decision, stim)
    result = logit_mod.fit()

    pars_list =result.params
    confidence_interval = result.conf_int()
    conf_int = [[i[0] for i in confidence_interval],[i[1] for i in confidence_interval]]
    conf_int = [pars_list-conf_int[0],conf_int[1]-pars_list]
    axs.errorbar(np.arange(0,len(pars_list),1),pars_list,conf_int,color = col,marker='s')

    return(confidence_interval)


# In[ ]:


"""Function that will divide the data into the sigma it is given"""
def divide_sigma(reward,decision,performance,stim,target_sigma,sigma):
    indexes = np.where(np.asarray(target_sigma) == sigma)[0]
    reward_1 = [reward[i] for i in indexes]
    decision_1 = [decision[i] for i in indexes]
    performance_1 = [performance[i] for i in indexes]
    stim_1 = [stim[i] for i in indexes]
    return reward_1,decision_1,performance_1,stim_1


# In[ ]:


"""Table with the recount of trials"""
def recount_trials_all_rats(sigma_lists,coherences_all_rats,length_all_rats,rat_name,coherence_vectors,rats,sigmas_all_rats,directory_images):
    len_matrix = np.zeros((len(sigmas_all_rats),len(coherences_all_rats)))

    for rat in range(len(rats)):
        for i in range(len(sigma_lists[rat])):
            for a in range(len(coherence_vectors[rat])):
                index_i = sigmas_all_rats.index(sigma_lists[rat][i])
                index_a = coherences_all_rats.index(coherence_vectors[rat][a])
                len_matrix[index_i][index_a] += length_all_rats[rat][i][a]

    fig, ax = plt.subplots(figsize = (15,15))
    im = ax.imshow(len_matrix,cmap = "YlOrBr")

    # We want to show all ticks...
    ax.set_xticks(np.arange(0,len(coherences_all_rats),1))
    ax.set_yticks(np.arange(0,len(sigmas_all_rats),1))
    # ... and label them with the respective list entries
    ax.set_xticklabels(coherences_all_rats)
    ax.set_yticklabels(sigmas_all_rats)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(sigmas_all_rats)):
        for j in range(len(coherences_all_rats)):
            text = ax.text(j, i, int(len_matrix[i, j]),
                           ha="center", va="center", color="k")

    ax.set_title("Number of trials per sigma and coherence for all rats")
    fig.tight_layout()
    plt.xlabel("coherence")
    plt.ylabel("sigma")
    fig.savefig(directory_images+'/recount_trials/recount_trials_all_rats.png', bbox_inches = 'tight')
    plt.close(fig)
    
def recount_trials(sigma_lists,coh_vec,len_lists,rat_name,coh_lists,directory_images):
    len_matrix = np.zeros((len(sigma_lists),len(coh_vec)))
    for i in range(len(len_lists)):
        for a in range(len(len_lists[i])):
            len_matrix[i][int(coh_vec.index(coh_lists[i][a]))] =len_lists[i][a]

    fig, ax = plt.subplots(figsize = (15,15))
    im = ax.imshow(len_matrix,cmap = "YlOrBr")

    # We want to show all ticks...
    ax.set_xticks(np.arange(0,len(coh_vec),1))
    ax.set_yticks(np.arange(0,len(sigma_lists),1))
    # ... and label them with the respective list entries
    ax.set_xticklabels(coh_vec)
    ax.set_yticklabels(sigma_lists)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(sigma_lists)):
        for j in range(len(coh_vec)):
            text = ax.text(j, i, int(len_matrix[i, j]),
                           ha="center", va="center", color="k")

    ax.set_title("Number of trials per sigma and coherence for "+rat_name)
    fig.tight_layout()
    plt.xlabel("coherence")
    plt.ylabel("sigma")
#     plt.show()
    fig.savefig(directory_images+'/recount_trials/recount_trials'+rat_name+'.png', bbox_inches = 'tight')
    plt.close(fig)


# In[ ]:


"""Function that will divide the data taking into account the sigma"""
def divide_sigma_1(reward,decision,performance,stim,target_sigma,sigma):
    
    reward_1 = []
    decision_1 = []
    performance_1 = []
    stim_1 = []

    indexes = [i for i, x in enumerate(target_sigma) if x == sigma[0] or x == sigma[1]]
    for i in indexes:
        reward_1.append(reward[i])
        decision_1.append(decision[i])
        performance_1.append(performance[i])
        stim_1.append(stim[i])

    return reward_1,decision_1,performance_1,stim_1

def compute_percentages(rats,results_divided_sigma,coherence_vectors):
    # The percentages for each coherence will be computed
    percentages_lists = []

    coherences_lists = []
    length_lists = []
    length_all_rats = []

    for rat in range(len(rats)):
        perc_list = []
        coh_lists = []
        len_lists = []
        len_lists_all = []
        for i in range(len(results_divided_sigma[rat][0])):
            perc = []
            coh_list = []
            len_list=[]
            len_list_all = []
            for a in range(len(results_divided_sigma[rat])):
                if len(results_divided_sigma[rat][a][i][1])>0:
                    perc.append(sum(results_divided_sigma[rat][a][i][1])/len(results_divided_sigma[rat][a][i][1]))
                    coh_list.append(coherence_vectors[rat][a])
                    len_list.append(len(results_divided_sigma[rat][a][i][1]))
                len_list_all.append(len(results_divided_sigma[rat][a][i][1]))
            perc_list.append(perc)
            coh_lists.append(coh_list)
            len_lists.append(len_list)
            len_lists_all.append(len_list_all)
        percentages_lists.append(perc_list)
        coherences_lists.append(coh_lists)
        length_lists.append(len_lists)
        length_all_rats.append(len_lists_all)
    return percentages_lists, coherences_lists, length_lists, length_all_rats

def bootstrap(results_divided_sigma):
    new_divided_sigma = []
    for rat in range(len(results_divided_sigma)):
        new_d_s = []
        for cohe in range(len(results_divided_sigma[rat])):
            new_d = []
            for sigm in range(len(results_divided_sigma[rat][cohe])):
                resampled = np.random.choice(results_divided_sigma[rat][cohe][sigm][1],size = len(results_divided_sigma[rat][cohe][sigm][1]),replace = True)
                new_d.append([0,resampled,0,0])
            new_d_s.append(new_d)
        new_divided_sigma.append(new_d_s)
    return new_divided_sigma

def compute_sensitivity(rats,coherences_lists,percentages_lists,sigma_lists,directory_images):
    # Fitted sensitivity of the psychometric curve
    def sigmoidscaled(x_vec, k,lower_lim,up_lim,bias):
        return [lower_lim + (1-lower_lim-up_lim)/(1+np.exp(-k*(x-bias))) for x in x_vec]
    all_popt = []
    all_sensitivity = []
    all_bias = []
    for rat in range(len(rats)):
        popt = []
        sensitivity_p = []
        bias_p = []
        for i in range(len(coherences_lists[rat])):
            try:
                p0 = [1,0,0,1]
                popt_1, pcov_1 = curve_fit(sigmoidscaled, coherences_lists[rat][i], percentages_lists[rat][i],p0,bounds = ((-np.inf,  0, 0,-np.inf), (np.inf, 1, 1,np.inf)))
                popt.append(popt_1)
                sensitivity_p.append(popt_1[0])
                bias_p.append(popt_1[3])
                fig, axs = plt.subplots(figsize = (8,5))
                axs.plot(coherences_lists[rat][i], sigmoidscaled(coherences_lists[rat][i], *popt_1), 'r-',label="fitted curve")
                axs.plot(coherences_lists[rat][i],percentages_lists[rat][i],label="real psychometric curve",marker = "o")
                axs.axis([-1,1,0,1])
                axs.vlines(0,-1,1,color = "lightgrey",linestyles = "--",label ="x = 0")
                axs.hlines(0.5,-1,1,color = "lightblue",linestyles = "--",label = "y = 0.5")
                fig.suptitle(rats[rat]+" Psychometric curve for sigma "+str(sigma_lists[rat][i]),fontsize = 18)
                fig.text(0.5, 0.02, "Stimulus evidence (coherence)", ha='center',fontsize = 16)
                fig.text(0.03, 0.5, "Probability to choose right", va='center', rotation='vertical',fontsize = 18)
                axs.legend(loc = "upper left",fontsize = 12)
                axs.spines['right'].set_visible(False)
                axs.spines['top'].set_visible(False)
                fig.savefig(directory_images+'fits/fit_'+rats[rat]+'_sigm_'+str(sigma_lists[rat][i])+'.png', bbox_inches = 'tight')
            except:
                popt.append(None)
                sensitivity_p.append(None)
                bias_p.append(None)      

        all_popt.append(popt)
        all_sensitivity.append(sensitivity_p)
        all_bias.append(bias_p)
    return all_sensitivity, all_bias 


def compute_sensitivity_boot(rats,coherences_lists,percentages_lists,sigma_lists):
    # Fitted sensitivity of the psychometric curve
    def sigmoidscaled(x_vec, k,lower_lim,up_lim,bias):
        return [lower_lim + (1-lower_lim-up_lim)/(1+np.exp(-k*(x-bias))) for x in x_vec]
    all_popt = []
    all_sensitivity = []
    all_bias = []
    for rat in range(len(rats)):
        popt = []
        sensitivity_p = []
        bias_p = []
        for i in range(len(coherences_lists[rat])):
            try:
                p0 = [1,0,0,1]
                popt_1, pcov_1 = curve_fit(sigmoidscaled, coherences_lists[rat][i], percentages_lists[rat][i],p0,bounds = ((-np.inf,  0, 0,-np.inf), (np.inf, 1, 1,np.inf)))
                popt.append(popt_1)
                sensitivity_p.append(popt_1[0])
                bias_p.append(popt_1[3])
            except:
                popt.append(None)
                sensitivity_p.append(None)
                bias_p.append(None)      

        all_popt.append(popt)
        all_sensitivity.append(sensitivity_p)
        all_bias.append(bias_p)
    return all_sensitivity, all_bias 

def transformed_stimulus(stimulus):

    transformed_stimulus = []
    for rat in range(len(stimulus)): # half_sec_stim_list_all or sec_stim_list_all
        transf_stim = []

        for stim in stimulus[rat]:
            stim_sorted = np.sort(stim)
            hist, base = np.histogram(stim,bins = 11,range = (-1,1))
            transf_stim.append(hist)
        transformed_stimulus.append(transf_stim)
    return transformed_stimulus

def divided_time_PK(rats,results_divided_sigma_nocoh,sigma_lists,rats_list,half_or_sec,directory_images):
     # results_divided_sigma_coh[rat][sigma][reward,decision,performance,stim]
    sec_stim_list_all = []
    decision_sec_list_all = []
    half_sec_stim_list_all = []
    decision_half_list_all = []
    for rat in rats:
        sec_stim_list = []
        decision_sec_list = []
        half_sec_stim_list = []
        decision_half_list = []
        for sigma in range(len(results_divided_sigma_nocoh[rat])):
            new_stimulus = []
            new_decision = []
            for stimulus in range(len(results_divided_sigma_nocoh[rat][sigma][3])):
                new_stimulus.append(results_divided_sigma_nocoh[rat][sigma][3][stimulus])
                new_decision.append(results_divided_sigma_nocoh[rat][sigma][1][stimulus])
            sec_stim, decision_sec, half_sec_stim, decision_half = divide_time(new_stimulus,new_decision)
            sec_stim_list.append(sec_stim)
            decision_sec_list.append(decision_sec)
            half_sec_stim_list.append(half_sec_stim)
            decision_half_list.append(decision_half)
        sec_stim_list_all.append(sec_stim_list)
        decision_sec_list_all.append(decision_sec_list)
        half_sec_stim_list_all.append(half_sec_stim_list)
        decision_half_list_all.append(decision_half_list)
        plt.figure()
        fig, ((axs1,axs2),(axs3,axs4)) = plt.subplots(2,2,figsize = (10,7))

        axs = [0,axs1,axs2,axs3,axs4,0,0]
        color = [0,"maroon","brown","red","orangered","orange","yellow"]
        index = 1
        aranges = [np.arange(2,6,1),np.arange(2,6,1),np.arange(2,6,1),np.arange(2,6,1),np.arange(1,5,1)]
        for a in aranges[rat]:
            if half_or_sec == "sec":
                conf_int = PK(decision_sec_list[a],sec_stim_list[a],axs[index],color[index])            
            else:
                conf_int = PK(decision_half_list[a],half_sec_stim_list[a],axs[index],color[index])
            axs[index].set_title("Sigma = "+str(sigma_lists[rat][a]))
            axs[index].spines['right'].set_visible(False)
            axs[index].spines['top'].set_visible(False)
            index +=1
        plt.ylim(-1.5,1.5)    
        fig.suptitle("PK for "+rats_list[rat],fontsize = 18)
        fig.text(0.5, 0.04, 'Frame nº', ha='center',fontsize = 16)
        fig.text(0.04, 0.5, 'Impact', va='center', rotation='vertical',fontsize = 18)
        
        fig.savefig(directory_images+'PK/temporal_PK'+rats_list[rat]+half_or_sec+'.png', bbox_inches = 'tight')
    return sec_stim_list_all, decision_sec_list_all, half_sec_stim_list_all, decision_half_list_all