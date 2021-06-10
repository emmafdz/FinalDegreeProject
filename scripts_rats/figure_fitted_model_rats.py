import sys
from os.path import expanduser
home = expanduser("~")
# home=home+'/cluster_home'
sys.path.insert(0, home+'/Dropbox/python_functions')
sys.path.insert(0, home+'/cluster_bsc/fit_behaviour_julia/scripts/functions')


import behaviour_analysis as ba
import behaviour_plot as bp
import numpy as np
import simulation_diffusion as sd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import matplotlib.cm as cmx
import help_plot as hp
import pickle
import function_plots as fp

reload(sd)
reload(ba)
reload(bp)
reload(fp)

SUBJECTS=[1,2,3,4,5]

# path="/home/gprat/cluster_home/model_fitting_julia/results/"
# path="/home/genis/cluster_home/model_fitting_julia/results/"
#local
path="/home/genis/cluster_bsc/fit_behaviour_julia/results/"
path_poster="/home/genis/Dropbox/DECISION_MAKING_DYNAMICS/TALKS_POSTERS/CNS2019/"
#laptop
#path="/home/gprat/cluster_bsc/fit_behaviour_julia/results/"

fontsize=12
aux=["","fit_"]
color=['darkorange','black']
fmt=["-o","-"]
model_num="2"


model_num="3"
param=["c20","c21","c40","c6", "st0","st1","st3","st4","st5","sigma_i"]
model_fit = "StLogisticTimePi"
filename_load=path_results+"/summary/model_fit"+model_fit+"/rat"+str(subject)+".json"
df=pd.read_json(filename_load)
imin=df["LL"].idxmin()
param_fit=np.array(df.loc[imin]["param"])
PR_fit=np.array(df.loc[imin]["PR"])

##############  cartoon model #################

fig_model,axes=plt.subplots(1,2,figsize=(15/2.54,8/2.54))


# x=np.linspace(-2,2)

# beta=3
# b0=0.4
# y=1/(1+np.exp(-beta*(x-b0)))


# axes[0].plot(x,y,"k-",linewidth=2)
# axes[0].plot([0,0.],[0.,1.],'--',color='grey')
# axes[0].plot([-2,2.],[0.5,.5],'--',color='grey')
# hp.remove_axis(axes[0])

# axes[0].set_xlabel("Decision variable (x)",fontsize=fontsize)
# axes[0].set_ylabel("Probability Right",fontsize=fontsize)


# hp.yticks(axes[0],[0,0.5,1],fontsize=fontsize)
# hp.xticks(axes[0],[-2,0,2],fontsize=fontsize)

# s0=4
# beta=1
# b=0.4
# for t in range(10):
# 	y=s0*(1./(1+np.exp(-(beta+b*t)*x))-0.5)
# 	axes[1].plot(x,y,"k-",linewidth=2,color=hp.colormap("Blues",t,-2,9))



# hp.remove_axis(axes[1])


# axes[1].set_xlabel("Stimulus strength",fontsize=fontsize)
# axes[1].set_ylabel("Evidence",fontsize=fontsize)
# hp.yticks(axes[1],[-2,0,2],fontsize=fontsize)
# hp.xticks(axes[1],[-2,0,2],fontsize=fontsize)



# plt.show()
# plt.tight_layout()
# fig_model.savefig(path_poster+"cartoon_model.pdf")
# #axes[0].plot()


# ##############  potential read_out stim_transform #################

# fig_potential,axes=plt.subplots(4,5,figsize=(23.75/2.54,18/2.54))

# for isub, subject in enumerate(SUBJECTS):
# 	fp.Plot_Potentials_fit(subject,"2",axes[isub],[0,5,9],fontsize=fontsize)


# for isub, subject in enumerate(SUBJECTS):
# 	fp.Plot_read_out(subject,"2",axes[isub][3],fontsize=fontsize)
# 	fp.Plot_stim_trans(subject,"2",axes[isub][4],fontsize=fontsize1)


# axes[3][1].set_xlabel("Decision variable (x)",fontsize=fontsize)
# axes[2][0].set_ylabel("Potential",fontsize=fontsize)


# axes[3][3].set_xlabel("Decision variable (x)",fontsize=fontsize)
# axes[2][3].set_ylabel("Probability  Right",fontsize=fontsize)



# axes[3][4].set_xlabel("Stim strength ",fontsize=fontsize)
# axes[2][4].set_ylabel("Evidence",fontsize=fontsize)

# plt.tight_layout()
# plt.show()
# fig_potential.savefig(path_poster+"potential_read_out.pdf")
