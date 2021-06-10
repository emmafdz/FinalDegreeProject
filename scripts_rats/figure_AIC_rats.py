import sys
from os.path import expanduser
home = expanduser("~")
# home=home+'/cluster_home'
sys.path.insert(0, home+'/cluster_home/fit_behaviour_julia/functions')

import os
# import behaviour_analysis as ba
import help_plot as hp

import numpy as np
import pickle
from importlib import reload
import json as js
import pandas as pd
import matplotlib.pyplot as plt



#
def AIC(LL,Nparam,model_base):
	Delta_AIC={}
	for model in LL.keys():
		Delta_AIC[model]=[]
		for isubject in range(len(LL[model_base])):
			AIC_model=2*Nparam[model]+2*LL[model][isubject]
			AIC_model_base=2*Nparam[model_base]+2*LL[model_base][isubject]
			Delta_AIC[model].append(AIC_model-AIC_model_base)
	return Delta_AIC

# reload(ba)

isub=1
SUBJECTS=[24,25,36,37]
iSUBJECTS=range(5)
subject=SUBJECTS[int(isub)]

fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(25./2.54,5./2.54))




Nparam=4
#MODELS=["StLinPi","StLogTimePi","StLinTimePi","StLogPi","StLinDw"]
MODELS=["StLinPi","StLinDw","StLogPi","StLinTimePi","StLogDw","StLogTimePi"]
# MODELS=["PI_Stconst","PI_StLogistic","Stconst_DW","StLogistic_DW","urgency_x0_biasd_st3_dxdt42"]
nmodels=len(MODELS)

colors_subject=["#cf4396","#59b94c","#a258c6","#a8b546","#5a6dc5",
"#d7a044","#ab8ad0","#5a8437","#d43f4b","#60c395","#d47baf",
"#358461","#c8683a","#4faad1","#8c7537","#b75d6c"]

path_figures="/home/emma/cluster_home/fit_behaviour_julia/figures/"
LL={}
Nparam={}
for model in MODELS:
	LL[model]=[]
	#Nparam[model]=[]
	for isubject,subject in enumerate(SUBJECTS):

		path_results_fit="/home/emma/cluster_home/fit_behaviour_julia/results/summary/model_fit"+model+"/"
		filename_load=path_results_fit+"rat"+str(subject)+"_05sec.json"
		with open(filename_load) as json_file:
		    df = js.load(json_file)
		df = pd.DataFrame(df)
		# df=pd.read_json(filename_load)
		index_min=[]
		imin1=df["LL_training"].idxmin()
		#print(df.loc[imin1]["LL_training"])

		LL[model].append(df.loc[imin1]["LL_training"])
		if isubject==0:
			df.loc[imin1]["LL_training"]
			#Nparam[model].append(len(df.loc[imin1]["param_fit"]))
			Nparam[model]=len(df.loc[imin1]["param_fit"])
		print(df.loc[imin1]["param_fit"])

DeltaAIC=AIC(LL,Nparam,"StLogTimePi")
xticks=range(int(nmodels/2)-1,nmodels*len(SUBJECTS)-1, nmodels)
for imodel,model in enumerate(MODELS):
	axes.bar( range(imodel,nmodels*len(SUBJECTS)+imodel,nmodels),DeltaAIC[model],width=0.8,color=colors_subject[imodel],label =model)
	axes.set_xticks(xticks)
	axes.set_xticklabels(SUBJECTS)

axes.set_xlabel("rat")
axes.set_ylabel(r"$\Delta$AIC ")

axes.set_ylim(-10,10)

fig.tight_layout()
axes.legend(loc = 'upper right',fontsize = 'xx-small')
fig.show()
fig.savefig(home+'/cluster_home/fit_behaviour_julia/results/figure_AIC_05sec.jpg')
#
# 	# for i in range(len(df)):
# 	# 	if df.ci[i][1]>0:
# 	# 		index_min.append(i)
# 	#
# 	# df=df.loc[index_min]
#
#
# 	imin=df["LL_training"].idxmin()
# 	print("ic:",len(df), " ",df.loc[imin]["LL_training"])
#
# 	idx=np.where(df["LL_training"]<df.loc[imin]["LL_training"]+1)[0]
# 	ci=df.loc[imin]["ci"]
#
# 	# if ci[0]<0:
# 	# 	for i in idx:
# 	# 		ci_aux=df.loc[i]["ci"]
# 	# 		if ci_aux[0]>0:
# 	# 			imin=i
# 	# param=df.loc[imin]["param_fit"]
# 	# ci=df.loc[imin]["ci"]
#
# 	for iparam in range(Nparam):
#
# 		# for i in idx:
# 		# 	axes[iparam].plot([isubject+0.2],[ df.loc[i]["param_fit"][iparam] ],".",color="grey")
#
# 		# if isubject%4==0:
# 		# 	hp.yticks(axes[iparam], range(Nparam),labels)
# 		# else:
# 		# 	hp.yticks(axes[iparam], range(Nparam),["","","","","","","","",""])
#
# 		if ci[0]>0:
# 			axes[iparam].errorbar([isubject],[ df.loc[imin]["param_fit"][iparam] ],yerr=[ ci[iparam] ],fmt="o",color=colors_subject[isubject])
# 		else:
# 			axes[iparam].plot([isubject],[ df.loc[imin]["param_fit"][iparam] ],"o",color=colors_subject[isubject])
#
#
# 		#axes[isubject].plot([0.1],[8],'|r')
# 		#axes[isubject].set_title("#min"+ str(len(df))+" # good"+str(len(idx)) )
# 		axes[iparam].set_title(labels[iparam])
#
# 		axes[-1].plot([0,15],[0.1,0.1],"k-",lw=0.1)
# fig.tight_layout()
#
# fig.savefig(path_figures+"param_fit_model"+model_fit+"organized_by_parameters.png")
# plt.show()
#
#
# fig2, axes2 = plt.subplots(nrows=4, ncols=4,figsize=(20./2.54,20./2.54))
# axes2=np.hstack(axes2)
#
# for isubject,subject in enumerate(SUBJECTS):
# 	axes=np.hstack(axes)
#
# 	path_results_fit="fit_behaviour_julia/results/summary/model_fit"+model_fit+"/"
# 	filename_load=path_results_fit+"subject"+str(subject)+".json"
# 	df=pd.read_json(filename_load)
# 	imin=df["LL_training"].idxmin()
#
#
#
# 	param=df.loc[imin]["param_fit"]
# 	ci=df.loc[imin]["ci"]
# 	idx=np.where(df["LL_training"]<df.loc[imin]["LL_training"]+1)[0]
#
#
#
# 	for i in idx:
# 		axes2[isubject].plot( df.loc[i]["param_fit"],np.array(range(Nparam))+0.2 ,".",color="grey")
#
# 	if isubject%4==0:
# 		hp.yticks(axes2[isubject], range(Nparam),labels)
# 	else:
# 		hp.yticks(axes2[isubject], range(Nparam),["","","","","","","","",""])
#
# 	if ci[0]>0:
# 		axes2[isubject].errorbar(df.loc[imin]["param_fit"], range(Nparam),xerr=ci ,fmt="o",color=colors_subject[isubject])
# 	else:
# 		axes2[isubject].plot(df.loc[imin]["param_fit"], range(Nparam),"s",color=colors_subject[isubject])
#
#
# 	axes2[isubject].plot([0.1],[8],'|r')
# 	axes2[isubject].set_title("#min"+ str(len(df))+" # good"+str(len(idx)) )
#
# 	#axes[-1].plot([0,15],[0.1,0.1],"k-",lw=0.1)
#
# fig2.tight_layout()
# fig2.savefig(path_figures+"param_fit_model"+model_fit+"organized_by_subjects.png")
# fig2.show()
