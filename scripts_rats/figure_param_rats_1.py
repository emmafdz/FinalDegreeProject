import sys
from os.path import expanduser
home = expanduser("~")
home=home+'/cluster_home'
print(home+'/fit_behaviour_julia/functions')
sys.path.insert(0, home+'/fit_behaviour_julia/functions')

import os
import behaviour_analysis as ba
import help_plot as hp

import numpy as np
import pickle
from importlib import reload
import pandas as pd
import matplotlib.pyplot as plt
import json as js

reload(ba)

isub=1
SUBJECTS=[25,35,37]
subject=SUBJECTS[int(isub)]





# model_fit="Stconst_DW"
# labels=[r"$c_{2,0}$", r"c_4",r"$k$"]
# Nparam=3
#
# fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(25./2.54,10./2.54))
# axes=np.hstack(axes)



#model_fit="StLinTimeDw"
model_fit="StLogTimeDw"
#uptade labels
#labels=[r"$A$",r"$\beta$",r"$\beta_t$"]
#labels=[r"$\beta$",r"$\beta_t$"]
# labels=[r"$c_{2,0}$",r"$c_{2,u}$",r"$c_{4}$",r"$\beta$"]
#labels=[r"$c_{2,0}$"]
#labels=[r"$c_{2}$",r"$c_{4}$",r"$\beta$",r"$\beta_t$"]
labels=[r"$c_{2}$",r"$c_{4}$",r"$A$",r"$\beta$"]
Nparam=len(labels)
#update nrwos, ncols
nrows=1
ncols=4
fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(25./2.54,10./2.54))
axes=np.hstack(axes)




colors_subject=["#cf4396","#59b94c","#a258c6","#a8b546","#5a6dc5",
"#d7a044","#ab8ad0","#5a8437","#d43f4b","#60c395","#d47baf",
"#358461","#c8683a","#4faad1","#8c7537","#b75d6c"]

path_figures=home+"/fit_behaviour_julia/figures/"

for isubject,subject in enumerate(SUBJECTS):

	path_results_fit=home+"/fit_behaviour_julia/results/summary/model_fit"+model_fit+"/"
	filename_load=path_results_fit+"rat"+str(subject)+"_1sec.json"
	with open(filename_load) as json_file:
		df = js.load(json_file)
	print(len(df["LL_training"]),len(df["ci"]))
	df["ci"] = [[0] for i in range(len(df["LL_training"]))]

	df = pd.DataFrame(df)
	index_min=[]
	imin1=df["LL_training"].idxmin()
	print(df.loc[imin1]["LL_training"])
	# for i in range(len(df)):
	# 	if df.ci[i][1]>0:
	# 		index_min.append(i)
	#
	# df=df.loc[index_min]


	imin=df["LL_training"].idxmin()
	print("ic:",len(df), " ",df.loc[imin]["LL_training"])

	idx=np.where(df["LL_training"]<df.loc[imin]["LL_training"]+1)[0]
	ci=df.loc[imin]["ci"]

	# if ci[0]<0:
	# 	for i in idx:
	# 		ci_aux=df.loc[i]["ci"]
	# 		if ci_aux[0]>0:
	# 			imin=i
	# param=df.loc[imin]["param_fit"]
	# ci=df.loc[imin]["ci"]

	for iparam in range(Nparam):

		# for i in idx:
		# 	axes[iparam].plot([isubject+0.2],[ df.loc[i]["param_fit"][iparam] ],".",color="grey")

		# if isubject%4==0:
		# 	hp.yticks(axes[iparam], range(Nparam),labels)
		# else:
		# 	hp.yticks(axes[iparam], range(Nparam),["","","","","","","","",""])

		if ci[0]>0:
			axes[iparam].errorbar([isubject],[ df.loc[imin]["param_fit"][iparam] ],yerr=[ ci[iparam] ],fmt="o",color=colors_subject[isubject])
		else:
			axes[iparam].plot([isubject],[ df.loc[imin]["param_fit"][iparam] ],"o",color=colors_subject[isubject])


		#axes[isubject].plot([0.1],[8],'|r')
		#axes[isubject].set_title("#min"+ str(len(df))+" # good"+str(len(idx)) )
		axes[iparam].set_title(labels[iparam])

		#axes[-1].plot([0,15],[0.1,0.1],"k-",lw=0.1)
fig.tight_layout()

fig.savefig(path_figures+"param_fit_model"+model_fit+"organized_by_parameters.png")
#print("hola0")

fig.show()

print("hola")
fig2, axes2 = plt.subplots(nrows=4, ncols=4,figsize=(20./2.54,20./2.54))
axes2=np.hstack(axes2)
print("hola2")

for isubject,subject in enumerate(SUBJECTS):
	print("hola3")

	axes=np.hstack(axes)

	path_results_fit=home+"/fit_behaviour_julia/results/summary/model_fit"+model_fit+"/"
	filename_load=path_results_fit+"rat"+str(subject)+"_1sec.json"
	with open(filename_load) as json_file:
		df = js.load(json_file)
	df["ci"] = [[0] for i in range(len(df["LL_training"]))]
	df = pd.DataFrame(df)
	imin=df["LL_training"].idxmin()



	param=df.loc[imin]["param_fit"]
	ci=df.loc[imin]["ci"]
	idx=np.where(df["LL_training"]<df.loc[imin]["LL_training"]+1)[0]



	for i in idx:
		print(1)
		#axes2[isubject].plot( df.loc[i]["param_fit"],np.array(range(Nparam))+0.2 ,".",color="grey")

	if isubject%4==0:
		hp.yticks(axes2[isubject], range(Nparam),labels)
	else:
		hp.yticks(axes2[isubject], range(Nparam),["" for i in range(Nparam)])

	if ci[0]>0:
		axes2[isubject].errorbar(df.loc[imin]["param_fit"], range(Nparam),xerr=ci ,fmt="o",color=colors_subject[isubject])
	else:
		axes2[isubject].plot(df.loc[imin]["param_fit"], range(Nparam),"s",color=colors_subject[isubject])


	#axes2[isubject].plot([0.1],[8],'|r')
	axes2[isubject].set_title("#min"+ str(len(df))+" # good"+str(len(idx)) )

	#axes[-1].plot([0,15],[0.1,0.1],"k-",lw=0.1)

fig2.tight_layout()
fig2.savefig(path_figures+"param_fit_model"+model_fit+"organized_by_subjects_1.png")
fig2.show()
