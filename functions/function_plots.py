
import behaviour_analysis as ba
import behaviour_plot as bp
import numpy as np
import simulation_diffusion as sd
import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# import matplotlib as mpl
# import matplotlib.cm as cmx	
import help_plot as hp 
import pickle
import pandas as pd
reload(sd)
reload(ba)
reload(bp)

subject=14
print "subejct", subject
path1="/home/gprat/"
path_data="/home/genis/cluster_archive/Master_Project/att_behavioral_data/"
#path_data="/home/gprat/cluster_archive/Master_Project/att_behavioral_data/"
# path_data="/archive/genis/Master_Project/att_behavioral_data/data_S"+str(subject)+"/Results_Exp_subject"+str(subject)+".pickle"


#path="/home/gprat/cluster_bsc/fit_behaviour_julia/"
path="/home/genis/cluster_bsc/fit_behaviour_julia/"

path_figures=path+'figures/'
path_results=path+'results/'
# model_num="1"
# param=["c20","c21","c40", "st0","st1","st3","sigma_i"]




def st_logistic_time(stim,st_p):
	Nframes=len(stim)
	s_trans=np.zeros(Nframes)
	for iframe in range(Nframes):
		s_trans[iframe]=st_p[0]*(1./(1+np.exp(-(st_p[1]+st_p[2]*(iframe-1))*stim[iframe] ) )-0.5 )
	return s_trans





def Plot_Potentials_fit(subject,model_num,axes,frames,fontsize=15,xlabel="",ylabel=""):
	fname_data=path_data+"data_S"+str(subject)+"/Results_Exp_subject"+str(subject)+".pickle"

	f=open(fname_data,'rb')
	data=pickle.load(f)
	f.close()
	mu=data['mu'][0]
	Nframes=len(data['stim'][0])



	MUS=[0,mu,0.5,1]
	STIM=[ mu2*np.ones(Nframes) for mu2 in MUS ]

	filename_load=path_results+"/summary_results_optimitzation_subject"+str(subject)+"_model_"+model_num+".json"
	df=pd.read_json(filename_load,orient='records')
	imin=df["LL"].idxmin()
	param_fit=np.array(df.loc[imin]["param"])
	bins=np.array(df.loc[imin]["bins"])
	c2=np.array(df.loc[imin]["c2"])
	c4=np.array(df.loc[imin]["c4"])

	ncol=len(frames)
	nrow=1
	
	ymax=0
	ymin=0


	for imu in range(len(MUS)):
		strans=st_logistic_time(STIM[imu],param_fit[3:6])
		for i,iframe in enumerate(frames):
			 ####param_st for model2
			irow=iframe/ncol
			icol=iframe%ncol

			y=hp.potential_DW(bins,[strans[iframe],c2[iframe],c4[iframe]])
			ymax=max([max(y),ymax])
			ymin=min([min(y),ymin])
			axes[i].plot(bins,y,color=hp.colormap('Purples',imu,-2,len(MUS)))

	for i,iframe in enumerate(frames):
		hp.remove_axis(axes[i])
		hp.xticks(axes[i],[round(bins[0],1),0,round(bins[-1],1)],fontsize=fontsize)
		hp.yticks(axes[i],[round(ymin,1),0,round(ymax,1)],["","",""],fontsize=fontsize)
		axes[i].set_xlabel(xlabel,fontsize=fontsize)

	hp.yticks(axes[0],[round(ymin,1),0,round(ymax,1)],fontsize=fontsize)
	axes[0].set_ylabel(ylabel,fontsize=fontsize)

	return axes





def Plot_read_out(subject,model_num,axes,fontsize=15,xlabel="",ylabel=""):

	filename_load=path_results+"/summary_results_optimitzation_subject"+str(subject)+"_model_"+model_num+".json"
	df=pd.read_json(filename_load,orient='records')
	imin=df["LL"].idxmin()
	param_fit=np.array(df.loc[imin]["param"])
	bins=np.array(df.loc[imin]["bins"])
	read_out=np.array(df.loc[imin]["ro"])

	axes.plot(bins,read_out,'ko-')
	axes.plot([bins[0],bins[-1]],[0.5,0.5],'k--')
	axes.plot([0,0],[0,1],'k--')
	hp.yticks(axes,[0,0.5,1], fontsize=fontsize)
	hp.xticks(axes,[ round(bins[0],1),0,round(bins[-1],1)], fontsize=fontsize)
	hp.remove_axis(axes)
	axes.set_xlabel(xlabel,fontsize=fontsize)
	axes.set_ylabel(ylabel,fontsize=fontsize)

	return axes


def Plot_stim_trans(subject,model_num,axes,fontsize=15,xlabel="",ylabel=""):

	filename_load=path_results+"/summary_results_optimitzation_subject"+str(subject)+"_model_"+model_num+".json"
	df=pd.read_json(filename_load,orient='records')
	imin=df["LL"].idxmin()
	param_fit=np.array(df.loc[imin]["param"])
	bins=np.array(df.loc[imin]["bins"])
	read_out=np.array(df.loc[imin]["ro"])

	Nframes=10
	stim_aux= np.array([ np.linspace(-2,2,100) for i in range(Nframes) ]) 
	stim_aux_t=np.transpose(stim_aux)
	stim_trans=np.transpose(np.array([ st_logistic_time(stim_aux_t[i],param_fit[3:6]) for i in range(len(stim_aux_t)) ]))


	for i in range(Nframes):
		axes.plot(stim_aux[i],stim_trans[i],color=hp.colormap('Blues',i,-2,Nframes))

	# axes.plot(bins,read_out,'ko-')
	# axes.plot([bins[0],bins[-1]],[0.5,0.5],'k--')
	# axes.plot([0,0],[0,1],'k--')
	hp.remove_axis(axes)
	axes.set_xlabel(xlabel,fontsize=fontsize)
	axes.set_ylabel(ylabel,fontsize=fontsize)

	hp.xticks(axes,[-2,0,2],fontsize=fontsize)
	hp.yticks(axes,[-1,0,1],fontsize=fontsize)

	return axes

