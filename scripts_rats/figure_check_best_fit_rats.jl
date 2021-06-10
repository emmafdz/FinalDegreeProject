using JLD
using Plots
using JSON

# local
# @everywhere include("/home/genis/cluster_home/model_fitting_julia/gradient_utils.jl")
# @everywhere include("/home/genis/cluster_home/model_fitting_julia/functions_julia.jl")
# path="/home/genis/cluster_home/model_fitting_julia/"

#portatil
#
path="/home/emma/fit_behaviour_julia/"
# include(path*"/scripts/functions/functions_julia.jl")
# include(path*"/scripts/functions/utils.jl")
# include(path*"/scripts/functions/analysis.jl")
# include(path*"/scripts/functions/parameter_functions.jl")
# include(path*"/scripts/functions/simulations.jl")


SUBJECTS=[1,2,4,5]

#model_fit="urgency_x0_biasd_st3_dxdt42"
model_fit="StLinearDw_dx"



##### PARAM simulations ######3

tau=1.0
Tframe=0.5
Nframes=10
T=Tframe*Nframes

dt=tau/100.0
NT=T/dt
#############################

#plotly(ticks=:native)
#fig,axes=subplots(4,4,figsize=(20/2.54,20/2.54))


#colors_subject=["#cf4396","#59b94c","#a258c6","#a8b546","#5a6dc5",
#"#d7a044","#ab8ad0","#5a8437","#d43f4b","#60c395","#d47baf",
#"#358461","#c8683a","#4faad1","#8c7537","#b75d6c"]


for isubject in 1:length(SUBJECTS)

    subject=SUBJECTS[isubject]
	 
    path_save=path*"figures/"
    filename_save=path_save*"check_PR_fit_PR_sim"*model_fit*"_subject"*string(subject)*".jld"
    data=JLD.load(filename_save)
    PR_sim=data["PR_sim"]
    PR_fit=data["PR_fit"]
    #axes[isubject].plot([0,1],[0,1],"k--")
    #axes[isubject].plot(PR_sim,PR_fit[1:length(PR_sim)],".",color=colors_subject[isubject])
    #axes[isubject].set_title(string(subject) )
    #if isubject==4
        #axes[isubject].set_xlabel("PR simulations")
    #end
    #if isubject==1 || isubject==4
        #axes[isubject].set_ylabel("PR Brunton")
    #end
end

#fig.tight_layout()
#fig.show()
#path_save=path*"figures/"
#fig.savefig(path_save*"PR_sim_PR_fit_model"*model_fit*".png")
