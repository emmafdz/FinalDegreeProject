using JLD
# using Pandas
using Base.Test
import Base.convert
using MAT
using ForwardDiff
using Polynomials
import ForwardDiff.DiffBase
using JSON
# using PyPlot


# local
# @everywhere include("/home/genis/cluster_home/model_fitting_julia/gradient_utils.jl")
# @everywhere include("/home/genis/cluster_home/model_fitting_julia/functions_julia.jl")
# path="/home/genis/cluster_home/model_fitting_julia/"

#portatil

path="/home/emma/fit_behaviour_julia/"
include(path*"functions/functions_julia.jl")
include(path*"functions/utils.jl")
include(path*"functions/analysis.jl")
include(path*"functions/parameter_functions.jl")
include(path*"functions/simulations.jl")


SUBJECTS=[24,25,36,37]

model_fit="StLogTimePi"

Tframe=0.5

model_f=Dict(:c2_f=>c2_const,:c4_f=>c4_const,:c6_f=>c6_zeros,:bias_d_f=>bias_d_const,
        :hbias_d_f=>history_bias_d_const,:st_f=>st_logistic_time,:x0_f=>initial_condition_bias_hbias,
        :ro_f=>read_out_perfect,:lambda=>0)


###### PARAM simulations ######

tau=1.0


#############################


subject=1

for subject in SUBJECTS
#subject=4
    path_load=path*"results/summary/model_fit"*model_fit*"/"
    filename_load=path_load*"rat"*string(subject)*"_05sec.jld"
    data_fit=load(filename_load)

    iminimum=find(x->x==minimum(data_fit["LL_training"]),data_fit["LL_training"])
    param_best=data_fit["param_all"][iminimum][1]
#    param_best = Dict("c2"=> [10, 0.0],"st"=> [1, 0.0, 0.0, 0.0],"sigma_a"=> [1.0],"x0"=> [0.0, 0.0],"bias_d"=> [0.0],"c4"=>[1],"hbias_d"=> [0.0])
	#println("param_ebst", param_best)
    PR_fit=data_fit["PR_training"][iminimum][1]
    path_data=path*"scripts_rats/data/"

    # data = Dict()
    open(path_data*"processed_data_rat"*string(subject)*"_dataset2_05.json", "r") do f

        global data
        dicttxt = readstring(f)  # file information to string
        data=JSON.parse(dicttxt)  # parse and transform data
    end
    n_stim_10 = data["n_stim_10"]
    n_stim_20 = data["n_stim_20"]

	#println("nstim_10! ",n_stim_10)
    #Nstim=100
    # stim=data["stim"][1:Nstim,:]
    # choices=data["choices"][1:Nstim]
    # i_sigma=data["i_sigma"][1:Nstim]

    stim=data["stim_10"]
    stim = permutedims(reshape(stim,(10,n_stim_10)))
    println("1stim ",stim[1,:])
    choices=data["choices_10"]
    i_sigmas = data["sigma_10"]
    Nframes=10




    tau=1.0
    Tframe=0.5
    T=Tframe*Nframes
    dt=0.04
    #NT=T/dt
    #println("T",T)
    #println(dt)
    #println(param_best)
    # stim=data["stim"]
    # choices=data["choices"]
    # i_sigmas=data["i_sigma"]
    Nstim=100
    Ntrials=500
    PR_sim=zeros(Nstim)
    for istim in 1:Nstim
        internal_noise_sigma=param_best["sigma_a"][1]*randn(Ntrials,50)
        stim_rep=repeat(transpose(stim[istim,:]),Ntrials)
        d,xfinal=simulation_Nframe_PI(param_best,stim_rep,internal_noise_sigma,T,tau,model_f)
        PR_sim[istim]=mean( (d+1)/2 )
    end

    save=Dict("PR_sim"=>PR_sim,"PR_fit"=>PR_fit)

    path_save=path*"figures/"
    filename_save=path_save*"check_PR_fit_PR_sim"*model_fit*"_subject"*string(subject)*"_05sec.jld"
    JLD.save(filename_save,save)
    json_string = json_string = JSON.json(save)
	open(path_save*"check_PR_fit_PR_sim"*model_fit*"_subject"*string(subject)*"_05sec.json","w") do f 
		write(f, json_string) 
	print("done")
	end

end

