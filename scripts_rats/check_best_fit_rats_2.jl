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


SUBJECTS=[1]

subject=6
# filename="synthetic_data_DW_Nstim10000.jld"
model_fit="StLinearDw_dx"

Tframe=0.05
#
model_f=Dict(:c2_f=>c2_urgency,:c4_f=>c4_const,:c6_f=>c6_zeros,:bias_d_f=>bias_d_const,
:hbias_d_f=>history_bias_d_const,:st_f=>st_linear,:x0_f=>initial_condition_bias_hbias,
:ro_f=>read_out_perfect,:lambda=>1.0)
#
# model_f=Dict(:bias_d_f=>bias_d_const,
# :hbias_d_f=>history_bias_d_const,:st_f=>st_linear,:x0_f=>initial_condition_bias_hbias,
# :ro_f=>read_out_perfect,:lambda=>1.0)
#
#args=[ ("st",[1]),("x0",0),("bias_d",0),("hbias_d",0),("sigma_a",0)]
# consts=[("st",[2]),("x0",[1,2]),("bias_d",[1]),
# ("hbias_d",[1]),("sigma_a",[1])]


###### PARAM simulations ######

tau=1.0
Tframe=0.05
#############################


subject=1

for subject in SUBJECTS
#subject=4
    path_load=path*"results/summary/model_fit"*model_fit*"/"
    filename_load=path_load*"rat"*string(subject)*".jld"
    data_fit=load(filename_load)

    iminimum=find(x->x==minimum(data_fit["LL_training"]),data_fit["LL_training"])
    param_best=data_fit["param_all"][iminimum][1]
    #println("param_best",param_best)
  #  param_best = Dict("c2"=> [10, 0.0],"st"=> [1, 0.0, 0.0, 0.0],"sigma_a"=> [1.0],"x0"=> [0.0, 0.0],"bias_d"=> [0.0],"c4"=>[1],"hbias_d"=> [0.0])
    println("param_best",param_best)
    PR_fit=data_fit["PR_training"][iminimum][1]
    path_data=path*"scripts_rats/data/"

    # data = Dict()
    open(path_data*"processed_data_rat"*string(subject)*"_dataset2.json", "r") do f
        global data
        dicttxt = readstring(f)  # file information to string
        data=JSON.parse(dicttxt)  # parse and transform data
    end
    println(keys(data))
    n_stim_10 = data["n_stim_10"]
    n_stim_20 = data["n_stim_20"]


    #Nstim=100
    # stim=data["stim"][1:Nstim,:]
    # choices=data["choices"][1:Nstim]
    # i_sigma=data["i_sigma"][1:Nstim]
    if n_stim_10 != 0
        stim=data["stim_10"]
        stim = permutedims(reshape(stim,(10,n_stim_10)))
        choices=data["choices_10"]
        i_sigmas = data["sigma_10"]
        Nframes=10

    else
        stim=data["stim_20"]
        stim = permutedims(reshape(stim,(20,n_stim_20)))
        choices=data["choices_20"]
        i_sigmas = data["sigma_20"]
        Nframes=20

    end


    tau=1.0
    Tframe=0.05
    T=Tframe*Nframes
    dt = 0.01
    NT=T/dt
    println("T",T)
    println(dt)
    println(param_best)
    # stim=data["stim"]
    # choices=data["choices"]
    # i_sigmas=data["i_sigma"]
    Nstim=1
    Ntrials=50
    PR_sim=zeros(Nstim)
	

    internal_noise_sigma=param_best["sigma_a"][1]*randn(Ntrials,Int(NT))
    
#    stim_rep=repeat(transpose(stim[1,:]),Ntrials)
	stim_rep = stim[50,:]
	println("stim",stim_rep)
    print(size(stim_rep))
    history_bias = [0,0,0,0,0,0,0,0,0]
    
    x,bins = pdf_x_time_1stim(stim_rep,Tframe,param_best,model_f,history_bias;B=0)

    print("he fet això")


    save=Dict("x"=>x,"bins"=>bins)

    path_save=path*"figures/"

    json_string = json_string = JSON.json(save)
	open(path_save*"one_pdf.json","w") do f
		write(f, json_string)
	print("done")
	end

end
