
using JLD
using JSON
Nproc=4

rmprocs(workers())
addprocs(Nproc)

@everywhere using Base.Test
@everywhere import Base.convert
@everywhere using MAT
@everywhere using ForwardDiff
@everywhere using Polynomials
@everywhere import ForwardDiff.DiffBase
@everywhere using Optim
@everywhere gc(true)

# local
# @everywhere path="/home/genis/fit_behaviour_julia/"
# cluster
@everywhere path="/home/hcli64/hcli64751/rats/fit_behaviour_julia/"

@everywhere include(path*"functions/functions_julia.jl")
@everywhere include(path*"functions/parameter_functions.jl")

@everywhere include(path*"functions/utils.jl")


model_fit="StLogTimeDw"

rat_num = ARGS[1]
#rat_num = "24"
dataset = ARGS[2]
#dataset ="2"

path_load=path*"scripts_rats/data/"
if dataset == "1"
    filename="processed_data_rat"*rat_num*".json"
else
    filename = "processed_data_rat"*rat_num*"_dataset2_1.json"
end

# @everywhere include(path*"scripts/functions/functions_julia.jl")
# @everywhere include(path*"scripts/functions/parameter_functions.jl")
# @everywhere include(path*"scripts/functions/utils.jl")
@everywhere include(path*"/functions/simulations.jl")


# model_fit="PI_StConst"

# isubject=parse(Int,ARGS[1])
# #isubject=1
# SUBJECTS=[1,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19]
# subject=SUBJECTS[isubject]

#
# path_data=path*"data/"
# data=load(path_data*"data_subject"*string(subject)*".jld")

data = Dict()
open(path_load*filename, "r") do f
    global data
    dicttxt = readstring(f)  # file information to string
    data=JSON.parse(dicttxt)  # parse and transform data
end


n_stim_10 = data["n_stim_10"]
n_stim_20 = data["n_stim_20"]


 
choices = data["choices_20"]
i_mu = data["coherences_20"]

#Nstim=100
# stim=data["stim"][1:Nstim,:]
# choices=data["choices"][1:Nstim]
# i_sigma=data["i_sigma"][1:Nstim]
stim=data["stim_20"]
stim = permutedims(reshape(stim,(20,n_stim_20)))
#choices=data["choices_10"]
i_sigma = data["sigma_20"]
# i_sigma=data["i_sigma"]
Nstim=size(stim)[1]


history_bias=zeros(Nstim) #so far no history bias

Nframes=20
Tframe=0.5


mu_vec = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.55, -0.5, -0.45, -0.4, -0.3, -0.2, -0.15, -0.1, 0.0,
 0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0, -0.65, 0.65, -0.35, 0.35]


args=[ ("c2",[1]),("c4",[1]),("st",[1,2,3]),("x0",0),("bias_d",0),("hbias_d",0),("sigma_a",0)]
consts=[("c2",[2]),("c4",0),("st",0),("x0",[1,2]),("bias_d",[1]),
("hbias_d",[1]),("sigma_a",[1])]


z=1.0*[0.0,0.0,0.0,0.0,0.0,1.0]
lb=[-50., 0.0,  0.0,0.0,-50.0 ]
ub=[50., 50.,  50., 50.0,  50.0]


aux=rand(length(lb))
x_ini=[aux[i]*(ub[i]-lb[i])+lb[i] for i in 1:length(ub)]
param2=make_dict(args,x_ini,consts,z)


println(param2)
println(length(lb)," ",length(ub)," ",length(x_ini))
println(model_fit)

model_f=Dict(:c2_f=>c2_urgency,:c4_f=>c4_const,:c6_f=>c6_zeros,:bias_d_f=>bias_d_const,
:hbias_d_f=>history_bias_d_const,:st_f=>st_logistic_time,:x0_f=>initial_condition_bias_hbias,
:ro_f=>read_out_perfect,:lambda=>0.0)
lambda = 0

println("Nstim: ",Nstim)

Nconditions=1
res=ModelFitting_rats(stim,i_sigma,i_mu,mu_vec,choices,history_bias, Tframe,args,consts,ub,lb, x_ini,z,model_f)
j=1
path_save_res=path*"results/model_fit"*model_fit*"/rat"*rat_num*"_1sec/"
mkpath(path_save_res)
filename_save=path_save_res*"icond"*string(j)*".jld"
param2=make_dict(args,res["x_bf"],consts,z)
PR=ProbRight2(stim,history_bias,Tframe,param2,model_f)

while isfile(filename_save)
        j=j+1
        filename_save=path_save_res*"icond"*string(j)*".jld"
end

println(j," ",res["x_bf"])
save(filename_save,"res",res,"PR_training",PR,"icond",j,"args",args,"consts",consts,"z",z,
"param",param2,"lambda",[lambda])
println("done")
