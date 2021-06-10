
using JSON
using JLD
# Nproc=4
#
# rmprocs(workers())
# addprocs(Nproc)

@everywhere using Base.Test
@everywhere import Base.convert
@everywhere using MAT
@everywhere using ForwardDiff
@everywhere using Polynomials
@everywhere import ForwardDiff.DiffBase
@everywhere using Optim
@everywhere gc(true)

# local
@everywhere path="/home/emma/fit_behaviour_julia/"
# cluster
# @everywhere path="/home/hcli64/hcli64751/rats/fit_behaviour_julia/"
#
@everywhere include(path*"functions/functions_julia.jl")
@everywhere include(path*"functions/parameter_functions.jl")
@everywhere include(path*"functions/utils.jl")

rat_num = ARGS[1]
#rat_num = "24"
#dataset = ARGS[2]
dataset ="2"

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



#Nstim=100
# stim=data["stim"][1:Nstim,:]
# choices=data["choices"][1:Nstim]
# i_sigma=data["i_sigma"][1:Nstim]
stim=data["stim_20"]
stim = permutedims(reshape(stim,(20,n_stim_20)))
choices = data["choices_20"]

# i_sigma=data["i_sigma"]
Nstim=size(stim)[1]


history_bias=zeros(Nstim) #so far no history bias

Nframes=20
Tframe=0.5

model_fit = "StLinPi"
model_f=Dict(:bias_d_f=>bias_d_const,:hbias_d_f=>history_bias_d_const,
        :st_f=>st_linear,:x0_f=>initial_condition_bias_hbias,
        :ro_f=>read_out_perfect,:lambda=>0.0)


args=[ ("st",[1]),("x0",0),("bias_d",0),("hbias_d",0),("sigma_a",0)]
consts=[("st",[2]),("x0",[1,2]),("bias_d",[1]),
("hbias_d",[1]),("sigma_a",[1])]

z=1.0*[0.0,0.,0.0,0.0,0,1.0]

ub=[50.]
lb=[0.0]



x_ini=zeros(length(ub))
# model_f=Dict(:bias_d_f=>bias_d_const,:hbias_d_f=>history_bias_d_const,
# :st_f=>st_const,:x0_f=>initial_condition_bias_hbias,
# :ro_f=>read_out_perfect,:lambda=>0.0)
#
#
# args=[ ("st",[1]),("x0",0),("bias_d",0),("sigma_a",0)]
# consts=[("st",0),("x0",[1,2]),("bias_d",[1]),("sigma_a",[1])]
#
# z=1.0*[0.0,0.0,0.0,1.0]
# #
# ub=[10.,10.]
# lb=[0.0,0.0]

for i in 1:300
        aux=2*(rand(8)-0.5)
        x_ini=[abs(aux[1])*10]
        #x_ini = [abs(aux[1])]
        param=make_dict(args,x_ini,consts,z)
        #PR=ProbRightPI(stim,history_bias,Tframe,param,model_f)
        #ll=ComputeLLPI(stim,choices,history_bias,Tframe,args,consts,x,z,model_f)
        control=true
        try
                global res=ModelFittingPI(stim,choices,history_bias, Tframe, args,consts,ub,lb, x_ini,z,model_f)
                println("Fit_done")

        catch
                println("something wrong fitting the model icond",i)
                control=false
        end

        if control
                j=1
                path_save_res=path*"results/model_fit"*model_fit*"/rat"*string(rat_num)*"_1sec/"
                mkpath(path_save_res)
                filename_save=path_save_res*"icond"*string(j)*".jld"
                param2=make_dict(args,res["x_bf"],consts,z)

                PR=ProbRightPI(stim,history_bias,Tframe,param2,model_f)

                while isfile(filename_save)
                        j=j+1
                        filename_save=path_save_res*"icond"*string(j)*".jld"
                end

                println(j," ",res["x_bf"])
                save(filename_save,"res",res,"PR_training",PR,"icond",j,"args",args,"consts",consts,"z",z,
                "param",param2)
                println("done")
        end
end
