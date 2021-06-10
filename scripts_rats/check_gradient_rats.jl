using JLD
using Base.Test
import Base.convert
using MAT
using ForwardDiff
using Polynomials
import ForwardDiff.DiffBase
using Optim
# using Pandas
using JSON

# @everywhere path="/home/hcli64/hcli64751/rats/fit_behaviour_julia/"
@everywhere path="/home/emma/github/fit_behaviour_julia/"

#
@everywhere include(path*"functions/functions_julia.jl")
@everywhere include(path*"functions/parameter_functions.jl")
@everywhere include(path*"functions/utils.jl")
@everywhere rat_num = "1"
@everywhere path_load=path*"scripts_rats/data/"
@everywhere filename="processed_data_rat"*rat_num*".json"



# local
# @everywhere path="/home/genis/fit_behaviour_julia/"
#######bsc#########
#
# @everywhere path_functions=path*"scripts/functions/"
# @everywhere path_data=path*"data/"
# @everywhere path_results=path*"results/"
#
# @everywhere include(path_functions*"parameter_functions.jl")
# @everywhere include(path_functions*"functions_julia.jl")
# @everywhere include(path_functions*"utils.jl")

Nframes=10
Tframe=.05

model_num="6"

@everywhere args=[("c2",[1]),("c4",0),("st",[1,2]),("x0",0),("bias_d",0),("hbias_d",0),("sigma_a",[1])]
@everywhere consts=[("c2",[2]),("c4",[1]),("st",[3,4]),("x0",[1,2]),("bias_d",[1]),("hbias_d",[1]),("sigma_a",0)]

@everywhere z = [0,1,0,0,0,0,0,0]

@everywhere lb=[-10.,0.0,0.0,0.1]
@everywhere ub=[10.,10,10,10.]


@everywhere model_f=Dict(:c2_f=>c2_urgency,:c4_f=>c4_const,:c6_f=>c6_zeros,:bias_d_f=>bias_d_const,
:hbias_d_f=>history_bias_d_const,:st_f=>st_logistic_bias2,:x0_f=>initial_condition_bias_hbias,
:ro_f=>read_out_perfect,:lambda=>0.0)

x_ini=[ 0.6,4.72,6.38,1.52]
@everywhere using JSON
@everywhere model_num="10"

@everywhere Nstim = 100

@everywhere data = Dict()
open(path_load*filename, "r") do f
    global data
    dicttxt = readstring(f)  # file information to string
    data=JSON.parse(dicttxt)  # parse and transform data
end

choices_10 = data["choices_10"][200+1:200+Nstim,:]
stim_10 = data["stim_10"][200+1:100+Nstim,:]
mu_vec = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.55, -0.5, -0.45, -0.4, -0.3, -0.2, -0.15, -0.1, 0.0,
 0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0, -0.65, 0.65, -0.35, 0.35]
mu_10 = data["coherences_10"][200+1:100+Nstim,:]
sigma_10 = data["sigma_10"][200+1:100+Nstim,:]

@everywhere history_bias=zeros(Nstim)
###### check the gradient ##########

#
# println("computeLL")
# tic()
# PR=ProbRight2(stim,history_bias,Tframe,param2,model_f)
# toc()


println("computeLL2")
tic()
ll= ComputeLL2_mu(stim_10,sigma_10,mu_10,mu_vec,choices_10,history_bias,Tframe, args, x_ini,consts,z,model_f)
toc()


println("grads")
tic()
ll_1,grads=LL_function_grads_mu(stim_10,sigma_10,mu_10,mu_vec,choices_10,history_bias,Tframe, args, x_ini,consts,z,model_f)
toc()


grad_num=zeros(length(x))
hess_num=zeros(length(x))

dxx=0.001
println("grad num")


tic()
for iparam in 1:length(x)
        # println(iparam)
        y=x[:]
        # print(y[iparam]+dxx)
        # y[iparam]=y[iparam]+dxx
        lldx=LL_original=ComputeLL2_mu(stim_10,sigma_10,mu_10,mu_vec,choices_10,history_bias,Tframe, args, x_ini,consts,z,model_f)
        grad_num[iparam]=(lldx-ll)/dxx

        #y[iparam]=y[iparam]-2*dxx
        #lldx_m=LL_original=ComputeLL2(stim,choices,history_bias,Tframe, args, y,consts,z,model_f)
        #hess_num[iparam]=(lldx-2.*ll+lldx_m)/(dxx^2)
end
toc()
