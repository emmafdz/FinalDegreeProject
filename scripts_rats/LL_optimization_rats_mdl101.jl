using JSON
Nproc=4
rmprocs(workers())
addprocs(Nproc)

println("workers: ",length(workers()))


@everywhere using Base.Test
# @everywhere import Base.convert
@everywhere using MAT
@everywhere using ForwardDiff
@everywhere using Polynomials
@everywhere import ForwardDiff.DiffBase
@everywhere using Optim
@everywhere gc(true)

# cluster idibaps
@everywhere path="/home/emma/github/fit_behaviour_julia/"
# cluster
# @everywhere path="/home/hcli64/hcli64751/rats/fit_behaviour_julia/"
#
@everywhere include(path*"functions/functions_julia.jl")
@everywhere include(path*"functions/parameter_functions.jl")
@everywhere include(path*"functions/utils.jl")

# rat_num = ARGS[1]
rat_num = "1"
path_load=path*"scripts_rats/data/"
filename="processed_data_rat"*rat_num*".json"
filename_1 = "set_trials_10.json"
data = Dict()
open(path_load*filename, "r") do f
    global data
    dicttxt = readstring(f)  # file information to string
    data=JSON.parse(dicttxt)  # parse and transform data
end
data_1 = Dict()
open(path_load*filename_1, "r") do f
    global data_1
    dicttxt = readstring(f)  # file information to string
    data_1=JSON.parse(dicttxt)  # parse and transform data
end

dataset = data_1[parse(Int64,rat_num)]
n_stim = size(dataset)[1]
stim_10 = data["stim_10"]
stim_20 = data["stim_20"]
choices_10 = data["choices_10"]
choices_20 = data["choices_20"]

n_stim_10 = data["n_stim_10"]
n_stim_20 = data["n_stim_20"]


# if you want to try with less n_stim
# n_stim_10 = 10000
# n_stim_20 = 10000

model_num="10"

args=[("c2",[1]),("c4",0),("st",[1,2]),("x0",0),("bias_d",0),("hbias_d",0),("sigma_a",[1])]
consts=[("c2",[2]),("c4",[1]),("st",[3,4]),("x0",[1,2]),("bias_d",[1]),("hbias_d",[1]),("sigma_a",0)]

z = [0,1,0,0,0,0,0,0]

lb=[-10.,0.0,0.0,0.1]
ub=[20.,20.,20.,20.]

model_f=Dict(:c2_f=>c2_urgency,:c4_f=>c4_const,:c6_f=>c6_zeros,:bias_d_f=>bias_d_const,
:hbias_d_f=>history_bias_d_const,:st_f=>st_logistic_bias2,:x0_f=>initial_condition_bias_hbias,
:ro_f=>read_out_perfect,:lambda=>0.0)

aux=2*(rand(4)-0.5)
x_ini=[ aux[1]*10.,abs(aux[2]*10.),abs(aux[3]*10.),rand()*9.9+0.1]



rats_sec = ["2","3","5"]
rats_half = ["1","4"]
tic()
mu_vec = [-1,-0.4,-0.1,0.1,0.4,1]
if n_stim_10 != 0
    history_bias=zeros(n_stim)
    Tframe = 0.05
    stim_10 = permutedims(reshape(stim_10,(10,n_stim_10)))[1:10,:]
    choices_10 = data["choices_10"][1:10,:]
    # mu_10 = data["coherences_10"][1:10,:]
    # sigma_10 = data["sigma_10"][1:10,:]
    sigma_10 = [0,1,2,3,4,5,6,7,8,9]
    fit = ModelFitting(stim_10,sigma_10,choices_10,history_bias, Tframe, args,consts,ub,lb, x_ini,z,model_f)
end
if n_stim_20 != 0
    history_bias=zeros(n_stim)
    Tframe = 0.05
    stim_20 = permutedims(reshape(stim_20,(20,n_stim_20)))[1:10,:]
    choices_20 = data["choices_20"][1:10,:]
    mu_20 = data["coherences_20"][1:10,:]
    sigma_20 = data["sigma_20"][1:10,:]
    fit = ModelFitting_rats(stim_20,sigma_20,mu_20,mu_vec,choices_20,history_bias, Tframe, args,consts,ub,lb, x_ini,z,model_f)
end
# else # la rata 2 té trials tant de 10 com de 20 frames COMPROBAR SI ALGUNA RATA MÉS
#     if ARGS[2] == "10"
#         Nstim_train = trunc(Int,n_stim_10*0.8)
#         stim_train = stim_10[1:Nstim_train,:]
#         stim_test = stim_10[Nstim_train+1:n_stim_10,:]
#         choices_train = choices_10[1:Nstim_train,:]
#         history_bias=zeros(Nstim_train)
#         history_bias_test=zeros(n_stim_10-Nstim_train)
#         Tframe = 10.
#         fit = ModelFitting(stim_train,choices_train,history_bias, Tframe, args,consts,ub,lb, x_ini,z,model_f)
#     elseif ARGS[2] == "20"
#         Nstim_train = trunc(Int,n_stim_10*0.8)
#         stim_train = stim_20[1:Nstim_train,:]
#         stim_test = stim_20[Nstim_train+1:n_stim_20,:]
#         choices_train = choices_20[1:Nstim_train,:]
#         history_bias=zeros(Nstim_train)
#         history_bias_test=zeros(n_stim_20-Nstim_train)
#         Tframe = 20.
#         fit = ModelFitting(stim_train,choices_train,history_bias, Tframe, args,consts,ub,lb, x_ini,z,model_f)
#     end
# end
time_fit = toc()
println("time to fit: "*string(time_fit))
param=make_dict(args,fit["x_bf"],consts,z)


## SAVE THE RESULTS ##
j=1
path_save_res=path*"results/2model10C2St1St2Sig/Rat"*rat_num*"/"
mkpath(path_save_res)
filename_save=path_save_res*"icond"*string(j)*".json"

while isfile(filename_save)
        j=j+1
        filename_save=path_save_res*"icond"*string(j)*".json"
end
tic()
results = Dict("res"=>fit,"icond"=>j,"param"=>param)
time_results = toc()
println("time to dict: "*string(time_results))

tic()
# pass data as a json string (how it shall be displayed in a file)
stringdata = JSON.json(results)

# write the file with the stringdata variable information
open(filename_save, "w") do f
        write(f, stringdata)
     end
time_json = toc()
println("time_to_json: "*string(time_json))

## FINISHED ##

println("done")
