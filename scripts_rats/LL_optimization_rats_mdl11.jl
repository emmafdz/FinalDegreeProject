using JSON
Nproc=4
rmprocs(workers())
addprocs(Nproc)

println("workers: ",length(workers()))


@everywhere using Base.Test
@everywhere import Base.convert
@everywhere using MAT
@everywhere using ForwardDiff
@everywhere using Polynomials
@everywhere import ForwardDiff.DiffBase
@everywhere using Optim
@everywhere gc(true)

# cluster idibaps
# @everywhere path="/home/emma/github/fit_behaviour_julia/"
# cluster
@everywhere path="/home/hcli64/hcli64751/rats/fit_behaviour_julia/"
#
@everywhere include(path*"functions/functions_julia.jl")
@everywhere include(path*"functions/parameter_functions.jl")
@everywhere include(path*"functions/utils.jl")

rat_num = ARGS[1]
# rat_num = "1"
dataset = ARGS[2]
# dataset ="2"

path_load=path*"scripts_rats/data/"
if dataset == "1"
    filename="processed_data_rat"*rat_num*".json"
else
    filename = "processed_data_rat"*rat_num*"dataset2.json"
end


data = Dict()
open(path_load*filename, "r") do f
    global data
    dicttxt = readstring(f)  # file information to string
    data=JSON.parse(dicttxt)  # parse and transform data
end



n_stim_10 = data["n_stim_10"]
n_stim_20 = data["n_stim_20"]

stim_10 = data["stim_10"]
stim_20 = data["stim_20"]
start = 10000
n_stim = 10


mu_vec = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.55, -0.5, -0.45, -0.4, -0.3, -0.2, -0.15, -0.1, 0.0,
 0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0, -0.65, 0.65, -0.35, 0.35]


model_num="11"

args=[("c2",[1]),("c4",0),("st",[1,2]),("x0",[1]),("bias_d",[1]),("hbias_d",0),("sigma_a",[1])]
consts=[("c2",[2]),("c4",[1]),("st",[3,4]),("x0",[2]),("bias_d",0),("hbias_d",[1]),("sigma_a",0)]

z = [0,1,0,0,0,0]

lb=[-80.,0.0,0.0,-80.,-80,0.1]
ub=[80.,100.,100.,80.,80.,80.]

model_f=Dict(:c2_f=>c2_urgency,:c4_f=>c4_const,:c6_f=>c6_zeros,:bias_d_f=>bias_d_const,
:hbias_d_f=>history_bias_d_const,:st_f=>st_logistic_bias2,:x0_f=>initial_condition_bias_hbias,
:ro_f=>read_out_perfect,:lambda=>0.0)


aux=2*(rand(6)-0.5)
x_ini=[ aux[1]*80.,abs(aux[2]*100.),abs(aux[3]*100.),aux[4]*80.,aux[5]*80.,rand()*79.9+0.1]
println("hello")

if n_stim_10 != 0
    if n_stim_10 > 10000
        n_stim = 10000
    else
        n_stim = n_stim_10
    end
    history_bias=zeros(n_stim_10)[1:n_stim,:]
    Tframe = 0.05
    stim_10 = permutedims(reshape(stim_10,(10,n_stim_10)))[1:n_stim,:]
    choices_10 = data["choices_10"][1:n_stim,:]
    mu_10 = data["coherences_10"][1:n_stim,:]
    sigma_10 = data["sigma_10"][1:n_stim,:]
    fit_10 = ModelFitting_rats(stim_10,sigma_10,mu_10,mu_vec,choices_10,history_bias, Tframe, args,consts,ub,lb, x_ini,z,model_f)
    param_10=make_dict(args,fit_10["x_bf"],consts,z)
    ## SAVE THE RESULTS ##
    j=1
    path_save_res=path*"results/fit"*dataset*"/Rat"*rat_num*"_10/"
    mkpath(path_save_res)
    filename_save=path_save_res*"icond"*string(j)*".json"

    while isfile(filename_save)
            j=j+1
            filename_save=path_save_res*"icond"*string(j)*".json"
    end
    tic()
    results = Dict("res"=>fit_10,"icond"=>j,"param"=>param_10)
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

end

if n_stim_20 != 0
    if n_stim_20 > 10000
        n_stim = 10000
    else
        n_stim = n_stim_20
    end
    history_bias=zeros(n_stim_20)[1:n_stim,:]
    Tframe = 0.05
    stim_20 = permutedims(reshape(stim_20,(20,n_stim_20)))[1:n_stim,:]
    choices_20 = data["choices_20"][1:n_stim,:]
    mu_20 = data["coherences_20"][1:n_stim,:]
    sigma_20 = data["sigma_20"][1:n_stim,:]
    fit_20 = ModelFitting_rats(stim_20,sigma_20,mu_20,mu_vec,choices_20,history_bias, Tframe, args,consts,ub,lb, x_ini,z,model_f)
    param_20=make_dict(args,fit_20["x_bf"],consts,z)
    ## SAVE THE RESULTS ##
    j=1
    path_save_res=path*"results/fit"*dataset*"/Rat"*rat_num*"_20/"
    mkpath(path_save_res)
    filename_save=path_save_res*"icond"*string(j)*".json"

    while isfile(filename_save)
            j=j+1
            filename_save=path_save_res*"icond"*string(j)*".json"
    end
    tic()
    results = Dict("res"=>fit_20,"icond"=>j,"param"=>param_20)
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
print("here")



## FINISHED ##

println("done")
