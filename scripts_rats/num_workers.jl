using JSON
Nproc=16
time_LL_grad=zeros(Nproc)
time_LL=zeros(Nproc)
time_PR=zeros(Nproc)


for i in 1:Nproc
    rmprocs(workers())
    addprocs(i)
    println("workers: ",length(workers()))
    @everywhere import Base.convert
    @everywhere using MAT
    @everywhere using ForwardDiff
    @everywhere using Optim
    @everywhere using Polynomials
    @everywhere using JSON
    @everywhere model_num="10"

    @everywhere Nstim = 80000

    @everywhere path="/home/hcli64/hcli64751/rats/fit_behaviour_julia/"
    #
    @everywhere include(path*"functions/functions_julia.jl")
    @everywhere include(path*"functions/parameter_functions.jl")
    @everywhere include(path*"functions/utils.jl")
    @everywhere rat_num = "1"
    @everywhere path_load=path*"scripts_rats/data/"
    @everywhere filename="processed_data_rat"*rat_num*".json"

    @everywhere data = Dict()
    open(path_load*filename, "r") do f
        global data
        dicttxt = readstring(f)  # file information to string
        data=JSON.parse(dicttxt)  # parse and transform data
    end

    stim_10 = data["stim_10"][1:Nstim,:]
    choices_10 = data["choices_10"][1:Nstim,:]

    @everywhere history_bias=zeros(Nstim)
    ###### check the gradient ##########

    @everywhere Nframes=10
    @everywhere Tframe=.05
    model_num="10"

    @everywhere args=[("c2",[1]),("c4",0),("st",[1,2]),("x0",0),("bias_d",0),("hbias_d",0),("sigma_a",[1])]
    @everywhere consts=[("c2",[2]),("c4",[1]),("st",[3,4]),("x0",[1,2]),("bias_d",[1]),("hbias_d",[1]),("sigma_a",0)]

    @everywhere z = [0,1,0,0,0,0,0,0]

    @everywhere lb=[-10.,0.0,0.0,0.1]
    @everywhere ub=[10.,10,10,10.]

    @everywhere model_f=Dict(:c2_f=>c2_urgency,:c4_f=>c4_const,:c6_f=>c6_zeros,:bias_d_f=>bias_d_const,
    :hbias_d_f=>history_bias_d_const,:st_f=>st_logistic_bias2,:x0_f=>initial_condition_bias_hbias,
    :ro_f=>read_out_perfect,:lambda=>0.0)

    x_ini=[ 0.6,4.72,6.38,1.52]
    LL = ComputeLL2(stim_10,choices_10,history_bias,Tframe, args, x_ini,consts,z,model_f)
    ll_t,grad_t=LL_function_grads(stim_10,choices_10,history_bias,Tframe, args, x_ini,consts,z,model_f)
    time_LL[i]=@elapsed ComputeLL2(stim_10,choices_10,history_bias,Tframe, args, x_ini,consts,z,model_f)
    time_LL_grad[i]=@elapsed ll_t,grad_t=LL_function_grads(stim_10,choices_10,history_bias,Tframe, args, x_ini,consts,z,model_f)
    param=make_dict(args,x_ini,consts,z)
    time_PR[i] = @elapsed PR=ProbRight2(stim_10,history_bias,Tframe,param,model_f)

end
@everywhere path_save="/home/hcli64/hcli64751/rats/fit_behaviour_julia/results/"
mkpath(path_save)
results = Dict("time_LL"=>time_LL,"time_LL_grad"=>time_LL_grad,"time_PR"=>time_PR,"workers"=>1:Nproc,"Nstim"=>Nstim)
# pass data as a json string (how it shall be displayed in a file)
stringdata = JSON.json(results)
# write the file with the stringdata variable information
open(path_save*"number_workers.json", "w") do f
        write(f, stringdata)
     end
println("done")
