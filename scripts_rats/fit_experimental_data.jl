using JSON
Nproc = 10

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


# local
#@everywhere path="/home/emma/github/fit_behaviour_julia/"

#cluster
@everywhere path="/home/emma/fit_behaviour_julia/"


@everywhere path_functions=path*"functions/"
@everywhere path_scripts=path*"scripts_emma/"

@everywhere include(path_functions*"parameter_functions.jl")
@everywhere include(path_functions*"functions_julia.jl")
@everywhere include(path_functions*"utils.jl")

##############################
Nframes=10
Tframe=0.2
Nstim= 1000


model_num="8"

# Model only c2 and sigma
args=[ ("c2",[1]),("c4",0),("st",0),("x0",0),("bias_d",0),("hbias_d",0),("sigma_a",[1])]
consts=[("c2",[2]),("c4",[1]),("st",[1,2,3,4]),("x0",[1,2]),("bias_d",[1]),("hbias_d",[1]),("sigma_a",0)]


x = [0.6,0.4]
z = [0,1,2,2,0,0,0,0,0,0]
param=make_dict(args,x,consts,z)

ub=[10., 10.]
lb=[-10., -10]
reg=[1,1]

model_f=Dict(:c2_f=>c2_urgency,:c4_f=>c4_const,:c6_f=>c6_zeros,:bias_d_f=>bias_d_const,
:hbias_d_f=>history_bias_d_const,:st_f=>st_logistic_bias2,:x0_f=>initial_condition_bias_hbias,
:ro_f=>read_out_perfect,:lambda=>0.0)

data = Dict()
open(path*"processed_data.json", "r") do f
    global data
    dicttxt = readstring(f)  # file information to string
    data=JSON.parse(dicttxt)  # parse and transform data
end

# print both dictionaries
# println(dict2)
stimulus = data[1]             # Data frames obtained from a gaussian distribution of a determined coherence
coherences = data[2]           # Target coherences used for each trial
rewards = data[3]              # Correct choice side
decisions = data[4]            # Actual decision made by the rat each trial
performances = data[5]         # 0 if the rat chose the correct choice, 1 if the rat chose the incorrect choice
target_sigmas = data[6]        # Target sigmas used for each trial

#one session
stim = zeros(size(stimulus[1][1])[1],size(stimulus[1][1][1])[1])
for s in size(stimulus[1][1])
    for i in size(stimulus[1][1][1])
        stim[s,i] = stimulus[1][1][s][i]
    end
end
println(size(stim))
choices = decisions[1][1]


##############################
# Per 30000 trials fer synthetic data i calcular com de bé calcula els parametres
Nframes=size(stim[1])
Tframe=0.2
Nstim= size(stim)

history_bias=zeros(Nstim)

PR = ProbRight2(stim,history_bias,Tframe,param,model_f)
choices = []
for prob in PR
  if rand() < prob
    append!(choices,1)
  else
    append!(choices,-1)
  end
end

# Call function ComputeLL2 to create a MAtrix of LL parametrically chanching two parameters
# The maximum LL should be for the genereative parameters (the parameters that we use to create the data)

modl = Dict()
for i in 1:100
    println(i)
    x = rand(1:10,2)
    x = x/10
    modl_f = ModelFitting(stim,choices,history_bias, Tframe, args,consts,ub,lb, x,z,model_f)
    modl[string(i)] = modl_f
end

## SAVE THE RESULTS ##
stringdata = JSON.json(modl)

# write the file with the stringdata variable information
open("datafit_rat_2par.json", "w") do f
        write(f, stringdata)
     end
println("done")
