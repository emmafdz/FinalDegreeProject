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


model_num=""

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

sigma_s = 0.2
stim = sigma_s*randn(Nstim,Nframes)


history_bias=zeros(Nstim)

tic()
PR=ProbRight2(stim,history_bias,Tframe,param,model_f)
toc()

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
# Afegir guardar cada vegada.
modl = Dict()
for i in 1:100
    println(i)
    x = rand(1:10,2)
    x = x/10
    modl_f = ModelFitting(stim,choices,history_bias, Tframe, args,consts,ub,lb, x,z,model_f)
    modl[string(i)] = modl_f
    stringdata = JSON.json(modl_f)

    # write the file with the stringdata variable information
    open("fit_2par_cond"*string(i)*".json", "w") do f
            write(f, stringdata)
         end
end

## SAVE THE RESULTS ##
# pass data as a json string (how it shall be displayed in a file)
stringdata = JSON.json(modl)

# write the file with the stringdata variable information
open("fit_2par.json", "w") do f
        write(f, stringdata)
     end
println("done")
