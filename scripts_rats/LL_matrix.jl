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
Nstim= 10000


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

save_data = Dict()
final_matrix = zeros(20,20)
println(size(final_matrix))
for c2_p in 1:1:20
    for sig in 1:1:20
        x=[c2_p/20,sig/20]
        LL =  ComputeLL2(stim,choices,history_bias,Tframe, args, x,consts,z,model_f)
        final_matrix[floor(Int8,c2_p),floor(Int8,sig)]= LL
    end
end
save_data["matrix"]=final_matrix


# Buscar el mínim i fer la hessiana en el mínim
x = [0.6,0.4]
f,grads,Hess = LL_function_grads(stim,choices,history_bias,Tframe, args, x,consts,z,model_f)
save_data["hess"] = Hess

## SAVE THE RESULTS ##
# pass data as a json string (how it shall be displayed in a file)
stringdata = JSON.json(save_data)
# write the file with the stringdata variable information
open("LL_2par_matrix.json", "w") do f
        write(f, stringdata)
    end
println("done")
