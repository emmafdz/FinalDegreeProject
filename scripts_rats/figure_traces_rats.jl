
#3rd party
import Base.convert
using MAT
using ForwardDiff
using Optim
using Polynomials
using JLD
using PyPlot

# path_functions="/home/genis/model_fitting_julia/scripts/functions/"
path_functions="/home/genis/fit_behaviour_julia/scripts/functions/"

include(path_functions*"parameter_functions.jl")
include(path_functions*"functions_julia.jl")
include(path_functions*"utils.jl")

model_num="6"

Nstim=50
# path_data="/home/genis/model_fitting_julia/synthetic_data/"
path_data="/home/genis/fit_behaviour_julia/synthetic_data/"
# path="/home/genis/model_fitting_julia/"
# filename="synthetic_data_DW_Nstim"*string(Nstim)*"_5sigmas_test_largeParam.jld"
# filename="synthetic_data_DW_Nstim"*string(Nstim)*"_5sigmas_urgency_SLogisticTransform.jld"
filename="synthetic_data_DW_Nstim"*string(Nstim)*"_model"*model_num*"_traces.jld"

data=load(path_data*filename)



# save_dict=Dict(["d"=>d,"stim"=>stim,"param"=>param,"sigma_a"=>sigma_a,"Tframe"=>T,"PR"=>PR_sim,"kernel"=>kernel_sigma, "PR_sim_test"=>PR_sim_test, "stim_test"=>stim_test,"d_test"=>d_test])
# save(path_save*filename,save_dict)
stim=data["stim"]
x_traces_sim=data["x_trace"]
Tframe=data["Tframe"]
param=data["param"]
Nstim,Ntraces,NT=size(x_traces_sim)
model_num="6"
# args=[ ("c2",2),"c4",("st",4),("x0",2),"bias_d","hbias_d","sigma_a"]
# #   c21 c22  c4  st1  st2 st3  st4   x0  bias_d sigma
# ub=[10., 10.,10., 10.,10., 10., 10.,10, 10, 10.,10,10.]
# #   c21  c22  c4    st1  st2  st3  st4    x0  bias_d  sigma
# lb=[-10.,-10,-10 ,0.,  0., -10, -10.,-10,-10,-10,-10., 0]
lambda=0
reg=[1,1,1,1,1,1,1,1,1,1,1,1]
model_f=Dict(:c2_f=>c2_urgency,:c4_f=>c4_const,:c6_f=>c6_zeros,:bias_d_f=>bias_d_const,
:hbias_d_f=>history_bias_d_const,:st_f=>st_logistic_bias2,:x0_f=>initial_condition_bias_hbias,
:ro_f=>read_out_perfect,:lambda=>lambda,:reg=>reg)

# param=Dict("c2"=>[1,0],"c4"=>[0.5],"x0"=>[0.1,0.05], "st"=>[1.,3,0.1,0.2],"bias_d"=>[0.1],
# "hbias_d"=>[0.2],"sigma_a"=>0.3)

fig,axes=subplots(5,2)
i_stim=[1,2,3,4,5]
T=Tframe*length(stim[1,:])
t=0:T/(length(x_traces_sim[1,1,:])-1):T
for i in 1:length(i_stim)
    figure()
    x,bins=pdf_x_time_1stim(stim[i_stim[i],:],[0.],Tframe,param,model_f)
    imshow(transpose(x),origin="lower",extent=[0,T,bins[1],bins[end]],aspect="auto",vmax=0.1,cmap="hot")



    for itrial in 1:Ntraces
        plot(t,x_traces_sim[i_stim[i],itrial,:],"-",color="darkcyan")
    end

    figure()
    plot(stim[i_stim[i],:])

end
show()
