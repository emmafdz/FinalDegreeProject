import Base.convert
using MAT
using ForwardDiff
using Optim
using Polynomials
using JLD
using ROCAnalysis
using PyPlot
import Base.convert
using MAT
using ForwardDiff
using Optim
using Polynomials
using JLD

@everywhere path="/home/emma/github/fit_behaviour_julia/"

path_functions=path*"functions/"
path_save=path*"results/"

include(path_functions*"parameter_functions.jl")
include(path_functions*"functions_julia.jl")
include(path_functions*"simulations.jl")



Nstim=7000
Nframes=10
Tframe=.2
T=Tframe*Nframes
tau=1   #### this is always 1 in the theory.
dt_sim=tau/100.
NT=T/dt_sim
NTframes=NT/Nframes
sigma_a=.3


# model_f=Dict(:c2_f=>c2_urgency,:c4_f=>c4_const,:st_f=>st_logistic_time)
# param=Dict("c2"=>[0.7,0.05], "st"=>[2.0,0.5,0.5],"c4"=>[1.0],"sigma_a"=>sigma_a)

# model_num="2"
# model_f=Dict(:c2_f=>c2_urgency,:c4_f=>c4_const,:st_f=>st_logistic_time,:ro_f=>read_out_soft_sim)
# param=Dict("c2"=>[0.7,0.05], "st"=>[2.0,0.5,0.05],"c4"=>[1.0],"sigma_a"=>sigma_a,"ro"=>[3,0.3])


# model_num="3"
# model_f=Dict(:c2_f=>c2_urgency,:c4_f=>c4_const,:c6_f=>c6_const,:st_f=>st_const,:ro_f=>read_out_TW_sim)
# param=Dict("c2"=>[-0.025,1],"c4"=>[-0.5],"c6"=>[0.4], "st"=>[1.],"sigma_a"=>sigma_a)

model_num="6"
model_f=Dict(:c2_f=>c2_urgency,:c4_f=>c4_const,:c6_f=>c6_zeros,:bias_d_f=>bias_d_const,
:hbias_d_f=>history_bias_d_const,:st_f=>st_logistic_bias2,:x0_f=>initial_condition_bias_hbias,
:ro_f=>read_out_perfect_sim)

param=Dict("c2"=>[1,0],"c4"=>[1],"x0"=>[0.0,0.0], "st"=>[2.,2,0.,0.],"bias_d"=>[0.],
"hbias_d"=>[0.0],"sigma_a"=>sigma_a)


Nsigmas=5
sigma_max=1


sigmas=exp10.(linspace(-1.17,0.,5))
stim=randn(Nstim,Nframes)
stim_test=randn(Nstim,Nframes)

internal_noise=sigma_a*randn(Nstim,Int(NT))
internal_noise_test=sigma_a*randn(Nstim,Int(NT))
stim_trans=zeros(Nstim,Nframes)

s=zeros(Nframes)

for istim in 1:Nstim
    stim[istim,:]=sigmas[(istim-1)%Nsigmas+1]*stim[istim,:]
    stim_test[istim,:]=sigmas[(istim-1)%Nsigmas+1]*stim_test[istim,:]
end

for istim in 1:Nstim
    # stim_trans[istim,:]=st_logistic_time(stim[istim,:],stim_trans[istim,:],param["st"])
    #st_logistic_time(stim[istim,:],s,param["st"])
    model_f[:st_f](stim[istim,:],s,param["st"])
    stim_trans[istim,:]=s
end


figure()
plot(stim[:],stim_trans[:],".")
#
#
d,xfinal=simulation_Nframe_general(param,stim,internal_noise,T,tau,model_f)
d_test,_=simulation_Nframe_general(param,stim_test,internal_noise_test,T,tau,model_f)
println("hola ",mean(0.5*(d+1)),mean(.5*(1+sign(xfinal))))
for isigma in 1:Nsigmas

    indices=isigma:Nsigmas:Nstim
end





#######Computing PR############
Ntrials=1000
Nstim_PR=100
d_pr=zeros(Ntrials,Nstim_PR)
d_pr_test=zeros(Ntrials,Nstim_PR)
x_pr=zeros(Ntrials)
for itrial in 1:Ntrials
    internal_noise=sigma_a*randn(Nstim_PR,Int(Nframes*NTframes))
    d_pr[itrial,:],x_pr=simulation_Nframe_general(param,stim[1:Nstim_PR,:],internal_noise,T,tau, model_f)
    internal_noise=sigma_a*randn(Nstim_PR,Int(Nframes*NTframes))
    d_pr_test[itrial,:],_=simulation_Nframe_general(param,stim_test[1:Nstim_PR,:],internal_noise,T,tau,model_f)
end


####compute trace ####
N_traces=20
istim_pdf=2
internal_noise=sigma_a*randn(N_traces,Int(Nframes*NTframes))
stim_pdf=transpose(repeat(stim[istim_pdf,:],1,N_traces))
d_traces,x=simulation_Nframe_general_trace(param,stim_pdf,internal_noise,T,tau, model_f)


PR_sim=zeros(Float64,Nstim_PR)
PR_sim_test=zeros(Float64,Nstim_PR)

for istim in 1:Nstim_PR
    PR_sim[istim]= mean( 0.5.*(d_pr[:,istim]+1))
    PR_sim_test[istim]= mean( 0.5.*(d_pr_test[:,istim]+1))
end




# path="/home/genis/model_fitting_julia/"
# filename="synthetic_data_DW_Nstim"*string(Nstim)*"_5sigmas_test_largeParam.jld"
filename="synthetic_data_DW_Nstim"*string(Nstim)*"_model"*model_num*".jld"

save_dict=Dict("d"=>d,"stim"=>stim,"param"=>param,"sigma_a"=>sigma_a,"Tframe"=>T,
"PR"=>PR_sim,"kernel"=>kernel_sigma, "PR_sim_test"=>PR_sim_test, "stim_test"=>stim_test,
"d_test"=>d_test,"Tframe"=>Tframe,"x_trace"=>x,"istim_pdf"=>istim_pdf)
JLD.save(path_save*filename,save_dict)
