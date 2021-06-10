using JLD
# using Pandas
using JSON
using Base.Profile
@everywhere using Base.Test
@everywhere import Base.convert
@everywhere using MAT
@everywhere using ForwardDiff
@everywhere using Polynomials
@everywhere import ForwardDiff.DiffBase


# local
# @everywhere include("/home/genis/cluster_home/model_fitting_julia/gradient_utils.jl")
# @everywhere include("/home/genis/cluster_home/model_fitting_julia/functions_julia.jl")
# path="/home/genis/cluster_home/model_fitting_julia/"

#portatil

path="/home/emma/fit_behaviour_julia/"
@everywhere include(path*"functions/functions_julia.jl")
@everywhere include(path*"functions/utils.jl")
@everywhere include(path*"functions/analysis.jl")
@everywhere include(path*"functions/parameter_functions.jl")


SUBJECTS=[24,25,36,37]
#SUBJECTS=[24]
# filename="synthetic_data_DW_Nstim10000.jld"
#model_fit="Stconst_DW"
#model_fit="StLogisticDw_2"
#model_fit="StLogTimeDw_synt7"
#model_fit="StLogTimeDw_synt"
#model_fit="StLinTimeDw"
#model_fit="PI_Stconst"
#for model_fit in ["StLinTimeDw","StLinDw","StLogDw_1","StLogTimeDw"]
for model_fit in ["StLogTimeDw"]
#for model_fit in ["StLinTimePi"]
	Nframes=10
	Tframe=0.5

	#subject=6
	for subject in SUBJECTS
		println("subject: ",subject)
		# Nconditions=1
		j=1
		path="/home/emma/fit_behaviour_julia/"
		path_files=path*"results/model_fit"*model_fit*"/rat"*string(subject)*"_05sec/"

		save_fit=analysis_minimization_subjects(path_files)
		path_save=path*"results/summary/model_fit"*model_fit*"/"
		mkpath(path_save)
		filename_save=path_save*"rat"*string(subject)*"_05sec.jld"
		JLD.save(filename_save,save_fit)
		# println(type(save_fit))
		# df=Pandas.DataFrame(save_fit)
		filename_save=path_save*"rat"*string(subject)*"_05sec.json"

		stringdata = JSON.json(save_fit)

		# write the file with the stringdata variable information
		open(filename_save, "w") do f
		        write(f, stringdata)
		     end
		# Pandas.to_json(save_fit,filename_save)
	end
end
