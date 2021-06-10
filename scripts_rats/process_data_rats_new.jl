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

# local
@everywhere path="/home/emma/github/fit_behaviour_julia/"
# cluster
# @everywhere path="/home/hcli64/hcli64751/rats/fit_behaviour_julia/"
#
@everywhere include(path*"/functions/functions_julia.jl")
@everywhere include(path*"/functions/parameter_functions.jl")
@everywhere include(path*"/functions/utils.jl")

rat_num = 5
path_load=path*"scripts_rats/"
filename="data/ordered_processed_dataset2_1sec.json"

data = Dict()
open(path_load*filename, "r") do f
    global data
    dicttxt = readstring(f)  # file information to string
    data=JSON.parse(dicttxt)  # parse and transform data
end

stimulus = data[1][rat_num]             # Data frames obtained from a gaussian distribution of a determined coherence
coherences = data[2][rat_num]           # Target coherences used for each trial
rewards = data[3][rat_num]              # Correct choice side
decisions = data[4][rat_num]            # Actual decision made by the rat each trial
performances = data[5][rat_num]         # 0 if the rat chose the correct choice, 1 if the rat chose the incorrect choice
target_sigmas = data[6][rat_num]        # Target sigmas used for each trial
dates = data[7][rat_num]

stim_10 = []
stim_20 = []
choices_10 = []
choices_20 = []
coherences_10 = []
coherences_20 = []
rewards_10 = []
rewards_20 = []
sigma_10 = []
sigma_20 = []
performance_10 = []
performance_20 = []
date_10 = []
date_20 = []

# Separar els estímuls de 10 frames dels de 20 frames
for session in 1:size(stimulus)[1]
    stim_1 = stimulus[session]
    if size(stim_1)[1] == 10
        stim_10 = [stim_10;stim_1]
        choices_10 = [choices_10;decisions[session]]
        coherences_10 = [coherences_10;coherences[session]]
        rewards_10 = [rewards_10;rewards[session]]
        sigma_10 = [sigma_10;target_sigmas[session]]
        performance_10 = [performance_10;performances[session]]
        date_10 = [date_10;dates[session]]
    end
    if size(stim_1)[1] == 20
        stim_20 = [stim_20;stim_1]
        choices_20 = [choices_20;decisions[session]]
        coherences_20 = [coherences_20;coherences[session]]
        rewards_20 = [rewards_20;rewards[session]]
        sigma_20 = [sigma_20;target_sigmas[session]]
        performance_20 = [performance_20;performances[session]]
        date_20 = [date_20;dates[session]]
    end
end
n_stim_10 = convert(Int64, size(stim_10)[1]/10)
n_stim_20 = convert(Int64, size(stim_20)[1]/20)


results = Dict("stim_10"=>stim_10,"stim_20"=>stim_20,"choices_10"=>choices_10,"choices_20"=>choices_20,"n_stim_10"=>n_stim_10,"n_stim_20"=>n_stim_20,
"coherences_10"=>coherences_10,"coherences_20"=>coherences_20,"rewards_10"=>rewards_10,"rewards_20"=>rewards_20,"sigma_10"=>sigma_10,
"sigma_20"=>sigma_20,"performance_10"=>performance_10,"performance_20"=>performance_20,"date_10"=>date_10,"date_20"=>date_20)

filename_save = path_load*"data/processed_data_rat"*string(rat_num)*"_dataset2_1.json"
# pass data as a json string (how it shall be displayed in a file)
stringdata = JSON.json(results)

# write the file with the stringdata variable information
open(filename_save, "w") do f
        write(f, stringdata)
     end
