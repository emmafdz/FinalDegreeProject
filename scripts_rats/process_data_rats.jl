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
@everywhere path="/home/emma/github/fit_behaviour_julia/"
# cluster
# @everywhere path="/home/hcli64/hcli64751/rats/fit_behaviour_julia/"
#
@everywhere include(path*"/functions/functions_julia.jl")
@everywhere include(path*"/functions/parameter_functions.jl")
@everywhere include(path*"/functions/utils.jl")

rat_num = 4
path_load=path*"scripts_rats/"
filename="/data/processed_data.json"

data = Dict()
open(path_load*filename, "r") do f
    global data
    dicttxt = readstring(f)  # file information to string
    data=JSON.parse(dicttxt)  # parse and transform data
end

stimulus = data[1][rat_num]             # Data frames obtained from a gaussian distribution of a determined coherence
# coherences = data[2][rat_num]           # Target coherences used for each trial
# rewards = data[3][rat_num]              # Correct choice side
decisions = data[4][rat_num]            # Actual decision made by the rat each trial
# performances = data[5][rat_num]         # 0 if the rat chose the correct choice, 1 if the rat chose the incorrect choice
# target_sigmas = data[6][rat_num]        # Target sigmas used for each trial


stim_10 = []
stim_20 = []
choices_10 = []
choices_20 = []

# Separar els estímuls de 10 frames dels de 20 frames
for session in 1:size(stimulus)[1]
    for trial in 1:size(stimulus[session])[1]
        stim_1 = stimulus[session][trial]
        if size(stim_1)[1] == 10
            stim_10 = [stim_10;stim_1]
            choices_10 = [choices_10;decisions[session][trial]]
        end
        if size(stim_1)[1] == 20
            stim_20 = [stim_20;stim_1]
            choices_20 = [choices_20;decisions[session][trial]]
        end
    end
end
n_stim_10 = convert(Int64, size(stim_10)[1]/10)
n_stim_20 = convert(Int64, size(stim_20)[1]/20)

# if n_stim_10 != 0
#     stim_10 = permutedims(reshape(stim_10,(10,n_stim_10)))
# end
# if n_stim_20 != 0
#     stim_20 = permutedims(reshape(stim_20,(20,n_stim_20)))
# end

results = Dict("stim_10"=>stim_10,"stim_20"=>stim_20,"choices_10"=>choices_10,"choices_20"=>choices_20,"n_stim_10"=>n_stim_10,"n_stim_20"=>n_stim_20)

filename_save = path_load*"data/processed_data_rat"*string(rat_num)*".json"
# pass data as a json string (how it shall be displayed in a file)
stringdata = JSON.json(results)

# write the file with the stringdata variable information
open(filename_save, "w") do f
        write(f, stringdata)
     end
