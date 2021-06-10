# using JLD
# using PyPlot

#
path = "/home/emma/github/fit_behaviour_julia/scripts_emma/resultats/"
# data = JLD.load("/home/emma/github/fit_behaviour_julia/LLdata_fit3.jld")
# model = data["mod"]
# println(model)
# model1 = model["1"]
# parameters = model1["f"]
using JSON

###################
### Read data #####
###################
# create variable to write the information
data = Dict()
open(path*"fit_2par.json", "r") do f
    global data
    dicttxt = readstring(f)  # file information to string
    data=JSON.parse(dicttxt)  # parse and transform data
end
fitted_min = data["1"]["f"]
ind_min = 1.
par_fit = data["1"]["x_bf"]
x_bf_all = zeros(100,2)
println(size(x_bf_all))
minimum = zeros(100,1)
for i in 1:100
    x_bf_all[i] = data[string(i)]["x_bf"][1]
    x_bf_all[100+i] = data[string(i)]["x_bf"][2]
    minimum[i] = data[string(i)]["f"]
    if data[string(i)]["x_converged"] ==true & data[string(i)]["f_converged"] ==true & data[string(i)]["g_converged"] ==true
        println(i)
        if data[string(i)]["f"]<fitted_min
            println(par_fit)
            par_fit = data[string(i)]["x_bf"]
            fitted_min = data[string(i)]["f"]
            ind_min = i
            println(ind_min)
        end
    end
end
println(par_fit)
# figure()
# plot(1:100,minimum)
# title("history.minimum")
# xlabel("iteration")
# ylabel("history.minimum()")
# plot(ind_min,fitted_min,"o",label ="fitted minimum")
# legend()
# figure()
# plot(1:100,x_bf_all[1:100],label = "c2")
# plot(1:100,x_bf_all[101:200],label = "sigma")
# plot(ind_min,par_fit[1],"o",label = "fitted minimum c2")
# plot(ind_min,par_fit[2],"o",label = "fitted minimum sigma")
# legend()
# title("parameters for each iteration")
# ylabel("parameter value")
# xlabel("iteration")

# println(data[string(Int(ind_min))])
hessian = data[string(Int(ind_min))]["Hessian"]
println(hessian)
hess = zeros(5,5)
println(size(hess))
for a in 1:size(hessian)[1]
    for i in 1:size(hessian)[1]
        hess[a,i]=hessian[a][i]
    end
end
# hess = [hessian[1][1] hessian[2][1]; hessian[1][2] hessian[2][2]]
# hess = [hessian[1][1] hessian[1][2]; hessian[2][1] hessian[2][2]]

println(hess)
inv_hess = inv((hess))
println(diag(inv_hess))
ci = 2*(sqrt(diag(inv_hess)))
println(ci)
println(par_fit)

#fer el likelyhood
# fittejar una funció gaussiana i mirar que em dona allà la sigma_a
#l Copyright (c) 2018 Copyright Holder All Rights Reserved.
