


function analysis_minimization(path,data)

    fnames=readdir(path)
    Nconditions=length(fnames)

    PR_fit=[]
    PR_fit_test=[]
    LL_training=[]
    LL_test=[]
    Res=[]
    param=[]#zeros(Nconditions)
    hessian=[]
    ci=[]

    for icond in 1:(Nconditions)
            control=true
            #icond=111
            try
                    aux=load(path*fnames[icond])
                    #println(fnames[icond])
            catch e
                    println("something wrong in open file icond",icond)
                    control=false
            end

            if control
                    res=aux["res"]
                    if length(res["x_bf"])==10
                            #println(icond)
                            push!(Res,res)
                            push!(PR_fit,aux["PR"])
                            push!(PR_fit_test,aux["PR_test"])
                            push!(LL_training,res["f"])
                            push!(param,res["x_bf"])
                            push!(hessian,res["Hessian"])
                            ### compute ci from Hessian ####
                            push!(ci,compute_confidence_interval(res["Hessian"]))
                            Nstim=length(aux["PR"])
                            aux=LL_from_PR(aux["PR_test"],data["choices"][Nstim:2*Nstim])
                            push!(LL_test,aux)
                            #println(icond," ",res["f"]," ",LL_test[end]," ",res["x_bf"]," ",res["fit_time"]/3600.," ",res["x_ini"])#,res["fit_time"]/res["f_calls"] )
                    end
            end
    end


    save_dict=Dict([("LL_training",LL_training),("param",param),("LL_test",LL_test),
    ("hessian",hessian),("ci",ci),("PR_fit",PR_fit),("Res",Res)])

    return save_dict

end

function analysis_minimization_subjects(path)

    fnames=readdir(path)
    Nconditions=length(fnames)

    PR_training=[]
    PR_test=[]
    LL_training=[]
    LL_test=[]
    #Res=[]
    param_fit=[]#zeros(Nconditions)
    param_all=[]
    hessian=[]
    ci=[]
    control_test=false
    if contains(path,"training_test")
            control_test=true
    end
    #labels_param=["c21","c22","c4","st1","st2","st3","x0","bias_d"]
    #labels_param=["c21","c22","c4","st1","st2","x0"]

    aux=load(path*fnames[1])
    labels_param=arg_labels(aux["args"])
    param_dict_all=Dict([])
    for label in labels_param
           param_dict_all[label]=[]
    end
    #Nparamm=6
    println(labels_param)
    for icond in 1:(Nconditions)
            control=true
            #icond=111
            println(icond)

            try
                    aux=load(path*fnames[icond])
                    #println(fnames[icond])
                    #aux=load(path*fnames[icond])
                    res=aux["res"]
                    println(icond)

                    #push!(Res,res)
                    push!(PR_training,aux["PR_training"])
                    push!(LL_training,res["f"])
                    if control_test
                            push!(PR_test,aux["PR_test"])
                            push!(LL_test,aux["LL_test"])
                    end
                    push!(param_fit,res["x_bf"])
                    for (ilabel,label) in enumerate(labels_param)
                            append!(param_dict_all[label],res["x_bf"][ilabel])
                    end
                    param_dict=make_dict(aux["args"],res["x_bf"],aux["consts"],aux["z"])
                    push!(param_all,param_dict)

                    push!(hessian,res["Hessian"])
                    ### compute ci from Hessian ####
                    #push!(ci,compute_confidence_interval(res["Hessian"]))
                    push!(ci,res["ci"])


            catch e
                    println("something wrong in open file icond",icond)
                    println(path*fnames[icond])
                    control=false
            end

    end

    if control_test
            save_dict=Dict([("LL_training",LL_training),("LL_test",LL_test),
            ("param_fit",param_fit),("hessian",hessian),("ci",ci),
            ("PR_training",PR_training),("PR_test",PR_test),
            ("param_all",param_all) ])
            save_dict_final=merge(save_dict,param_dict_all)

    else
            save_dict=Dict([("LL_training",LL_training),("param_fit",param_fit),
            ("hessian",hessian),("ci",ci),("PR_training",PR_training),
            ("param_all",param_all)])
            save_dict_final=merge(save_dict,param_dict_all)
    end
    return save_dict_final

end



function compute_confidence_interval(H)
    z_aux=1.96
    ci=zeros(size(H)[1])
    try
        HI=inv(H)
        if all(diag(HI).>0)
            ci[:]=z_aux.*sqrt.(diag(HI))
            #println("A minimum ", ci)
        else
            ci[:]=-1.0
            #println("Not a minimum ", diag(HI))
        end

    catch e
        ci[:]=-2.0
        #println("H is not invertible")
    end
    return ci
end
