
function ComputeLL2_kiani{T}(stim,N,Tframe, args, x::Vector{T},model_f)

    param=make_dict2(args,x)
    PR,ll=ProbRight2_kiani(stim,N,Tframe,param,model_f)

    reg=model_f[:reg]
    return -ll +model_f[:lambda]*(dot(reg,abs.(x)) + dot(1-reg,1./(abs.(x)) ) )
end


function ProbRight2_kiani(stim,N,Tframe,param,model_f)

    Ntrials=length(stim)
    Nframes=length(stim[end])

    # println(param)
    aux=param["sigma"][1]
    sigmas=zeros(typeof(aux),Nframes)
    c2=zeros(typeof(aux),Nframes) #We need to define here c2, array can not be define in functions because of ForwardDiff
    c4=zeros(typeof(aux),Nframes)
    c6=zeros(typeof(aux),Nframes)

    model_f[:sigma_f](sigmas,param["sigma"])
    model_f[:c2_f](c2,param["c2"])
    model_f[:c4_f](c4,param["c4"])
    if "c6" in keys(param)
        model_f[:c6_f](c6,param["c6"])
    else
        model_f[:c6_f](c6)
    end

    # c2=param["c2"][1]+9*param["c2"][2]
    # c4=c4_f(Nframes;c4_all=param["c4"]) #### This is wrong#####




    Nsteps=convert(Int,floor(Tframe/dt))
    LL = 0.

    y=zeros(Float64,4) ### ForwardDiff is blind to thwe computation of the bound. It is not important for LL computation ####################
    y[1]=minimum(c2)
    y[2]=c4[1] ###### This is assuming c4 is constant during stimulus ############
    y[3]=c6[1] ###### This is assuming c6 is constant during stimulus ############
    y[4]=maximum(sigmas)

    B,absorbing=compute_bound_all(c2,c4,c6,sigmas)


    binN=Int(ceil(B/dx))
    Nbins=2*binN+1
    bins=zeros(typeof(aux),Nbins)
    make_bins(bins,B,dx,binN)



    LLs =zeros(typeof(aux),Ntrials)
    PR=zeros(typeof(aux),Ntrials)
    for itrial in 1:Ntrials

        Nframes=length(stim[itrial])

        if Nframes==2
            c2[2]=c2[1]
            sigmas[2]=sigmas[1]
        else
            model_f[:c2_f](c2,param["c2"])
            model_f[:sigma_f](sigmas,param["sigma"])
        end

        s_trans=zeros(typeof(aux),Nframes)
        s=zeros(typeof(aux),Nframes)
        for iframe in 1:Nframes
            s[iframe]=stim[itrial][iframe]
        end
        model_f[:st_f](s,s_trans,param["st"])
        ro=zeros(typeof(aux),Nbins)


        if model_f[:ro_f]==read_out_perfect
            model_f[:ro_f](ro,binN)
        elseif model_f[:ro_f]==read_out_TW
            model_f[:ro_f](ro,bins,c2[Nframes],c4[Nframes],c6[Nframes])
        elseif model_f[:ro_f]==read_out_DW
            model_f[:ro_f](ro,bins,c2[Nframes],c4[Nframes])
        else
            model_f[:ro_f](ro,bins,param["ro"])
        end

        PR[itrial]=ProbRight_1stim_fast_kiani(s_trans,bins,Nbins,Nsteps,Nframes,absorbing;c2=c2,c4=c4,c6=c6,sigmas=sigmas,ro=ro)
        LLs[itrial]=N[itrial,1]*log(PR[itrial]+epsilon2)+N[itrial,2]*log(1-PR[itrial]+epsilon2)
    end

    LL = sum(LLs)

    return  PR,LL


end


function ProbRight_1stim_fast_kiani(s_trans,bins,Nbins,Nsteps,Nframes,absorbing;
    c2=zeros(Nframes),c4=zeros(Nframes),c6=zeros(Nframes),sigmas=ones(Nframes),ro=vcat(zeros((Nbins-1)/2),[0.5],ones( (Nbins-1)/2)))

    sigma_effective=dt*sigmas[1]^2   #here I transform std to variance. The energy landscape is symtric for pos and neg sitgma_a
    F=zeros(typeof(sigma_effective),Nbins,Nbins)
    x0=zeros(typeof(sigma_effective),Nbins)
    binN=convert(Int,(Nbins-1)/2)
    x0[convert(Int,binN+1)]=1
    params2=zeros(typeof(sigma_effective),5)
    params2[1]=s_trans[1]
    params2[2]=c2[1]
    params2[3]=c4[1]
    params2[4]=c6[1]
    params2[5]=sigma_effective
    # params2=[k*s[1],c2,c4,sigma_effective]
    Fmatrix2(F,params2,bins,absorbing)
    x=F*x0

    for iframe in 1:Nframes

        params2[1]=s_trans[iframe]
        params2[2]=c2[iframe]
        params2[3]=c4[iframe]
        params2[4]=c6[iframe]
        params2[5]=dt*sigmas[iframe]^2
        # params2=[k*s[iframe],c2,c4,sigma_effective]
        F=zeros(typeof(sigma_effective),Nbins,Nbins)
        Fmatrix2(F,params2,bins,absorbing)
        if iframe==1
            N_aux=Nsteps-1
        else
            N_aux=Nsteps
        end
        for iN in 1:N_aux
            x=F*x
        end
    end
    pr=dot(x,ro)

    return pr
end


function pdf_x_time_1stim_kiani(s,Tframe,param,model_f)

    Nframes=length(s)
    stim_trans=zeros(Nframes)
    # println(param)
    aux=param["sigma"][1]
    sigmas=zeros(typeof(aux),Nframes)
    c2=zeros(typeof(aux),Nframes) #We need to define hwere c2, array can not be define in functions because of ForwardDiff
    c4=zeros(typeof(aux),Nframes)
    c6=zeros(typeof(aux),Nframes)
    model_f[:c2_f](c2,param["c2"])
    model_f[:c4_f](c4,param["c4"])
    model_f[:sigma_f](sigmas,param["sigma"])

    if "c6" in keys(param)
        model_f[:c6_f](c6,param["c6"])
    else
        model_f[:c6_f](c6)
    end

    if (Nframes==2) & (model_f[:c2_f]==c2_kiani)
        c2[2]=c2[1]#### c2 second frame equal than the first one when there is no delay ####
        c2[2]=c2[3] ### c2 second frame equal to frame 3 when there is no delay #####
        sigmas[2]=sigmas[1]
    else
        model_f[:sigma_f](sigmas,param["sigma"])
        model_f[:c2_f](c2,param["c2"])
    end

    Nsteps=convert(Int,floor(Tframe/dt))

    # y=zeros(Float64,4) ### ForwardDiff is blind to thwe computation of the bound. It is not important for LL computation ####################
    # y[1]=minimum(c2)
    # y[2]=c4[1] ###### This is assuming c4 is constant during stimulus ############
    # y[3]=c6[1]
    # y[4]=sigma_a
    # B,absorbing=compute_bound(c2=y[1],c4=y[2],c6=y[3],sigma_a=y[4])


    B,absorbing=compute_bound_all(c2,c4,c6,sigmas)
    s_trans=zeros(Nframes)
    model_f[:st_f](s,s_trans,param["st"])

    binN=Int(ceil(B/dx))
    Nbins=2*binN+1
    bins=zeros(Float64,Nbins)
    make_bins(bins,B,dx,binN)


    #println(B)

    F=zeros(typeof(sigmas[1]),Nbins,Nbins)
    sigma_effective=dt*sigmas[1]^2   #here I transform std to variance. The energy landscape is symtric for pos and neg sitgma_a
    x0=zeros(typeof(aux),Nbins)
    binN=convert(Int,(Nbins-1)/2)
    x0[convert(Int,binN+1)]=1
    params2=zeros(typeof(aux),5)
    params2[1]=s_trans[1]
    params2[2]=c2[1]
    params2[3]=c4[1]
    params2[4]=c6[1]
    params2[5]=sigma_effective

    x=zeros(typeof(aux),Nsteps*Nframes,Nbins)
    j=1
    Fmatrix2(F,params2,bins,absorbing)
    x[1,:]=F*x0

    for iframe in 1:Nframes

        params2[1]=s_trans[iframe]
        params2[2]=c2[iframe]
        params2[3]=c4[iframe]
        params2[4]=c6[iframe]
        params2[5]=dt*sigmas[iframe]^2
        F=zeros(typeof(aux),Nbins,Nbins)
        Fmatrix2(F,params2,bins,absorbing)
        if iframe==1
            N_aux=Nsteps-1
        else
            N_aux=Nsteps
        end
        for iN in 1:N_aux
            #println(sum(x[j,:]))
            x[j+1,:]=F*x[j,:]
            j+=1
        end

    end
    return x,bins,F
end

function ModelFitting_kiani(stim,N,Tframe::Float64, args,ub,lb, x_ini::Vector{Float64},model_f;verbose=false)

    function LL_f(x)
        # println("param: ",x)
        return ComputeLL2_kiani(stim,N,Tframe,args,x,model_f)
    end

    #println("x_ini hola ",x_ini)
    #println("lower bound hola ",lb)
    #println("upper bound hola ",ub)
    d4 = OnceDifferentiable(LL_f,x_ini;autodiff=:forward)
    # d4 = OnceDifferentiable(Optim.only_fg!(LL_fg!),x_ini)
    #
    # println(d4.f(x_ini))
    # grads=zeros(4)
    # # d4.df(x_ini,grads)
    # d4.df(grads,x_ini)
    # println(grads)


    tic()

    history = optimize(d4, lb, ub, x_ini, Fminbox(LBFGS(linesearch=LineSearches.BackTracking())),Optim.Options(show_trace = verbose,store_trace=true,extended_trace = true,iterations=500,g_tol = 1e-6,f_tol = 1e-12,x_tol=1e-12) )
    # history = optimize(d4, x_ini, LBFGS(),Optim.Options(store_trace=true,extended_trace = true,time_limit=2*3600,show_trace = true,iterations=50,g_tol = 1e-5,f_tol = 1e-4,x_tol=1e-5,g_calls_limit=2000) )
    # history = optimize(d4, x_ini, LBFGS(),Optim.Options(store_trace=true,extended_trace = true,time_limit=2*3600,show_trace = true,iterations=50,g_tol = 1e-5,f_tol = 1e-4,x_tol=1e-5,g_calls_limit=2000) )


    # history = optimize(d4, l, u, x_ini, Fminbox(LBFGS()),Optim.Options(g_tol = 1e-5,
    #                                                                        iterations=1,
    #                                                                        x_tol = 1e-2,
    #                                                                        f_tol = 1e-1,                                                                        iterations = 10,
    #                                                                        store_trace = true,
    #                                                                        show_trace = true,
    #                                                                        extended_trace = true,
    #                                                                        ))

    # history = optimize(d4, l, u, x_ini, Fminbox(LBFGS()),Optim.Options(g_tol = 1e-5,
    #                                                                        x_tol = 1e-5,
    #                                                                        f_tol = 1e-2,                                                                        iterations = 10,
    #                                                                        store_trace = true,
    #                                                                        show_trace = true,
    #                                                                        extended_trace = true,
    #                                                                        iterations=500))

     # ;
     #         optimizer_o = Optim.Options(g_tol = 1e-12,
     #                                                                        x_tol = 1e-10,
     #                                                                        f_tol = 1e-6,                                                                        iterations = 10,
     #                                                                        store_trace = true,
     #                                                                        show_trace = true,
     #                                                                        extended_trace = true))
    # history=optimize(LL_f,x_ini,LBFGS(); autodiff=:forward)

    # history=Optim.optimize(Optim.only_fg!(LL_fg!), x_ini, Optim.LBFGS())
    # history=Optim.optimize(Optim.only_fg!(LL_fg!), l, u, Fminbox(),x_ini, Optim.LBFGS())

    fit_time=toc()
    # println(history.minimizer)
    #println(history)

    x_bf = history.minimizer

    Hess=ForwardDiff.hessian(LL_f, x_bf)
    Gs = zeros(length(history.trace),length(x_ini))
    Xs = zeros(length(history.trace),length(x_ini))
    fs = zeros(length(history.trace))

    for i=1:length(history.trace)
        tt = getfield(history.trace[i],:metadata)
        fs[i] = getfield(history.trace[i],:value)
        Gs[i,:] = tt["g(x)"]
        Xs[i,:] = tt["x"]
    end

    D = Dict([("x_ini",x_ini),
                ("parameters",args),
                ("trials",size(stim)[1]),
                ("f",history.minimum),
                ("x_converged",history.x_converged),
                ("f_converged",history.f_converged),
                ("g_converged",history.g_converged),
                ("grad_trace",Gs),
                ("f_trace",fs),
                ("x_trace",Xs),
                ("fit_time",fit_time),
                ("x_bf",history.minimizer),
                ("myfval", history.minimum),
                ("f_calls", history.f_calls),
                ("g_calls",history.g_calls),
                ("iterations",history.iterations),
                ("Hessian",Hess),
                ])
    return D

end


function LL_function_grads_kiani(stim,N,Tframe, args, x,model_f)

    function LL_f23(y)
        # println("param: ",x)
        #return 3*y[1]+2*y[2]+y[3]
        return ComputeLL2_kiani(stim,N,Tframe,args,y,model_f)
    end

    d4=OnceDifferentiable(LL_f23,x;autodiff=:forward)
    # @time f=d4.f(x_ini)
    grads=zeros(length(x))
    f=d4.fdf(grads,x)
    # d4.df(x_ini,grads)
    # @time d4.df(grads,x_ini)
    println(grads)
    Hess=ForwardDiff.hessian(LL_f23, x)
    return f,grads,Hess

end
