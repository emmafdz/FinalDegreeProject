# Global variables
const epsilon = 10.0^(-10);
const epsilon2 = 1e-8;

const dx = 0.1;
#const dt = 0.05;
const dt = 0.02; ### kiani ###

const total_rate = 40;
using Polynomials

# 3rd party
# import Base.convert
# using MAT
# using ForwardDiff
# using Optim

# import ForwardDiff.DiffBase
# include("/home/genis/Desktop/parameter_functions.jl")
# include("/home/genis/model_fitting_julia/scripts/functions/utils.jl")

convert(::Type{Float64}, x::ForwardDiff.Dual) = Float64(x.value)
function convert(::Array{Float64}, x::Array{ForwardDiff.Dual})
    y = zeros(size(x));
    for i in 1:prod(size(x))
        y[i] = convert(Float64, x[i])
    end
    return y
end

immutable NumericPair{X,Y} <: Number
  x::X
  y::Y
end
Base.isless(a::NumericPair, b::NumericPair) = (a.x<b.x) || (a.x==b.x && a.y<b.y)


"""
function bin_centers = make_bins(B, dx, binN)

Makes a series of points that will indicate bin centers. The first and
last points will indicate sticky bins. No "bin edges" are made-- the edge
between two bins is always implicity at the halfway point between their
corresponding centers. The center bin is always at x=0; bin spacing
(except for last and first bins) is always dx; and the position
of the first and last bins is chosen so that |B| lies exactly at the
midpoint between 1st (sticky) and 2nd (first real) bins, as well as
exactly at the midpoint between last but one (last real) and last
(sticky) bins.

Playing nice with ForwardDiff means that the *number* of bins must be predetermined.
So this function will not actually set the number of bins; what it'll do is determine their
locations. To accomplish this separation, the function uses as a third parameter binN,
which should be equal to the number of bins with bin centers > 0, as follows:
   binN = ceil(B/dx)
and then the total number of bins will be 2*binN+1, with the center one always corresponding
to position zero. Use non-differentiable types for B and dx for this to work.
"""
function make_bins{T}(bins, B, dx::T, binN)
    cnt = 1
    for i=-binN:binN
        #println(cnt)
        bins[cnt] = i*dx
        cnt = cnt+1
    end

    if binN*dx == B
        bins[end] = B + dx
        bins[1] = -B - dx
    else
        bins[end] = 2*B - (binN-1)*dx
        bins[1] = -2*B + (binN-1)*dx
    end
end


function Fmatrix2{T}(F::AbstractArray{T,2},params::Vector, bin_centers,absorbing=0)

    stim=params[1]
    c2=params[2]
    c4=params[3]
    c6=params[4]
    sigma2=params[5]
    # println("sigmaaaa ",sigma2)
    # sigma2_sbin = convert(typeof(sigma2), sigma2)
    sigma2_sbin = convert(Float64, sigma2)


    # println(typeof(sigma2))
    # println(size(sigma2))
    # println(sigma2)
    # println("Converted ", sigma2_sbin)
    # println(typeof(sigma2_sbin))


    n_sbins = max(70, ceil(10*sqrt(sigma2_sbin)/dx))
    if absorbing==1
        F[1,1] = 1;
        F[end,end] = 1;
    else
        F[2,1] = 1;
        F[end-1,end] = 1;
    end
    swidth = 5.*sqrt(sigma2_sbin)

    sbinsize = swidth/n_sbins #sbins[2] - sbins[1]
    base_sbins    = collect(-swidth:sbinsize:swidth)

    ps       = exp.(-base_sbins.^2/(2*sigma2)) # exp(Array) -> exp.(x)
    ps       = ps/sum(ps);

    sbin_length = length(base_sbins)
    binN = length(bin_centers)

    mu = 0.
    for j in 2:binN-1
        mu = bin_centers[j] - (-stim-c2*bin_centers[j]+c4*bin_centers[j]^3+c6*bin_centers[j]^5)*dt#(exp(lam*dt))
        # mu=bin_centers[j] +stim*dt

        for k in 1:sbin_length
            sbin = (k-1)*sbinsize + mu - swidth

            if sbin <= bin_centers[1] #(bin_centers[1] + bin_centers[2])/2
                F[1,j] = F[1,j] + ps[k]
            elseif bin_centers[end] <= sbin#(bin_centers[end]+bin_centers[end-1])/2 <= sbins[k]
                F[end,j] = F[end,j] + ps[k]
            else # more condition
                if (sbin > bin_centers[1] && sbin < bin_centers[2])
                    lp = 1; hp = 2;
                elseif (sbin > bin_centers[end-1] && sbin < bin_centers[end])
                    lp = binN-1; hp = binN;
                else
                    lp = floor(Int,((sbin-bin_centers[2])/dx)) + 2#find(bin_centers .<= sbins[k])[end]
                    hp = ceil(Int,((sbin-bin_centers[2])/dx)) + 2#lp+1#Int(ceil((sbins[k]-bin_centers[2])/dx) + 1);
                end

                if lp == hp
                    F[lp,j] = F[lp,j] + ps[k]
                else
                    F[hp,j] = F[hp,j] + ps[k]*(sbin - bin_centers[lp])/(bin_centers[hp] - bin_centers[lp])
                    F[lp,j] = F[lp,j] + ps[k]*(bin_centers[hp] - sbin)/(bin_centers[hp] - bin_centers[lp])
                end
            end
        end
    end
end


function ProbRight2(stim,Tframe,param,model_f)
    Ntrials=size(stim)[1]
    PR_model=zeros(Ntrials)
    for itrial in 1:Ntrials
        PR_model[itrial]=ProbRight_1stim(stim[itrial,:],Tframe,param,model_f)
    end
    return PR_model
end


function ProbRight_1stim(s,Tframe,param,model_f)

    Nframes=length(s)
    sigma_a=param["sigma_a"][1]
    c2=zeros(typeof(sigma_a),Nframes) #We need to define hwere c2, array can not be define in functions because of ForwardDiff
    c4=zeros(typeof(sigma_a),Nframes)
    c6=zeros(typeof(sigma_a),Nframes)
    s_trans=zeros(typeof(sigma_a),Nframes)

    model_f[:c2_f](c2,param["c2"])
    model_f[:c4_f](c4,param["c4"])

    if "c6" in keys(param)
        model_f[:c6_f](c6,param["c6"])
    else
        model_f[:c6_f](c6)
    end


    Nsteps=convert(Int,floor(Tframe/dt))


    B,absorbing=compute_bound_all(c2,c4,c6,sigma_a)

    model_f[:st_f](s,s_trans,param["st"])

    binN=Int(ceil(B/dx))
    Nbins=2*binN+1
    bins=zeros(typeof(sigma_a),Nbins)

    make_bins(bins,B,dx,binN)

    ro=zeros(typeof(sigma_a),Nbins)
    if model_f[:ro_f]==read_out_perfect
        model_f[:ro_f](ro,binN)
    elseif ro_f==read_out_TW
        model_f[:ro_f](ro,bins,c2[end],c4[end],c6[end])
    else:
        model_f[:ro_f](ro,bins,param["ro"])
    end
    x0=zeros(typeof(sigma_a),Nbins)
    if model_f[:x0_f]==initial_condition
        model_f[:x0_f](x0)
    else
        model_f[:x0_f](x0,bins,param["x0"])

    end
    return ProbRight_1stim_fast(s_trans,bins,Nbins,Nsteps,Nframes,absorbing;c2=c2,c4=c4,sigma_a=sigma_a,ro=ro,x0=x0)
end

function ProbRight_1stim_fast(s_trans,bins,Nbins,Nsteps,Nframes,absorbing;
    c2=zeros(Nframes),c4=zeros(Nframes),c6=zeros(Nframes),sigma_a=0.,ro=vcat(zeros((Nbins-1)/2),[0.5],ones( (Nbins-1)/2)),
    x0=vcat(zeros((Nbins-1)/2),[1.0],ones( (Nbins-1)/2))    )


    F=zeros(typeof(sigma_a),Nbins,Nbins)
    sigma_effective=dt*sigma_a^2   #here I transform std to variance. The energy landscape is symtric for pos and neg sitgma_a
    #x0=zeros(typeof(sigma_a),Nbins)
    #binN=convert(Int,(Nbins-1)/2)
    #x0[convert(Int,binN+1)]=1
    params2=zeros(typeof(sigma_a),5)
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
        # params2=[k*s[iframe],c2,c4,sigma_effective]
        F=zeros(typeof(sigma_a),Nbins,Nbins)
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



function LogLikelihood2(s,d,bins,Nbins,Nsteps,Nframes,absorbing;
    c2=zeros(Nframes),c4=zeros(Nframes),c6=zeros(Nframes),sigma_a=0.,ro=vcat(zeros((Nbins-1)/2),
    [0.5],ones( (Nbins-1)/2)),x0=vcat(zeros((Nbins-1)/2),[1.0],ones( (Nbins-1)/2)) )
    #println(kwargs)
    pr=ProbRight_1stim_fast(s,bins,Nbins,Nsteps,Nframes,absorbing;c2=c2,c4=c4,c6=c6,sigma_a=sigma_a,ro=ro,x0=x0)

    if d==1
        # return log(ProbRight_1stim(s,bins,Nsteps,Nframes; c2=c2,c4=c4,sigma_a=sigma_a,lapse=lapse,B=B,k=k)+epsilon)
        # pr=ProbRight_1stim(s,bins,Nsteps,Nframes; kwargs...)+epsilon2
        return log(pr+epsilon2)

    else
        # pr=ProbRight_1stim(s,bins,Nsteps,Nframes; kwargs...)+epsilon2
        return log(1.-pr+epsilon2)

        # return log(1-ProbRight_1stim(s,bins,Nsteps,Nframes; c2=c2,c4=c4,sigma_a=sigma_a,lapse=lapse,B=B,k=k)+epsilon)
    end
end



function ComputeLL2{T}(stim,choices,Tframe::Float64, args, x::Vector{T},model_f)

    Ntrials,Nframes=(size(stim))
    param=make_dict2(args,x)
    # println(param)
    sigma_a=param["sigma_a"][1]
    c2=zeros(typeof(sigma_a),Nframes) #We need to define here c2, array can not be define in functions because of ForwardDiff
    c4=zeros(typeof(sigma_a),Nframes)
    c6=zeros(typeof(sigma_a),Nframes)

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


    B,absorbing=compute_bound_all(c2,c4,c6,sigma_a)


    binN=Int(ceil(B/dx))
    Nbins=2*binN+1
    bins=zeros(typeof(sigma_a),Nbins)
    make_bins(bins,B,dx,binN)
    #println(B)
    #println(bins)
    ro=zeros(typeof(sigma_a),Nbins)
    if model_f[:ro_f]==read_out_perfect
        model_f[:ro_f](ro,binN)
    elseif ro_f==read_out_TW
        model_f[:ro_f](ro,bins,c2[end],c4[end],c6[end])
    else:
        model_f[:ro_f](ro,bins,param["ro"])
    end

    x0=zeros(typeof(sigma_a),Nbins)
    if model_f[:x0_f]==initial_condition
        model_f[:x0_f](x0)
    else
        model_f[:x0_f](x0,bins,param["x0"])

    end
    #println(x0)
    LLs = SharedArray{ typeof(x[1]) }(Ntrials)
    s_trans=zeros(typeof(sigma_a),Nframes)
    @sync @parallel for itrial in 1:Ntrials

        model_f[:st_f](stim[itrial,:],s_trans,param["st"])
        LLs[itrial]=LogLikelihood2(s_trans,choices[itrial],bins,Nbins,Nsteps,Nframes,absorbing;c2=c2,c4=c4,c6=c6,sigma_a=sigma_a,ro=ro,x0=x0)
    end

    LL = -sum(LLs)

    return  LL +model_f[:lambda]*(dot(model_f[:reg],x.*x) + dot(1-model_f[:reg],1./(x.*x) ) )


end


function compute_bound_all(c2,c4,c6,sigmas)
    y=zeros(Float64,4)
    b=zeros(Float64,length(c2))
    sign=zeros(Float64,length(c2))
    for iframe in 1:length(c2)
        y[1]=c2[iframe]
        y[2]=c4[iframe]
        y[3]=c6[iframe]
        #y[4]=sigmas[iframe]
        y[4]=sigmas[1]

        b[iframe],sign[iframe]=compute_bound(;c2=y[1],c4=y[2],c6=y[3],sigma_a=y[4])
    end
    idx=find(b.==maximum(b))[1]
    return b[idx],sign[idx]
end

function compute_bound(;c2=1,c4=1,c6=0,sigma_a=0.1)
    # if sigma_a<2.
    #     a=Polynomials.poly([,1.035,5.471])
    # else
    #     a=Polynomials.poly([34.626,-23.743,10.040])
    # end
    #a=Polynomials.Poly([0.0038,2.575,1.30375])### 99%
    a=Polynomials.Poly([0.6865, 2.6837, 2.6068])### 99,9%
    fprima=Polynomials.polyval(a,sigma_a)
    #p=Poly([1.64*2./sqrt(dt)*sigma_a^2, -c2, 0, c4,0,c6])
    p=Poly([-fprima,-c2,0,c4,0,c6])
    roots_p=roots(p)
    # println("roots :",roots_p,c2," ", c4, " ",sigma_a)
    max_bound=8.
    if typeof(roots_p[1])!=Float64
        aux=filter(x-> x.im==0.0,roots_p)
        roots_real=zeros(length(aux))
        for iroot in 1:length(aux)
            roots_real[iroot]=abs(aux[iroot].re)
        end
        b=minimum( [maximum(roots_real),max_bound])
    else
        b=minimum( [maximum(abs.(roots_p)),max_bound])
    end
    # println(roots_p," ",b)
    return b+dx,-sign(b*potential_derivative(b,0.,c2,c4,c6))
end

function potential_derivative(x,s,c2,c4,c6)

    return -s-c2*x+c4*x^3+c6*x^5
end

function ModelFitting(stim,choices, Tframe::Float64, args,ub,lb, x_ini::Vector{Float64},model_f)

    function LL_f(x)
        # println("param: ",x)
        return ComputeLL2(stim,choices,Tframe,args,x,model_f)
    end

    println("x_ini hola ",x_ini)
    println("lower bound hola ",lb)
    println("upper bound hola ",ub)
    d4 = OnceDifferentiable(LL_f,x_ini;autodiff=:forward)
    # d4 = OnceDifferentiable(Optim.only_fg!(LL_fg!),x_ini)
    #
    # println(d4.f(x_ini))
    # grads=zeros(4)
    # # d4.df(x_ini,grads)
    # d4.df(grads,x_ini)
    # println(grads)


    tic()

    history = optimize(d4, lb, ub, x_ini, Fminbox(LBFGS()),Optim.Options(show_trace = true,store_trace=true,extended_trace = true,iterations=500,g_tol = 1e-5,f_tol = 1e-5,x_tol=1e-5) )
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
    println(history)

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





function LL_function_grads(stim,choices,Tframe, args, x,model_f)

    function LL_f23(y)
        # println("param: ",x)
        return ComputeLL2(stim,choices,Tframe, args,y,model_f)
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


function LL_from_PR(PR,choices)
    LL=0.
    for i in 1:length(choices)
        if choices[i]==1
            LL=LL-log(PR[i]+epsilon2)
        else
            LL=LL-log(1-PR[i]+epsilon2)
        end
    end
    return LL

end



function pdf_x_time_1stim(s,Tframe,param,model_f)

    Nframes=length(s)
    stim_trans=zeros(Nframes)
    # println(param)
    sigma_a=param["sigma_a"][1]
    c2=zeros(typeof(sigma_a),Nframes) #We need to define hwere c2, array can not be define in functions because of ForwardDiff
    c4=zeros(typeof(sigma_a),Nframes)
    c6=zeros(typeof(sigma_a),Nframes)
    model_f[:c2_f](c2,param["c2"])
    model_f[:c4_f](c4,param["c4"])

    if "c6" in keys(param)
        model_f[:c6_f](c6,param["c6"])
    else
        model_f[:c6_f](c6)
    end


    Nsteps=convert(Int,floor(Tframe/dt))

    # y=zeros(Float64,4) ### ForwardDiff is blind to thwe computation of the bound. It is not important for LL computation ####################
    # y[1]=minimum(c2)
    # y[2]=c4[1] ###### This is assuming c4 is constant during stimulus ############
    # y[3]=c6[1]
    # y[4]=sigma_a
    # B,absorbing=compute_bound(c2=y[1],c4=y[2],c6=y[3],sigma_a=y[4])


    B,absorbing=compute_bound_all(c2,c4,c6,sigma_a)
    compute_bound_all(c2,c4,c6,sigma_a)
    s_trans=zeros(Nframes)
    model_f[:st_f](s,s_trans,param["st"])

    binN=Int(ceil(B/dx))
    Nbins=2*binN+1
    bins=zeros(Float64,Nbins)
    make_bins(bins,B,dx,binN)




    F=zeros(typeof(sigma_a),Nbins,Nbins)
    sigma_effective=dt*sigma_a^2   #here I transform std to variance. The energy landscape is symtric for pos and neg sitgma_a
    binN=convert(Int,(Nbins-1)/2)

    x0=zeros(typeof(sigma_a),Nbins)
    if model_f[:x0_f]==initial_condition
        model_f[:x0_f](x0)
    else
        model_f[:x0_f](x0,bins,param["x0"])

    end


    params2=zeros(typeof(sigma_a),5)
    params2[1]=s_trans[1]
    params2[2]=c2[1]
    params2[3]=c4[1]
    params2[4]=c6[1]
    params2[5]=sigma_effective

    x=zeros(typeof(sigma_a),Nsteps*Nframes,Nbins)
    j=1
    Fmatrix2(F,params2,bins,absorbing)
    x[1,:]=F*x0

    for iframe in 1:Nframes

        params2[1]=s_trans[iframe]
        params2[2]=c2[iframe]
        params2[3]=c4[iframe]
        params2[4]=c6[iframe]
        F=zeros(typeof(sigma_a),Nbins,Nbins)
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
    return x,bins
end
