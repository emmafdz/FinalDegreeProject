# Global variables
const epsilon = 10.0^(-10);
const epsilon2 = 1e-8;
const dx=0.15
const dt = 0.04;

const total_rate = 40;
const maximum_stim=3 ### update this with the magnitude of the maximum stimulus ####
using Polynomials
using Distributions
using ForwardDiff
using Polynomials
import ForwardDiff.DiffBase


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
    bias_d=params[6]
    hb_d=params[7]
    # println("sigmaaaa ",sigma2)
    # sigma2_sbin = convert(typeof(sigma2), sigma2)
    sigma2_sbin = convert(Float64, sigma2)


    # println(typeof(sigma2))
    # println(size(sigma2))
    # println(sigma2)
    # println("Converted ", sigma2_sbin)
    # println(typeof(sigma2_sbin))


    n_sbins = max(70, ceil(10*sqrt(sigma2_sbin)/dx))
    #println(n_sbins,sigma2_sbin)
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
        mu = bin_centers[j] - (-(stim+bias_d+hb_d)-c2*bin_centers[j]+c4*bin_centers[j]^3+c6*bin_centers[j]^5)*dt#(exp(lam*dt))
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

function Fmatrix_simple{T}(F::AbstractArray{T,2},params::Vector, bin_centers,absorbing=0)

    stim=params[1]
    c2=params[2]
    c4=params[3]
    c6=params[4]
    sigma2=params[5]
    bias_d=params[6]
    hb_d=params[7]
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

        #mu = bin_centers[j] - (-(stim+bias_d+hb_d)-c2*bin_centers[j]+c4*bin_centers[j]^3+c6*bin_centers[j]^5)*dt#(exp(lam*dt))
        aux_mu=stim+bias_d+hb_d
        mu=(bin_centers[j] + aux_mu/c2)*exp(c2*dt) - aux_mu/c2

        for k in 1:sbin_length
            sbin = (k-1)*sbinsize + mu - swidth

            if sbin <= bin_centers[1] #(bin_centers[1] + bin_centers[2])/2
                F[1,j] = F[1,j] + ps[k]
            elseif bin_centers[end] <= sbin#(bin_centers[end]+bin_centers[end-1])/2 <= sbins[k]
                F[end,j] = F[end,j] + ps[k]
            else # more condition
                control=true
                i=2
                while control && i< length(bin_centers)
                    if sbin< bin_centers[i]+dx/2
                        F[i,j]=F[i,j]+ps[k]
                        control=false
                    end
                    i=i+1
                end
            end
        end
    end
end






function ProbRight2_Crank(stim,history_bias,Tframe,param,model_f)

    Ntrials,Nframes=(size(stim))
    PR_model=zeros(Ntrials)
    Nsteps=convert(Int,floor(Tframe/dt))

    sigma_a,c2,c4,c6,bias_d,hbias_d,x0,ro,bins,Nbins,absorbing=param_aux(stim,param,model_f,Ntrials,Nframes,history_bias)
    s_trans=zeros(typeof(sigma_a),Nframes)

    for itrial in 1:Ntrials
        model_f[:st_f](stim[itrial,:],s_trans,param["st"])

        PR_model[itrial]=ProbRight_1stim_Crank_Nicolson(s_trans,history_bias[itrial],bins,Nbins,Nsteps,Nframes,absorbing,c2,c4,c6,bias_d,
        hbias_d,sigma_a,ro,x0[itrial,:])

    end
    return PR_model
end



function ProbRight_1stim_Crank_Nicolson(s_trans,hb,bins,Nbins,Nsteps,Nframes,absorbing,
    c2,c4,c6,bias_d, hbias_d,sigma_a,ro,x0)

    # P[:]=0
    # mu[:]=0
    mu=zeros(typeof(sigma_a),Nbins)
    P=zeros(typeof(sigma_a),Nframes*Nsteps+1,Nbins)
    P[1,:]=x0
    aux=dt/(2*dx)
    D=(sigma_a^2)/2


    nu=D*dt/(dx^2)

    M=Tridiagonal(zeros(Nbins-1),2*nu*ones(Nbins),zeros(Nbins-1))
    I=Diagonal(ones(Nbins))
    it_aux=1
    #println("D: ",D,"nu: ",nu)
    for iframe in 1:Nframes
        for ix in 1:Nbins
            #print(mu[ix]," ",s_trans[iframe]," ",bias_d[iframe]," ",hbias_d[iframe]," ",bins[ix]," ",c2[iframe]," ",c4[iframe]," ",c6[iframe],"\n")
            mu[ix]=-(-(s_trans[iframe]+bias_d[iframe]+hbias_d[iframe]*hb[1])-c2[iframe]*bins[ix]+c4[iframe]*bins[ix]^3+c6[iframe]*bins[ix]^5)

        end
        chi=(dt/dx)*mu
        M.dl[:]=-nu-0.5*chi[2:end]
        M.du[:]=-nu+0.5*chi[2:end]


        #build Matrix

        for it in 1:Nsteps
                #println(it_aux)
                P[it_aux+1,:]=(I+0.5*M) \ ( ( I-0.5*M)*P[it_aux,:]  )
                # indx=find(x->x<0,P[it_aux+1,:])
                # P[it_aux+1,indx]=0
                P[it_aux+1,:]=P[it_aux+1,:]/(sum(P[it_aux+1,:]))
                it_aux=it_aux+1

        end
    end

    pr=dot(P[end,:],ro)
    return pr
end

function ProbRight2_parallel(stim,history_bias,Tframe,param,model_f)

    Ntrials,Nframes=(size(stim))
    #PR_model=zeros(Ntrials)
    PR_model = SharedArray{ typeof(x[1]) }(Ntrials)

    Nsteps=convert(Int,floor(Tframe/dt))
    sigma_a,c2,c4,c6,bias_d,hbias_d,x0,ro,bins,Nbins,absorbing=param_aux(stim,param,model_f,Ntrials,Nframes,history_bias)
    println("Nbins",Nbins," bins[1] ",bins[1])

    @sync @parallel for itrial in 1:Ntrials
        F=zeros(typeof(sigma_a),Nbins,Nbins)
        s_trans=zeros(typeof(sigma_a),Nframes)

        model_f[:st_f](stim[itrial,:],s_trans,param["st"])

        PR_model[itrial]=ProbRight_1stim_fast(s_trans,history_bias[itrial],bins,Nbins,Nsteps,Nframes,absorbing,c2,c4,c6,bias_d,
        hbias_d,sigma_a,ro,x0[itrial,:],F)

    end
    return PR_model
end

function ProbRight2(stim,history_bias,Tframe,param,model_f)

    Ntrials,Nframes=(size(stim))
    PR_model=zeros(Ntrials)
    Nsteps=convert(Int,floor(Tframe/dt))
    sigma_a,c2,c4,c6,bias_d,hbias_d,x0,ro,bins,Nbins,absorbing=param_aux(stim,param,model_f,Ntrials,Nframes,history_bias)
    println("Nbins",Nbins," bins[1] ",bins[1])
    s_trans=zeros(typeof(sigma_a),Nframes)
    F=zeros(typeof(sigma_a),Nbins,Nbins)

    for itrial in 1:Ntrials
        model_f[:st_f](stim[itrial,:],s_trans,param["st"])

        PR_model[itrial]=ProbRight_1stim_fast(s_trans,history_bias[itrial],bins,Nbins,Nsteps,Nframes,absorbing,c2,c4,c6,bias_d,
        hbias_d,sigma_a,ro,x0[itrial,:],F)

    end
    return PR_model
end



function ProbRight_1stim_fast(s_trans,hb,bins,Nbins,Nsteps,Nframes,absorbing,
    c2,c4,c6,bias_d,hbias_d,sigma_a,ro,x0,F)

    F[:]=0
    #F=zeros(typeof(sigma_a),Nbins,Nbins)
    sigma_effective=dt*sigma_a^2   #here I transform std to variance. The energy landscape is symtric for pos and neg sitgma_a
    #x0=zeros(typeof(sigma_a),Nbins)
    #binN=convert(Int,(Nbins-1)/2)
    #x0[convert(Int,binN+1)]=1
    params2=zeros(typeof(c2[1]),7)
    params2[1]=s_trans[1]
    params2[2]=c2[1]
    params2[3]=c4[1]
    params2[4]=c6[1]
    params2[5]=sigma_effective
    params2[6]=bias_d[1]
    params2[7]=hbias_d[1]*hb
    # params2=[k*s[1],c2,c4,sigma_effective]
    #Fmatrix_simple(F,params2,bins,absorbing)
    Fmatrix2(F,params2,bins,absorbing)

    x=F*x0

    for iframe in 1:Nframes

        params2[1]=s_trans[iframe]
        params2[2]=c2[iframe]
        params2[3]=c4[iframe]
        params2[4]=c6[iframe]
        params2[6]=bias_d[iframe]
        params2[7]=hbias_d[iframe]*hb
        # params2=[k*s[iframe],c2,c4,sigma_effective]
        #F=zeros(typeof(sigma_a),Nbins,Nbins)

        if iframe==1
            N_aux=Nsteps-1
        else
            N_aux=Nsteps
            F[:]=0
            Fmatrix2(F,params2,bins,absorbing)
        end

        #println(N_aux)
        for iN in 1:N_aux
            x=F*x
        end
    end
    pr=dot(x,ro)

    return pr
end



# function ProbRight_1stim_fast_forward_euler(s_trans,hb,bins,Nbins,Nsteps,Nframes,absorbing;
#     c2=zeros(Nframes),c4=zeros(Nframes),c6=zeros(Nframes),bias_d=zeros(Nframes),
#     hbias_d=zeros(Nframes),sigma_a=0.,ro=vcat(zeros((Nbins-1)/2),[0.5],ones( (Nbins-1)/2)),
#     x0=vcat(zeros((Nbins-1)/2),[1.],zeros( (Nbins-1)/2))
#     )
#     Nframes=length(s_trans)
#     mu=zeros(typeof(sigma_a),Nbins)
#     P=zeros(typeof(sigma_a),Nbins,Nframes*Nsteps+1)
#
#     P[:,1]=x0
#     aux=dt/(2*dx)
#     D=(sigma_a^2)/2
#     aux2=D*dt/(dx^2)
#     for iframe in 1:Nframes
#         it_aux=(iframe-1)*Nsteps
#         for ix in 1:Nbins
#             #print(mu[ix]," ",s_trans[iframe]," ",bias_d[iframe]," ",hbias_d[iframe]," ",bins[ix]," ",c2[iframe]," ",c4[iframe]," ",c6[iframe],"\n")
#             mu[ix]=-(-(s_trans[iframe]+bias_d[iframe]+hbias_d[iframe]*hb[1])-c2[iframe]*bins[ix]+c4[iframe]*bins[ix]^3+c6[iframe]*bins[ix]^5)
#
#         end
#         for it in 1:Nsteps
#             for ix in 2:Nbins-1
#                 P[ix,it+1+it_aux]=P[ix,it+it_aux]-aux*( mu[ix+1]*P[ix+1,it+it_aux]-mu[ix-1]*P[ix-1,it+it_aux])+ aux2*(P[ix+1,it+it_aux]+P[ix-1,it+it_aux]-2*P[ix,it+it_aux])
#             end
#
#             ix=1
#             P[2,it+it_aux]=P[2,it+it_aux]-aux*( mu[ix+1]*P[ix+1,it+it_aux])+ aux2*(P[ix+1,it+it_aux]-2*P[ix,it+it_aux])
#             ix=Nbins-1
#             P[Nbins-2,it+it_aux]=P[Nbins-2,it+it_aux]-aux*(-mu[ix-1]*P[ix-1,it+it_aux])+ aux2*(P[ix-1,it+it_aux]-2*P[ix,it+it_aux])
#         end
#     end
#
#     pr=dot(P[:,end],ro)
#
#     return pr
# end

function LogLikelihood2(s,d,hb,bins,Nbins,Nsteps,Nframes,absorbing,model_f,
    c2,c4,c6,bias_d,hbias_d,sigma_a,ro,x0)

    F=zeros(typeof(c2[1]),Nbins,Nbins)
    pr=ProbRight_1stim_fast(s,hb,bins,Nbins,Nsteps,Nframes,absorbing,c2,c4,c6,bias_d,
    hbias_d,sigma_a,ro,x0,F)


    if d==1
        # return log(ProbRight_1stim(s,bins,Nsteps,Nframes; c2=c2,c4=c4,sigma_a=sigma_a,lapse=lapse,B=B,k=k)+epsilon)
        # pr=ProbRight_1stim(s,bins,Nsteps,Nframes; kwargs...)+epsilon2
        #println("pr3:", pr)
        return log( maximum( [pr epsilon2] ) )

    else
        #println("pr2:", pr)
        # pr=ProbRight_1stim(s,bins,Nsteps,Nframes; kwargs...)+epsilon2
        return log( maximum([1.0-pr epsilon2] ) )

        # return log(1-ProbRight_1stim(s,bins,Nsteps,Nframes; c2=c2,c4=c4,sigma_a=sigma_a,lapse=lapse,B=B,k=k)+epsilon)
    end
end


function LogLikelihood_sigma0(s,hb,bins,Nbins,Nsteps,Nframes,absorbing,model_f,
    c2,c4,c6,bias_d,hbias_d,sigma_a,ro,x0)
    F=zeros(typeof(c2[1]),Nbins,Nbins)

    pr=ProbRight_1stim_fast(s,hb,bins,Nbins,Nsteps,Nframes,absorbing,c2,c4,c6,bias_d,
    hbias_d,sigma_a,ro,x0,F)


    return log( maximum( [pr epsilon2] ) ), log( maximum([1.0-pr epsilon2] ) )

end



function param_aux(stim,param,model_f,Ntrials,Nframes,history_bias)


    sigma_a=param["sigma_a"][1]

    c2=zeros(typeof(param["st"][1]),Nframes) #We need to define here c2, array can not be define in functions because of ForwardDiff
    c4=zeros(typeof(param["st"][1]),Nframes)
    c6=zeros(typeof(param["st"][1]),Nframes)
    bias_d=zeros(typeof(param["st"][1]),Nframes)
    hbias_d=zeros(typeof(param["st"][1]),Nframes)
    model_f[:c2_f](c2,param["c2"])
    model_f[:c4_f](c4,param["c4"])
    if "c6" in keys(param)
        model_f[:c6_f](c6,param["c6"])
    else
        model_f[:c6_f](c6)
    end

    if "bias_d" in keys(param)
        model_f[:bias_d_f](bias_d,param["bias_d"])
    else
        model_f[:bias_d_f](bias_d)
    end

    if "hbias_d" in keys(param)
        model_f[:hbias_d_f](hbias_d,param["hbias_d"])
    else
        model_f[:hbias_d_f](hbias_d)
    end
    max_mu=zeros(1)
    stim_max=zeros(1)+maximum_stim
    model_f[:st_f](stim_max,max_mu,param["st"])

    if "B" in keys(param)
        B=param["B"]
        absorbing=param["absorbing"]
    else
        B,absorbing=compute_bound_all(max_mu[1],c2,c4,c6,sigma_a)
    end
    #println("B:",B[1])
    binN=Int(ceil(B[1]/dx))
    Nbins=2*binN+1
    bins=zeros(typeof(param["st"][1]),Nbins)
    make_bins(bins,B[1],dx,binN)

    ro=zeros(typeof(param["st"][1]),Nbins)
    x0=zeros(typeof(param["st"][1]),(Ntrials,Nbins))


    if model_f[:ro_f]==read_out_perfect
        model_f[:ro_f](ro,binN)
    elseif ro_f==read_out_TW
        model_f[:ro_f](ro,bins,c2[end],c4[end],c6[end])
    else:
        model_f[:ro_f](ro,bins,param["ro"])
    end



    if model_f[:x0_f]==initial_condition
        x0_a=zeros(typeof(param["st"][1]),Nbins)
        model_f[:x0_f](x0_a)
        x0[:]=repeat(x0_a,1,Ntrials)
    elseif model_f[:x0_f]==initial_condition_bias
        x0_a=zeros(typeof(param["st"][1]),Nbins)
        model_f[:x0_f](x0_a,bins,param["x0"])
        x0[:]=repeat(x0_a,1,Ntrials)
    elseif model_f[:x0_f]==initial_condition_bias_hbias
        x0_m=param["x0"][1]+param["x0"][2]*history_bias
        model_f[:x0_f](x0,bins,x0_m)
    end

    return sigma_a,c2,c4,c6,bias_d,hbias_d,x0,ro,bins,Nbins,absorbing

end



function ComputeLL2(stim,i_sigma,choices,history_bias,Tframe,args,x,consts,y,model_f;fastsigma0=true)

    Ntrials,Nframes=(size(stim))
    Nsteps=convert(Int,floor(Tframe/dt))
    param=make_dict(args,x,consts,y)

    sigma_a,c2,c4,c6,bias_d,hbias_d,x0,ro,bins,Nbins,absorbing=param_aux(stim,param,model_f,Ntrials,Nframes,history_bias)
    LLs = SharedArray{ typeof(x[1]) }(Ntrials)
    #LLs=zeros(typeof(sigma_a),Ntrials)
    s_trans=zeros(typeof(param["st"][1]),Nframes)

    ### comment if there is no sigma=0##
    if fastsigma0
        isig0=findfirst(x->x==0,i_sigma)
        s_r=abs.(stim[isig0,:])
        model_f[:st_f](s_r,s_trans,param["st"])
        ll_rr,ll_rl=LogLikelihood_sigma0(s_trans,0.0,bins,Nbins,Nsteps,Nframes,absorbing,model_f,
        c2,c4,c6,bias_d,hbias_d,sigma_a,ro,x0[1,:])
        #ll_rr,ll_rl ll of a right trial with sigma=0 and  decision r(l)
        s_l=-abs.(stim[isig0,:]) #left stimulus with sigma=0
        model_f[:st_f](s_l,s_trans,param["st"])
        ll_lr,ll_ll=LogLikelihood_sigma0(s_trans,0.0,bins,Nbins,Nsteps,Nframes,absorbing,model_f,
        c2,c4,c6,bias_d,hbias_d,sigma_a,ro,x0[1,:])
    end

    @sync @parallel for itrial in 1:Ntrials
    #for itrial in 1:Ntrials
        if i_sigma[itrial]==0
            if stim[itrial,1]>0
                if choices[itrial]==1
                    LLs[itrial]=ll_rr
                else
                    LLs[itrial]=ll_rl
                end
            else
                if choices[itrial]==1
                    LLs[itrial]=ll_lr
                else
                    LLs[itrial]=ll_ll
                end
            end

        else
            model_f[:st_f](stim[itrial,:],s_trans,param["st"])
            LLs[itrial]=LogLikelihood2(s_trans,choices[itrial],history_bias[itrial],bins,Nbins,Nsteps,Nframes,absorbing,model_f,
            c2,c4,c6,bias_d,hbias_d,sigma_a,ro,x0[itrial,:])

        end
    end

    LL = -sum(LLs)

    return  LL + model_f[:lambda]*sum(abs.(x))

end




function ComputeLL2_rats(stim,i_sigma,i_mu,mu_vec,choices,history_bias,Tframe,args,x,consts,y,model_f;)
    Ntrials,Nframes=(size(stim))
    Nsteps=convert(Int,floor(Tframe/dt))
    param=make_dict(args,x,consts,y)

    sigma_a,c2,c4,c6,bias_d,hbias_d,x0,ro,bins,Nbins,absorbing=param_aux(stim,param,model_f,Ntrials,Nframes,history_bias)
    LLs = SharedArray{ typeof(x[1]) }(Ntrials)
    #LLs=zeros(typeof(sigma_a),Ntrials)

    ll_r = zeros(typeof(sigma_a),size(mu_vec))
    ll_l = zeros(typeof(sigma_a),size(mu_vec))
    counter = 1
    s_trans=zeros(typeof(sigma_a),Nframes)
    aux = zeros(typeof(sigma_a),Nframes)
    for mu in mu_vec
        # i_sig0 = find(x->x==0, i_sigma)
        # x_sig0 = x0[i_sig0,:]
        # i_mu0 = i_mu[i_sig0]
        # imu=findfirst(x->x==mu,i_mu0)
        # if imu == 0
        #     # println("imu = 0")
        #     ll_r[counter],ll_l[counter] = 0,0
        # else
        # println("imu != 0")
        # x0_0 = x_sig0[imu,:] # mirar si sempre és la mateixa
        s = aux+mu
        # println(s)

        model_f[:st_f](s,s_trans,param["st"])
        ll_r[counter],ll_l[counter]=LogLikelihood_sigma0(s_trans,0.0,bins,Nbins,Nsteps,Nframes,absorbing,model_f,
        c2,c4,c6,bias_d,hbias_d,sigma_a,ro,x0[1,:])
        counter += 1
    end
    # s_trans=zeros(typeof(sigma_a),Nframes)
    @sync @parallel for itrial in 1:Ntrials
        # print(itrial)
    # for itrial in 1:Ntrials
        if i_sigma[itrial]==0
            # println("isigma = 0")
            mu_trial = i_mu[itrial]
            mu_ind = findfirst(x -> x == mu_trial,mu_vec) # Provar de fer entrar un vector d'index mu
            if choices[itrial] == 1
                LLs[itrial] = ll_r[mu_ind[1]]
            else
                LLs[itrial] = ll_l[mu_ind[1]]
            end

        else
            # println("isigma != 0")
            model_f[:st_f](stim[itrial,:],s_trans,param["st"])
            LLs[itrial]=LogLikelihood2(s_trans,choices[itrial],history_bias[itrial],bins,Nbins,Nsteps,Nframes,absorbing,model_f,
            c2,c4,c6,bias_d,hbias_d,sigma_a,ro,x0[itrial,:])
        end
    end

    LL = -sum(LLs)

    return  LL +model_f[:lambda]*sum(abs.(x))

end




function compute_bound_all(max_mu,c2,c4,c6,sigmas)
    y=zeros(Float64,4)
    b=zeros(Float64,length(c2))
    sign=zeros(Float64,length(c2))
    for iframe in 1:length(c2)
        y[1]=c2[iframe]
        y[2]=c4[iframe]
        y[3]=c6[iframe]
        #y[4]=sigmas[iframe]
        y[4]=sigmas[1]

        b[iframe],sign[iframe]=compute_bound(;max_mu=max_mu,c2=y[1],c4=y[2],c6=y[3],sigma_a=y[4])
    end
    idx=find(b.==maximum(b))[1]
    return b[idx],sign[idx]
end

function compute_bound(;max_mu=0,c2=1,c4=1,c6=0,sigma_a=0.1)
    # if sigma_a<2.
    #     a=Polynomials.poly([,1.035,5.471])
    # else
    #     a=Polynomials.poly([34.626,-23.743,10.040])
    # end
    #a=Polynomials.Poly([0.0038,2.575,1.30375])### 99%
    a=Polynomials.Poly([0.6865, 2.6837, 2.6068])### 99,9%
    fprima=Polynomials.polyval(a,sigma_a)
    #p=Poly([1.64*2./sqrt(dt)*sigma_a^2, -c2, 0, c4,0,c6])
    p=Poly([-fprima-max_mu,-c2,0,c4,0,c6])
    roots_p=roots(p)
    # println("roots :",roots_p,c2," ", c4, " ",sigma_a)
    max_bound=40.
    if length(roots_p)==0
        b=max_bound

    elseif typeof(roots_p[1])!=Float64
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

function ModelFitting(stim,i_sigma,choices,history_bias, Tframe::Float64, args,consts,ub,lb, x_ini::Vector{Float64},y,model_f;fastsigma0=true)

    function LL_f(x)
        # println("param: ",x)ComputeLL2
        return ComputeLL2(stim,i_sigma,choices,history_bias,Tframe, args,x,consts,y,model_f,fastsigma0=fastsigma0)
    end

    println("x_ini hola ",x_ini)
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

    history = optimize(d4, lb, ub, x_ini, Fminbox(LBFGS()),Optim.Options(show_trace = false,store_trace=false,extended_trace = false,iterations=50,g_tol = 1e-8,f_tol = 1e-8,x_tol=1e-8) )
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

    ci=compute_confidence_interval(Hess)

    for i=1:length(history.trace)
        tt = getfield(history.trace[i],:metadata)
        fs[i] = getfield(history.trace[i],:value)
        Gs[i,:] = tt["g(x)"]
        Xs[i,:] = tt["x"]
    end
    param3=make_dict(args,history.minimizer,consts,y)
    D = Dict([("x_ini",x_ini),
                ("parameters",param3),
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
                ("ci",ci),
                ])
    return D

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



function ModelFitting_rats(stim,i_sigma,i_mu,mu_vec,choices,history_bias, Tframe::Float64, args,consts,ub,lb, x_ini::Vector{Float64},y,model_f)

    function LL_f(x)
        # println("param: ",x)ComputeLL2
        return ComputeLL2_rats(stim,i_sigma,i_mu,mu_vec,choices,history_bias,Tframe, args,x,consts,y,model_f)
    end

    println("x_ini hola ",x_ini)
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

    history = optimize(d4, lb, ub, x_ini, Fminbox(LBFGS()),Optim.Options(show_trace = false,store_trace=false,extended_trace = false,iterations=500,g_tol = 1e-8,f_tol = 1e-8,x_tol=1e-8) )
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
    param3=make_dict(args,history.minimizer,consts,y)
    D = Dict([("x_ini",x_ini),
                ("parameters",param3),
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





function LL_function_grads(stim,i_sigma,choices,history_bias,Tframe, args, x,consts,y,model_f)

    function LL_f23(z)
        # println("param: ",x)
        return ComputeLL2(stim,i_sigma,choices,history_bias,Tframe, args,z,consts,y,model_f)
    end



    d4=OnceDifferentiable(LL_f23,x;autodiff=:forward)
    # @time f=d4.f(x_ini)
    grads=zeros( length(x))

    f=d4.fdf(grads,x)

    # d4.df(x_ini,grads)
    # @time d4.df(grads,x_ini)
    println("grad: ",grads)
    # tic()
    # Hess=ForwardDiff.hessian(LL_f23, x)
    # toc()
    return f,grads#,Hess

end


function LL_from_PR(PR,choices)
    LL=0.
    for i in 1:length(choices)
        if choices[i]==1
            LL=LL-log(PR[i]+epsilon2)
        else
            try
                LL=LL-log(1-PR[i]+epsilon2)
            catch e
                LL=LL
            end
        end
    end
    return LL

end



function pdf_x_time_1stim_forward_euler(s,hb,Tframe,param,model_f)



    Nframes=length(s)
    stim_trans=zeros(Nframes)
    # println(param)
    sigma_a=param["sigma_a"][1]
    c2=zeros(typeof(sigma_a),Nframes) #We need to define hwere c2, array can not be define in functions because of ForwardDiff
    c4=zeros(typeof(sigma_a),Nframes)
    c6=zeros(typeof(sigma_a),Nframes)
    bias_d=zeros(typeof(sigma_a),Nframes)
    hbias_d=zeros(typeof(sigma_a),Nframes)
    model_f[:c2_f](c2,param["c2"])
    model_f[:c4_f](c4,param["c4"])

    if "c6" in keys(param)
        model_f[:c6_f](c6,param["c6"])
    else
        model_f[:c6_f](c6)
    end

    if "bias_d" in keys(param)
        model_f[:bias_d_f](bias_d,param["bias_d"])
    else
        model_f[:bias_d_f](bias_d)
    end

    if "hbias_d" in keys(param)
        #println(hbias_d)
        model_f[:hbias_d_f](hbias_d,param["hbias_d"])
    else
        model_f[:hbias_d_f](hbias_d)
    end
    Nsteps=convert(Int,floor(Tframe/dt))
    #println("Nsteps ",Nsteps,"Nframes: ",Nframes)

    # y=zeros(Float64,4) ### ForwardDiff is blind to thwe computation of the bound. It is not important for LL computation ####################
    # y[1]=minimum(c2)
    # y[2]=c4[1] ###### This is assuming c4 is constant during stimulus ############
    # y[3]=c6[1]
    # y[4]=sigma_a
    # B,absorbing=compute_bound(c2=y[1],c4=y[2],c6=y[3],sigma_a=y[4])

    s_trans=zeros(Nframes)
    model_f[:st_f](s,s_trans,param["st"])
    max_mu=maximum(abs(s))

    B,absorbing=compute_bound_all(max_mu,c2,c4,c6,sigma_a)


    binN=Int(ceil(B/dx))
    Nbins=2*binN+1
    bins=zeros(Float64,Nbins)
    make_bins(bins,B,dx,binN)




    sigma_effective=dt*sigma_a^2   #here I transform std to variance. The energy landscape is symtric for pos and neg sitgma_a
    binN=convert(Int,(Nbins-1)/2)

    Ntrials=1
    x0_a=zeros(typeof(sigma_a),Nbins)
    if model_f[:x0_f]==initial_condition
        model_f[:x0_f](x0_a)
        x0=repeat(x0_a,1,Ntrials)
    elseif model_f[:x0_f]==initial_condition_bias
        x0_a=zeros(typeof(sigma_a),Nbins)
        model_f[:x0_f](x0_a,bins,param["x0"])
        x0=repeat(x0_a,1,Ntrials)
    elseif model_f[:x0_f]==initial_condition_bias_hbias
        x0=zeros(typeof(sigma_a),(Ntrials,Nbins))
        x0_m=param["x0"][1]+param["x0"][2]*hb
        model_f[:x0_f](x0,bins,x0_m)
    end

    Nx=Nbins
    mu=zeros(typeof(sigma_a),Nx)
    P=zeros(typeof(sigma_a),Nx,Nframes*Nsteps+1)

    P[:,1]=x0
    aux=dt/(2*dx)
    D=(sigma_a^2)/2
    aux2=D*dt/(dx^2)
    for iframe in 1:Nframes
        it_aux=(iframe-1)*Nsteps
        for ix in 1:Nx
            #print(mu[ix]," ",s_trans[iframe]," ",bias_d[iframe]," ",hbias_d[iframe]," ",bins[ix]," ",c2[iframe]," ",c4[iframe]," ",c6[iframe],"\n")
            mu[ix]=-(-(s_trans[iframe]+bias_d[iframe]+hbias_d[iframe]*hb[1])-c2[iframe]*bins[ix]+c4[iframe]*bins[ix]^3+c6[iframe]*bins[ix]^5)

        end
        for it in 1:Nsteps
            for ix in 2:Nx-1
                P[ix,it+1+it_aux]=P[ix,it+it_aux]-aux*( mu[ix+1]*P[ix+1,it+it_aux]-mu[ix-1]*P[ix-1,it+it_aux])+ aux2*(P[ix+1,it+it_aux]+P[ix-1,it+it_aux]-2*P[ix,it+it_aux])
            end

            ix=1
            P[2,it+it_aux]=P[2,it+it_aux]-aux*( mu[ix+1]*P[ix+1,it+it_aux])+ aux2*(P[ix+1,it+it_aux]-2*P[ix,it+it_aux])
            ix=Nx-1
            P[Nx-2,it+it_aux]=P[Nx-2,it+it_aux]-aux*(-mu[ix-1]*P[ix-1,it+it_aux])+ aux2*(P[ix-1,it+it_aux]-2*P[ix,it+it_aux])
        end
    end
    ro=vcat(zeros(binN),[0.5],ones(binN))
    print("PR: ", dot(P[:,end],ro) )


    return P,bins
end



function pdf_x_time_1stim(s,Tframe,param,model_f,history_bias;B=0)

    # Ntrials,Nframes=(size(s))
    Ntrials=1
    Nframes=size(s)[1]
    Nsteps=convert(Int,floor(Tframe/dt))
    #param=make_dict(args,x,consts,y)

    sigma_a,c2,c4,c6,bias_d,hbias_d,x0,ro,bins,Nbins,absorbing=param_aux(s,param,model_f,Ntrials,Nframes,history_bias)


    #
    # Nframes=length(s)
    # stim_trans=zeros(Nframes)
    # sigma_a=param["sigma_a"][1]
    # c2=zeros(typeof(sigma_a),Nframes) #We need to define hwere c2, array can not be define in functions because of ForwardDiff
    # c4=zeros(typeof(sigma_a),Nframes)
    # c6=zeros(typeof(sigma_a),Nframes)
    # model_f[:c2_f](c2,param["c2"])
    # model_f[:c4_f](c4,param["c4"])
    #
    # bias_d=zeros(typeof(sigma_a),Nframes)
    # hbias_d=zeros(typeof(sigma_a),Nframes)
    #
    # if "bias_d" in keys(param)
    #     model_f[:bias_d_f](bias_d,param["bias_d"])
    # else
    #     model_f[:bias_d_f](bias_d)
    # end
    #
    # if "hbias_d" in keys(param)
    #     model_f[:hbias_d_f](hbias_d,param["hbias_d"])
    # else
    #     model_f[:hbias_d_f](hbias_d)
    # end
    #
    #
    # if "c6" in keys(param)
    #     model_f[:c6_f](c6,param["c6"])
    # else
    #     model_f[:c6_f](c6)
    # end
    #
    #
    # Nsteps=convert(Int,floor(Tframe/dt))
    #
    #
    # if B==0
    #     max_mu=zeros(1)
    #     stim_max=zeros(1)+maximum_stim
    #     model_f[:st_f](stim_max,max_mu,param["st"])
    #
    #     B,absorbing=compute_bound_all(max_mu[1],c2,c4,c6,sigma_a)
    #     #compute_bound_all(c2,c4,c6,sigma_a)
    # else
    #     absorbing=1
    # end
    #



    s_trans=zeros(Nframes)
    model_f[:st_f](s,s_trans,param["st"])
    println("s_trans: ",s_trans)

    # binN=Int(ceil(B/dx))
    # Nbins=2*binN+1
    # bins=zeros(Float64,Nbins)
    # make_bins(bins,B,dx,binN)
    #

    #
    #
    sigma_effective=dt*sigma_a^2   #here I transform std to variance. The energy landscape is symtric for pos and neg sitgma_a
    # binN=convert(Int,(Nbins-1)/2)
    # Ntrials=1
    # itrial=1
    # x0_a=zeros(typeof(sigma_a),Nbins)
    # if model_f[:x0_f]==initial_condition
    #     model_f[:x0_f](x0_a)
    #     x0=repeat(x0_a,1,Ntrials)
    # elseif model_f[:x0_f]==initial_condition_bias
    #     x0_a=zeros(typeof(sigma_a),Nbins)
    #     model_f[:x0_f](x0_a,bins,param["x0"])
    #     x0=repeat(x0_a,1,Ntrials)
    # elseif model_f[:x0_f]==initial_condition_bias_hbias
    #     x0=zeros(typeof(sigma_a),(Ntrials,Nbins))
    #     x0_m=param["x0"][1]+param["x0"][2]*hb
    #     model_f[:x0_f](x0,bins,x0_m)
    # end

    #println("sigma",sigma_a," c2 ",c2,"c4 ",c4," bias_d ",bias_d)
    #println(" hbias_d ",hbias_d," bins ", bins," Nbins ", Nbins, "abs", absorbing)

    F=zeros(typeof(sigma_a),Nbins,Nbins)

    params2=zeros(typeof(sigma_a),7)
    params2[1]=s_trans[1]
    params2[2]=c2[1]
    params2[3]=c4[1]
    params2[4]=c6[1]
    params2[5]=sigma_effective
    params2[6]=bias_d[1]
    params2[7]=hbias_d[1]*history_bias[1]

    x=zeros(typeof(sigma_a),Nsteps*Nframes,Nbins)
    j=1
    itrial=1
    Fmatrix2(F,params2,bins,absorbing)
    x[1,:]=F*x0[itrial,:]

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




######################## functions PI #################





function ProbRightPI(stim,history_bias,Tframe,param,model_f)

    Ntrials,Nframes=(size(stim))
    PR_model=zeros(typeof(param["st"][1]),Ntrials)
    T=Nframes*Tframe
    s_trans=zeros(typeof(param["st"][1]),Nframes)
    std=sqrt(T)*param["sigma_a"][1]
    for itrial in 1:Ntrials
        model_f[:st_f](stim[itrial,:],s_trans,param["st"])
        mean=param["x0"][1]+sum(Tframe*s_trans) +T*param["bias_d"][1]
        f=Normal(mean,std)
        PR_model[itrial]=1-cdf(f,0)
    end

    return PR_model
end


function ComputeLLPI(stim,choices,history_bias,Tframe,args,consts, x,y,model_f)

    param=make_dict(args,x,consts,y)
    PR=ProbRightPI(stim,history_bias,Tframe,param,model_f)
    LL=LL_from_PR(PR,choices)

    return LL+model_f[:lambda]*sum(abs.(x))

end



function ModelFittingPI(stim,choices,history_bias, Tframe, args,consts,ub,lb, x_ini,y,model_f)

    function LL_f(x)
        # println("param: ",x)ComputeLL2
        return ComputeLLPI(stim,choices,history_bias,Tframe, args,consts,x,y,model_f)
    end

    println("x_ini hola ",x_ini)

    d4 = OnceDifferentiable(LL_f,x_ini;autodiff=:forward)


    tic()
    history = optimize(d4, lb, ub, x_ini, Fminbox(LBFGS()),Optim.Options(show_trace = false,store_trace=false,extended_trace = false,iterations=500,g_tol = 1e-8,f_tol = 1e-8,x_tol=1e-8) )
    fit_time=toc()
    # println(history.minimizer)
    println(history)

    x_bf = history.minimizer

    Hess=ForwardDiff.hessian(LL_f, x_bf)
    Gs = zeros(length(history.trace),length(x_ini))
    Xs = zeros(length(history.trace),length(x_ini))
    fs = zeros(length(history.trace))

    ci=compute_confidence_interval(Hess)

    for i=1:length(history.trace)
        tt = getfield(history.trace[i],:metadata)
        fs[i] = getfield(history.trace[i],:value)
        Gs[i,:] = tt["g(x)"]
        Xs[i,:] = tt["x"]
    end
    param3=make_dict(args,history.minimizer,consts,y)
    D = Dict([("x_ini",x_ini),
                ("parameters",param3),
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
                ("ci",ci),
                ])
    return D

end




function LL_function_grads_PI(stim,choices,history_bias,Tframe, args,consts,x,y,model_f)

    function LL_f23(z)
        return ComputeLLPI(stim,choices,history_bias,Tframe, args,consts,z,y,model_f)
    end



    d4=OnceDifferentiable(LL_f23,x;autodiff=:forward)
    # @time f=d4.f(x_ini)
    grads=zeros( length(x))

    f=d4.fdf(grads,x)

    # d4.df(x_ini,grads)
    # @time d4.df(grads,x_ini)
    println("grad: ",grads)
    # tic()
    # Hess=ForwardDiff.hessian(LL_f23, x)
    # toc()
    return f,grads#,Hess

end
