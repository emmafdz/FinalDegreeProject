const dx=0.05
const dt = 0.02;


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



function pdf_x_time_1stim(s,Tframe,param;B=0,hb=0)

    Nframes=length(s)
    stim_trans=zeros(Nframes)
    sigma_a=param["sigma_a"][1]

    Nsteps=convert(Int,floor(Tframe/dt))



    c2=zeros(typeof(sigma_a),Nframes)
    c4=zeros(typeof(sigma_a),Nframes)
    c6=zeros(typeof(sigma_a),Nframes)
    bias_d=zeros(typeof(sigma_a),Nframes)
    hbias_d=zeros(typeof(sigma_a),Nframes)



    binN=Int(ceil(B/dx))
    Nbins=2*binN+1
    bins=zeros(Float64,Nbins)
    make_bins(bins,B,dx,binN)


    F=zeros(typeof(sigma_a),Nbins,Nbins)
    sigma_effective=dt*sigma_a^2   #here I transform std to variance. The energy landscape is symtric for pos and neg sitgma_a
    binN=convert(Int,(Nbins-1)/2)

    params2=zeros(typeof(sigma_a),7)
    params2[1]=s[1]
    params2[2]=c2[1]
    params2[3]=c4[1]
    params2[4]=c6[1]
    params2[5]=sigma_effective
    params2[6]=bias_d[1]
    params2[7]=hbias_d[1]*hb

    x=zeros(typeof(sigma_a),Nsteps*Nframes,Nbins)
    j=1
    absorbing=1
    Fmatrix2(F,params2,bins,absorbing)
    x0=zeros(typeof(sigma_a),Nbins)
    x0[Int( (Nbins+1)/2)]=1
    println("sizeF:",size(F), " size x0",size(x0))
    x[1,:]=F*x0

    for iframe in 1:Nframes

        params2[1]=s[iframe]
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





function simulation_Nframe_general_trace(param,stim,internal_noise,T,tau)
    #println("hola")
    Nstim,Nframes=size(stim)
    aux2=size(internal_noise)
    NT=aux2[2]
    NTframes=NT/Nframes
    deltat=T/NT
    dt_sim=deltat/tau
    dt_sqrt=sqrt(dt_sim)
    d=zeros(Float64,Nstim)
    x=zeros(Float64,Nstim,NT+1)
    c2=zeros(Nframes)
    c4=zeros(Nframes)
    c6=zeros(Nframes)
    println("hola ",dt_sim," ", deltat," ",T," ", NT)
    hbias_d=param["hbias_d"][1]
    bias_d=param["bias_d"][1]

    s=stim
    #println(hb,hbias_d)
    for itrial in 1:Nstim
        it=2
        x[itrial,1]=param["x0"][1]

        #println(s)
        for iframe in 1:Nframes
            for i in 1:NTframes

                # println("it: ",it-1, " itrial ", itrial," iframe ", iframe)
                x[itrial,it]=x[itrial,it-1]-(-(s[iframe])-c2[iframe]*x[itrial,it-1]+c4[iframe]*x[itrial,it-1]^3+c6[iframe]*x[itrial,it-1]^5)*dt_sim+dt_sqrt*internal_noise[itrial,it-1]
                it=it+1
            end
        end

        d[itrial]=sign(x[itrial,end])

    end
    return d,x
end
