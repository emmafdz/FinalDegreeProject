const max_stimuluss=2.1
const epsilonn=1e-7
####################### c4 ####################

function c4_const(c4,c4_p)
    for i in 1:length(c4)
        c4[i]=0.1*c4_p[1]
    end
end


####################### c6 ####################

function c6_const(c6,c6_p)
    for i in 1:length(c6)
        c6[i]=c6_p[1]
    end
end

function c6_zeros(c6)
    for i in 1:length(c6)
        c6[i]=0.
    end
end



####################### c2 ####################

function c2_const(c2,c2_p)
    for i in 1:length(c2)
        c2[i]=c2_p[1]
    end
end


function c2_urgency(c2,c2_p)
    for i in 1:length(c2)
        c2[i]=c2_p[1]+0.1*c2_p[2]*(i-1)
    end
end

################# bias decision ##########

function bias_d_const(bias_d,bias_d_p)
    for i in 1:length(bias_d)
        bias_d[i]=bias_d_p[1]
    end
end

function bias_d_zeros(bias_d)
    for i in 1:length(bias_d)
        bias_d[i]=0.
    end
end


################# history bias decision ##########

function history_bias_d_const(hbias_d,bias_d_p)
    for i in 1:length(hbias_d)
        hbias_d[i]=bias_d_p[1]
    end
end

function history_bias_d_zeros(hbias_d)
    for i in 1:length(hbias_d)
        hbias_d[i]=0.
    end
end


##############################################3

function c2_kiani(c2,c2_p)
    for i in 1:length(c2)
        c2[i]=c2_p[i]
    end
end


###### initial condition#############################
function initial_condition(x0)
    #### initial condition
    #### x0_p[1] is the mean ###
    binN = length(x0)
    x0[Int( (binN-1)/2 )]=1
end



function initial_condition_bias(x0,bin_centers,x0_p)
    binN = length(x0)
    if x0_p[1]<bin_centers[1]
        x0[1]=1
    elseif  (x0_p[1]>bin_centers[end])
            x0[end]=1

    else
        if (x0_p[1] > bin_centers[1] && x0_p[1] < bin_centers[2])
            lp = 1; hp = 2;
        elseif (x0_p[1] > bin_centers[end-1] && x0_p[1] < bin_centers[end])
            lp = binN-1; hp = binN;
        else
            lp = floor(Int,((x0_p[1]-bin_centers[2])/dx)) + 2#find(bin_centers .<= sbins[k])[end]
            hp = ceil(Int,((x0_p[1]-bin_centers[2])/dx)) + 2#lp+1#Int(ceil((sbins[k]-bin_centers[2])/dx) + 1);
        end
        #println(k," ",lp," ",hp," ",sbin)
        if lp == hp
            x0[lp] = x0[lp] + 1
        else
            x0[hp] = x0[hp] + 1*(x0_p[1] - bin_centers[lp])/(bin_centers[hp] - bin_centers[lp])
            x0[lp] = x0[lp] + 1*(bin_centers[hp] - x0_p[1])/(bin_centers[hp] - bin_centers[lp])
        end
    end

end


function initial_condition_bias_hbias(x0,bin_centers,bias_total)

    #Overall bias x0_p[1]
    #hbias_x0 parameter x0_p[2]
    #historybias from data x0_p[3]

    Ntrials,binN = size(x0)

    for itrial in 1:Ntrials

        if bias_total[itrial]<bin_centers[1]
            x0[itrial,1]=1
        elseif  (bias_total[itrial]>bin_centers[end])
                x0[itrial,end]=1

        else
            if (bias_total[itrial] > bin_centers[1] && bias_total[itrial] < bin_centers[2])
                lp = 1; hp = 2;
            elseif (bias_total[itrial] > bin_centers[end-1] && bias_total[itrial] < bin_centers[end])
                lp = binN-1; hp = binN;
            else
                lp = floor(Int,((bias_total[itrial]-bin_centers[2])/dx)) + 2#find(bin_centers .<= sbins[k])[end]
                hp = ceil(Int,((bias_total[itrial]-bin_centers[2])/dx)) + 2#lp+1#Int(ceil((sbins[k]-bin_centers[2])/dx) + 1);
            end
            #println(k," ",lp," ",hp," ",sbin)
            if lp == hp
                x0[itrial,lp] = x0[lp] + 1
            else
                x0[itrial,hp] = x0[itrial,hp] + 1*(bias_total[itrial] - bin_centers[lp])/(bin_centers[hp] - bin_centers[lp])
                x0[itrial,lp] = x0[itrial,lp] + 1*(bin_centers[hp] - bias_total[itrial])/(bin_centers[hp] - bin_centers[lp])
            end
        end
    end
end




function initial_condition_bias_variance(x0,bin_centers,x0_p)
    #### we add variability and bias in the initial condition
    #### x0_p[1] is the mean, x0_p[2] is the std

    binN = length(bin_centers)
    sigma0=x0_p[2]
    n_sbins = 11#max(70, ceil(10*x0_p[2]/dx))
    swidth = 5.0*sigma0
    println("hola")
    println(swidth)
    println(n_sbins)
    sbinsize = swidth/n_sbins #
    base_sbins    = collect(-swidth:sbinsize:swidth)
    ps= exp.(-base_sbins.^2/(2*(x0_p[2]^2)))
    ps= ps/sum(ps);
    sbin_length = length(base_sbins)
    for k in 1:sbin_length
        sbin = (k-1)*sbinsize + x0_p[1] - swidth
        #println(sbin)
        if sbin <= bin_centers[1] #(bin_centers[1] + bin_centers[2])/2
            x0[1] = x0[1] + ps[k]
        elseif bin_centers[end] <= sbin#(bin_centers[end]+bin_centers[end-1])/2 <= sbins[k]
            x0[end] = x0[end] + ps[k]
        else
            if (sbin > bin_centers[1] && sbin < bin_centers[2])
                lp = 1; hp = 2;
            elseif (sbin > bin_centers[end-1] && sbin < bin_centers[end])
                lp = binN-1; hp = binN;
            else
                lp = floor(Int,((sbin-bin_centers[2])/dx)) + 2#find(bin_centers .<= sbins[k])[end]
                hp = ceil(Int,((sbin-bin_centers[2])/dx)) + 2#lp+1#Int(ceil((sbins[k]-bin_centers[2])/dx) + 1);
            end
            #println(k," ",lp," ",hp," ",sbin)
            if lp == hp
                x0[lp] = x0[lp] + ps[k]
            else
                x0[hp] = x0[hp] + ps[k]*(sbin - bin_centers[lp])/(bin_centers[hp] - bin_centers[lp])
                x0[lp] = x0[lp] + ps[k]*(bin_centers[hp] - sbin)/(bin_centers[hp] - bin_centers[lp])
            end
        end
    end

    return x0
end

#######################stimulus transformation####################
function st_linear(s,s_trans,st_p)
    for iframe in 1:length(s)
        s_trans[iframe]=s[iframe]*(st_p[1])
    end
end


function st_linear_adaptation(s,s_trans,st_p)
    for iframe in 1:length(s)
        s_trans[iframe]=s[iframe]*(st_p[1]-0.1*st_p[2]*(iframe-1) )
    end
end



function st_linear_adaptation_plot(s,s_trans,st_p,iframe)
    for i in 1:length(s)
        s_trans[i]=s[i]*(st_p[1]-0.1*st_p[2]*(iframe-1) )
    end
end


function st_linear_adaptation_exp(s,s_trans,st_p)
    for iframe in 1:length(s)
        s_trans[iframe]=s[iframe]*(st_p[1]*exp(st_p[2]*(iframe-1)))
    end
end



function st_linear_adaptation_exp_bias(s,s_trans,st_p)
    for iframe in 1:length(s)
        s_trans[iframe]=s[iframe]*(st_p[1]*exp(st_p[2]*(iframe-1)) -st_p[3])
    end
end


function st_linear_adaptation_exp_plot(s,s_trans,st_p,iframe)
    for i in 1:length(s)
        s_trans[i]=s[i]*(st_p[1]*exp(st_p[2]*(iframe-1)) )
    end
end


function st_logistic_frame(s,st_p,k)
    beta= sign(st_p[2])*maximum([epsilonn,abs(st_p[2]) ]) +epsilonn/10.0
    return (st_p[1]/k)*(1.0/(1+exp(-( beta*s  ) ) )-0.5 )

end

function st_logistic(stim,s_trans,st_p)
    beta= sign(st_p[2])*maximum([epsilonn,abs(st_p[2]) ]) +epsilonn/10.0
    println(beta)
    k=1.0/(1+exp(-beta*max_stimuluss) ) -0.5

    Nframes=length(stim)
    for iframe in 1:Nframes
        #s_trans[iframe]=(st_p[1]/k)*(1.0/(1+exp(-( beta*stim[iframe]  ) ) )-0.5 )
        s_trans[iframe]=(1.0/(1+exp(-( beta*stim[iframe]  ) ) )-0.5 )

    end
end

function st_logistic_bias(stim,s_trans,st_p)
    beta= sign(st_p[2])*maximum([epsilonn,abs(st_p[2]) ]) +epsilonn/10.0
    println(beta)
    k=1.0/(1+exp(-beta*max_stimuluss) ) -0.5

    Nframes=length(stim)
    for iframe in 1:Nframes
        if stim[iframe]>0
            s_trans[iframe]=st_logistic_frame(stim[iframe],[st_p[1],st_p[2]],k)
        else
            s_trans[iframe]=st_logistic_frame(stim[iframe],[st_p[1]-st_p[3],st_p[2]],k)
        end
    end
end


function st_logistic_adaptation(stim,s_trans,st_p)
    beta= sign(st_p[2])*maximum([epsilonn,abs(st_p[2]) ]) +epsilonn/10.0
    k=1.0/(1+exp(-beta*max_stimuluss) ) -0.5

    Nframes=length(stim)
    for iframe in 1:Nframes
        s_trans[iframe]=st_logistic_frame(stim[iframe],[st_p[1]-0.1*st_p[3]*(iframe-1),st_p[2]],k)
    end
end


function st_logistic_adaptation_plot(stim,s_trans,st_p,ifrmae)

    k=1.0/(1+exp(-beta*max_stimuluss) ) -0.5

    Nframes=length(stim)
    for i in 1:Nframes
        s_trans[i]=st_logistic_frame(stim[i],[st_p[1]-0.1*st_p[3]*(iframe-1),st_p[2]],k)
    end
end


function st_logistic_adaptation_bias(stim,s_trans,st_p)

    beta= sign(st_p[2])*maximum([epsilonn,abs(st_p[2]) ]) +epsilonn/10.0
    k=1.0/(1+exp(-beta*max_stimuluss) ) -0.5

    Nframes=length(stim)
    for iframe in 1:Nframes
        if stim[iframe]>0
            s_trans[iframe]=st_logistic_frame(stim[iframe],[st_p[1]-0.1*st_p[3]*(iframe-1),st_p[2]],k)
        else
            s_trans[iframe]=st_logistic_frame(stim[iframe],[st_p[1]-0.1*st_p[3]*(iframe-1)-st_p[4],st_p[2]],k)
        end
    end
end


function st_logistic_adaptation_bias_plot(stim,s_trans,st_p,iframe)

    beta= sign(st_p[2])*maximum([epsilonn,abs(st_p[2]) ]) +epsilonn/10.0
    k=1.0/(1+exp(-beta*max_stimuluss) ) -0.5

    Nframes=length(stim)
    for i in 1:Nframes
        if stim[iframe]>0
            s_trans[i]=st_logistic_frame(stim[i],[st_p[1]-0.1*st_p[3]*(iframe-1),st_p[2]],k)
        else
            s_trans[i]=st_logistic_frame(stim[i],[st_p[1]-0.1*st_p[3]*(iframe-1)-st_p[4],st_p[2]],k)
        end
    end
end




# function st_logistic(stim,s_trans,st_p)
#     Nframes=length(stim)
#     for iframe in 1:Nframes
#         s_trans[iframe]=2.0*(1/st_p[1])*(1.0/(1+exp(-( st_p[2]*st_p[1]*stim[iframe]  ) ) )-0.5 )
#     end
# end

#
# function st_logistic_adaptation(stim,s_trans,st_p)
#     Nframes=length(stim)
#     for iframe in 1:Nframes
#         s_trans[iframe]=2.0*(1/st_p[1])*(1.0/(1+exp(-( (st_p[2]-0.1*st_p[3]*(iframe-1))*st_p[1]*stim[iframe]  ) ) )-0.5 )
#         #s_trans[iframe]=2.0*(1.0/st_p[1])*(1.0/(1+exp(-( (1/(st_p[2]-0.1*st_p[3]*(iframe-1)) )*stim[iframe] ) ) )-0.5 )
#     end
# end
#
#
# function st_logistic_adaptation_plot(stim,s_trans,st_p,iframe)
#     Nframes=length(stim)
#     for i in 1:Nframes
#         s_trans[i]=2.0*(1/st_p[1])*(1.0/(1+exp(-( (st_p[2]-0.1*st_p[3]*(iframe-1))*st_p[1]*stim[i]  ) ) )-0.5 )
#     end
# end
#


#
# function st_logistic_bias(stim,s_trans,st_p)
#     Nframes=length(stim)
#     for iframe in 1:Nframes
#             if stim[iframe]-st_p[3] > 0
#                 s_trans[iframe]=2*st_p[1]*(1.0/(1+exp(-(st_p[2]*(stim[iframe]-st_p[3] ) ) ) ) -0.5 )
#             else
#                 #s_trans[iframe]=2*st_p[1]*(1./(1+exp(-(st_p[2]*(stim[iframe]-st_p[3] ) ) ) ) -0.5 )
#                 s_trans[iframe]=2*(st_p[1]-st_p[4])*(1.0/(1+exp(-( (st_p[2]-st_p[5])*(stim[iframe]-st_p[3] ) ) ) ) -0.5 )
#             end
#     end
# end

# function st_logistic_bias(stim,s_trans,st_p)
#     Nframes=length(stim)
#     for iframe in 1:Nframes
#             if stim[iframe] > 0
#                 s_trans[iframe]=2.0*(1/st_p[1])*(1.0/(1+exp(-( st_p[2]*st_p[1]*stim[iframe]  ) ) )-0.5 )
#
#                 #s_trans[iframe]=2*(1.0/st_p[1])*(1.0/(1+exp(-( (1./st_p[2])*(stim[iframe] ) ) ) ) -0.5 )
#             else
#                 #s_trans[iframe]=2*st_p[1]*(1./(1+exp(-(st_p[2]*(stim[iframe]-st_p[3] ) ) ) ) -0.5 )
#                 #s_trans[iframe]=2*( (1.0/st_p[1])-st_p[3])*(1.0/(1+exp(-( ( (1./st_p[2])-st_p[4])*(stim[iframe] ) ) ) ) -0.5 )
#                 s_trans[iframe]=2.0*( 1/st_p[1] -st_p[3] )*(1.0/(1+exp(-( (st_p[2]-st_p[4])*st_p[1]*stim[iframe]  ) ) )-0.5 )
#             end
#     end
# end
#
#
#
# function st_logistic_adaptation_bias(stim,s_trans,st_p)
#     Nframes=length(stim)
#     for iframe in 1:Nframes
#             if stim[iframe] > 0
#                 #s_trans[iframe]=2*(1.0/st_p[1])*(1.0/(1+exp(-( (1./(st_p[2]-0.1*st_p[3]*(iframe-1))    )*(stim[iframe] ) ) ) ) -0.5 )
#                 s_trans[iframe]=2.0*(1/st_p[1])*(1.0/(1+exp(-( (st_p[2]-0.1*st_p[3]*(iframe-1))*st_p[1]*stim[iframe] )  ) )-0.5 )
#
#             else
#                 #s_trans[iframe]=2*( (1.0/st_p[1])-st_p[4])*(1.0/(1+exp(-( ( (1./ (st_p[2]-0.1*st_p[3]*(iframe-1))  )-st_p[5])*(stim[iframe] ) ) ) ) -0.5 )
#                 s_trans[iframe]=2.0*(1/st_p[1]-st_p[4])*(1.0/(1+exp(-( ( (st_p[2]-0.1*st_p[3]*(iframe-1)) -st_p[5] )*st_p[1]*stim[iframe] )  ) )-0.5 )
#
#             end
#     end
# end
#
#
#
#
# function st_logistic_adaptation_bias_plot(stim,s_trans,st_p,iframe)
#     Nframes=length(stim)
#     for i in 1:Nframes
#             if stim[i] > 0
#                 #s_trans[iframe]=2*(1.0/st_p[1])*(1.0/(1+exp(-( (1./(st_p[2]-0.1*st_p[3]*(iframe-1))    )*(stim[iframe] ) ) ) ) -0.5 )
#                 s_trans[i]=2.0*(1/st_p[1])*(1.0/(1+exp(-( (st_p[2]-0.1*st_p[3]*(iframe-1))*st_p[1]*stim[i]   ) ) ) -0.5 )
#
#             else
#                 #s_trans[iframe]=2*( (1.0/st_p[1])-st_p[4])*(1.0/(1+exp(-( ( (1./ (st_p[2]-0.1*st_p[3]*(iframe-1))  )-st_p[5])*(stim[iframe] ) ) ) ) -0.5 )
#                 s_trans[i]=2.0*(1/st_p[1]-st_p[4])*(1.0/(1+exp(-( ( (st_p[2]-0.1*st_p[3]*(iframe-1)) -st_p[5] )*st_p[1]*stim[i] ) ) )-0.5 )
#
#             end
#     end
# end
#





##############sigma ########
function sigma_const(sigma,sigma_p)
    for iframe in 1:length(sigma)
        sigma[iframe]=sigma_p[1]
    end
    #return s_trans
end


function sigma_delay_kiani(sigma,sigma_p)
    for iframe in 1:length(sigma)
        if iframe%2==0
            sigma[iframe]=sigma_p[1]/sqrt(2.)
        else
            sigma[iframe]=sigma_p[1]
        end
    end
    #return s_trans
end



#######################dv  read out####################

function read_out_soft(read_out,bins,ro_p)
    for ibin in 1:length(bins)
        read_out[ibin]=1.0/(1+exp.(-10*ro_p[1]*(bins[ibin]-ro_p[2])))
    end

end




function read_out_perfect(read_out,binN)
    read_out[1:binN]=0
    read_out[binN+1]=0.5
    read_out[binN+2:end]=1
end



function potential_DW(param,x)
    return -param[0]-param[1]*x+pram[2]*x^3
end

function read_out_TW(ro,bin_centers,c2,c4,c6)
    delta=c4^2+4*c2*c6
    if  (c2<0) & (c4<0) & (c6>0) & (delta>0)
        b=sqrt( (-c4-sqrt(delta))/(2*c6)   )
        i=1
        accu=0.
        while (i<=length(ro)) & (bin_centers[i]<-b)
            ro[i]=0
            i=i+1
        end
        while (i<=length(ro)) & (bin_centers[i]<b)
            ro[i]=0.5
            i=i+1
        end
        while i<=length(ro)
            ro[i]=1
            i=i+1
        end
    elseif (c2<0) & (c4>0) & (c6>0)
        ro[:]=0.5

    else
        N=(length(bin_centers)-1)/2
        ro[:]=vcat(zeros(N),[0.5],ones(N))
    end
    return ro
end


function read_out_DW(ro,bin_centers,c2,c4)
    if  (c2<0) & (c4<0)

        b=sqrt(c2/c4)
        i=1
        accu=0.
        while  (i<=length(ro)) & (bin_centers[i]<-b)
            ro[i]=0
            i=i+1
        end
        while  (i<=length(ro)) & (bin_centers[i]<-b)
            ro[i]=0.5
            i=i+1
        end
        while i<=length(ro)
            ro[i]=1
            i=i+1
        end
    elseif (c2<0) & (c4>0)
        ro[:]=0.5

    else
        N=(length(bin_centers)-1)/2
        ro[:]=vcat(zeros(N),[0.5],ones(N))
    end
    return ro
end
