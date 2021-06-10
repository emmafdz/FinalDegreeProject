

# function simulation_1frame_traces(coef_stim::Array{Float64}, internal_noise:: Array{Float64},T:: Float64,tau::Float64,x0::Float64)
function simulation_1frame_traces(coef_stim, internal_noise,T,tau,x0)
    a=size(internal_noise)
    #println(a)
    NT=a[2]
    Nstim=a[1]
    deltat=T/NT
    dt_sim=deltat/tau
    dt_sqrt=sqrt(dt_sim)
    d=zeros(Nstim)
    x=zeros(Nstim,NT)
    for itrial in 1:Nstim
         for it in 1:NT-1
             x[itrial,it+1]=x[itrial,it]-( -coef_stim[1]-coef_stim[2]*x[itrial,it]+coef_stim[3]*x[itrial,it]^3)*dt_sim+dt_sqrt*internal_noise[itrial,it]
         end
    end

    return x
 end


 function simulation_1frame(coef_stim, internal_noise,T,tau,x0)
     a=size(internal_noise)
     println(a)
     NT=a[2]
     Nstim=a[1]
     deltat=T/NT
     dt_sim=deltat/tau
     dt_sqrt=sqrt(dt_sim)
     d=zeros(Nstim)
     x=zeros(Nstim,NT)
     for itrial in 1:Nstim
         x=x0
         for it in 1:NT-1
             x=x-( -coef_stim[1]-coef_stim[2]*x+coef_stim[3]*x^3)*dt_sim+dt_sqrt*internal_noise[itrial,it]
         end
         if x<0
             d[itrial]=-1
        else
            d[itrial]=1
        end
     end

     return d
end


function simulation_Nframe_traces(coef_stim,stim,internal_noise,T,tau,x0,urgency=0.)


    # println("hola")
    # println("puta")
    aux2=size(internal_noise)
    #println(a)
    aux=size(stim)
    Nframes=aux[2]
    Nstim=aux[1]
    NT=aux2[2]/Nframes
    deltat=T/NT
    dt_sim=deltat/tau
    dt_sqrt=sqrt(dt_sim)
    d=zeros(Nstim)
    x=zeros(Nstim,Nframes*NT+1)
    println(Nframes)
    for itrial in 1:Nstim
        it=2
        x[itrial,1]=x0

        for iframe in 1:Nframes
            u=urgency*(iframe-1)
            for i in 1:NT
                # println("it: ",it-1, " itrial ", itrial," iframe ", iframe)
                x[itrial,it]=x[itrial,it-1]-( -stim[itrial,iframe]-(coef_stim[2]+u)*x[itrial,it-1]+coef_stim[3]*x[itrial,it-1]^3)*dt_sim+dt_sqrt*internal_noise[itrial,it-1]
                it=it+1
            end
        end
    end

      return x
end


function simulation_Nframe(coef_stim,stim,internal_noise,T,tau,x0;urgency=0.)


    # println("hola")
    # println("puta")
    aux2=size(internal_noise)
    #println(a)
    aux=size(stim)
    Nframes=aux[2]
    Nstim=aux[1]
    NT=aux2[2]/Nframes
    deltat=T/NT
    dt_sim=deltat/tau
    dt_sqrt=sqrt(dt_sim)
    d=zeros(Float64,Nstim)
    # println(Nframes)
    # println("hola soc ",dt_sim)
    for itrial in 1:Nstim
        it=2
        x=x0

        for iframe in 1:Nframes
            u=urgency*(iframe-1)
            for i in 1:NT
                # println("it: ",it-1, " itrial ", itrial," iframe ", iframe)
                x=x-( -stim[itrial,iframe]-(coef_stim[2]+u)*x+coef_stim[3]*x^3)*dt_sim+dt_sqrt*internal_noise[itrial,it-1]
                it=it+1
            end
        end
        d[itrial]=sign(x)
    end

      return d
end


###########URGENCY+STIM_TRANSFORM+SOFT_BOUND################
function read_out_soft_sim(x,ro_p)
    pr=1./(1+exp.(-10*ro_p[1]*(x-ro_p[2])))
    if rand()<pr
        d=1
    else
        d=-1
    end
    return d
end

function read_out_perfect_sim(x)
    return sign(x)
end


function read_out_TW_sim(x,c2,c4,c6)
    delta=c4^2+4*c2*c6
    #println(delta," ", delta>0)

    if  (c2<0) & (c4<0) & (c6>0) & (delta>0)
        b=sqrt( (-c4-sqrt(delta))/(2*c6)   )
        if abs(x)<b
            a=sign( rand() -0.5)
            return sign( rand() -0.5)
        else
            return sign(x)
        end
    else
        return sign(x)
    end
end


function simulation_Nframe_general(param,stim,internal_noise,T,tau,model_f)
    """
    param: Dict with the parameter models
    internal_noise: interncal_noise= sigma*chi(t)
    T is total time T=Tframe*Nframes
    """
    Nstim,Nframes=size(stim)
    aux2=size(internal_noise)
    NT=aux2[2]
    NTframes=NT/Nframes
    deltat=T/NT
    dt_sim=deltat/tau
    dt_sqrt=sqrt(dt_sim)
    d=zeros(Float64,Nstim)
    xfinal=zeros(Float64,Nstim)
    c2=zeros(Nframes)
    c4=zeros(Nframes)
    c6=zeros(Nframes)
    #println("hola ",dt_sim," ", deltat," ",T," ", NT)
    model_f[:c2_f](c2,param["c2"])
    model_f[:c4_f](c4,param["c4"])
    if "c6" in keys(param)
        model_f[:c6_f](c6,param["c6"])
    else
        model_f[:c6_f](c6)
    end
    xfinal=zeros(Nstim)
    hbias_d=param["hbias_d"][1]
    bias_d=param["bias_d"][1]

    s=zeros(Nframes)
    hb=0
    #println(hb,hbias_d)
    for itrial in 1:Nstim
        #println(itrial)
        it=2
        x=param["x0"][1]+hb*param["x0"][1]

        model_f[:st_f](stim[itrial,:],s,param["st"])
        #println(s)
        for iframe in 1:Nframes
            for i in 1:NTframes

                #println("it: ",it-1, " itrial ", itrial," iframe ", iframe)
                x=x-(-(s[iframe]+hb*hbias_d+bias_d)-c2[iframe]*x+c4[iframe]*x^3+c6[iframe]*x^5)*dt_sim+dt_sqrt*internal_noise[itrial,it-1]
                it=it+1
            end
        end
        xfinal[itrial]=x
        if model_f[:ro_f]==read_out_TW_sim
            d[itrial]=model_f[:ro_f](x,c2[end],c4[end],c6[end])
        elseif model_f[:ro_f]==read_out_perfect_sim
            d[itrial]=model_f[:ro_f](x)
        else
            d[itrial]=model_f[:ro_f](x,param["ro"])
        end
        #hb=d[itrial]
    end
    return d,xfinal
end



function simulation_Nframe_general_trace(param,stim,internal_noise,T,tau,model_f)
    """
    T is total time T=Tframe*Nframes
    """
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
    #println("hola ",dt_sim," ", deltat," ",T," ", NT)
    model_f[:c2_f](c2,param["c2"])
    model_f[:c4_f](c4,param["c4"])
    if "c6" in keys(param)
        model_f[:c6_f](c6,param["c6"])
    else
        model_f[:c6_f](c6)
    end
    hbias_d=param["hbias_d"][1]
    bias_d=param["bias_d"][1]
    # println(Nframes)
    # println("hola soc ",dt_sim)
    s=zeros(Nframes)
    hb=0 ###########canvia això per posar histroy bias
    #println(hb,hbias_d)
    Bound=100
    for itrial in 1:Nstim
        it=2
        x[itrial,1]=param["x0"][1]+hb*param["x0"][1]

        model_f[:st_f](stim[itrial,:],s,param["st"])
        #println(s)
        for iframe in 1:Nframes
            for i in 1:NTframes

                # println("it: ",it-1, " itrial ", itrial," iframe ", iframe)
                if abs(x[itrial,it-1])>Bound
                    x[itrial,it]=x[itrial,it-1]
                else

                    x[itrial,it]=x[itrial,it-1]-(-(s[iframe]+hb*hbias_d+bias_d)-c2[iframe]*x[itrial,it-1]+c4[iframe]*x[itrial,it-1]^3+c6[iframe]*x[itrial,it-1]^5)*dt_sim+dt_sqrt*internal_noise[itrial,it-1]
                end
                it=it+1

            end
        end
        if model_f[:ro_f]==read_out_TW_sim
            d[itrial]=model_f[:ro_f](x[itrial,end],c2[end],c4[end],c6[end])
        elseif model_f[:ro_f]==read_out_perfect_sim
            d[itrial]=model_f[:ro_f](x[itrial,end])
        else
            d[itrial]=model_f[:ro_f](x[itrial,end],param["ro"])
        end
        #hb=d[itrial]
    end
    return d,x
end



function simulation_Nframe_PI(param,stim,internal_noise,T,tau,model_f)

    Nstim,Nframes=size(stim)
    aux2=size(internal_noise)
    NT=aux2[2]
    NTframes=NT/Nframes
    deltat=T/NT
    dt_sim=deltat/tau
    dt_sqrt=sqrt(dt_sim)
    d=zeros(Float64,Nstim)
    x=zeros(Float64,Nstim,NT+1)

    s=zeros(Nframes)
    hb=0 ###########canvia això per posar histroy bias
    #println(hb,hbias_d)
    for itrial in 1:Nstim
        it=2
        x[itrial,1]=param["x0"][1]#+hb*param["x0"][1]

        model_f[:st_f](stim[itrial,:],s,param["st"])
        #println(s)
        for iframe in 1:Nframes
            for i in 1:NTframes
                x[itrial,it]=x[itrial,it-1]+s[iframe]*dt_sim+dt_sqrt*internal_noise[itrial,it-1]
                it=it+1
            end
        end
        d[itrial]=sign(x[itrial,end])

    end
    return d,x
end



function DW_potential_prima(x,coef)
    return -coef[1]-coef[2]*x+coef[3]*x^3
end


function compute_kernel(stim,d)
    right=find(x->x==1,d)
    left=find(x->x==-1,d)
    aux=size(stim)
    kernel=zeros(aux[2])
    for iframe in 1:aux[2]
        println(iframe)
        r=roc(stim[left,iframe],stim[right,iframe])
        kernel[iframe]=auc(r)
    end
    return kernel
end



###### Simulations DDMA ####
function simulation_Nframe_general_DDMA_trace(param,stim,internal_noise,T,tau,model_f)
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
    hbias_d=param["hbias_d"][1]
    bias_d=param["bias_d"][1]
    # println(Nframes)
    # println("hola soc ",dt_sim)
    s=zeros(Nframes)
    hb=0 ###########canvia això per posar histroy bias
    #println(hb,hbias_d)
    Bound=param["B"]
    for itrial in 1:Nstim
        it=2
        x[itrial,1]=param["x0"][1]+hb*param["x0"][1]

        model_f[:st_f](stim[itrial,:],s,param["st"])
        #println(s)
        for iframe in 1:Nframes
            for i in 1:NTframes

                if abs(x[itrial,it-1])>Bound
                    x[itrial,it]=x[itrial,it-1]
                else
                    x[itrial,it]=x[itrial,it-1]-(-(s[iframe]+hb*hbias_d+bias_d))*dt_sim+dt_sqrt*internal_noise[itrial,it-1]
                end
                it=it+1

            end
        end

        if model_f[:ro_f]==read_out_perfect_sim
            d[itrial]=model_f[:ro_f](x[itrial,end])
        else
            d[itrial]=model_f[:ro_f](x[itrial,end],param["ro"])
        end
    end
    return d,x
end

###### Simulations DDMA ####
function simulation_Nframe_general_DDMA(param,stim,internal_noise,T,tau,model_f)
    #println("hola")
    Nstim,Nframes=size(stim)
    aux2=size(internal_noise)
    NT=aux2[2]
    NTframes=NT/Nframes
    deltat=T/NT
    dt_sim=deltat/tau
    dt_sqrt=sqrt(dt_sim)
    d=zeros(Float64,Nstim)
    x=0.0
    xfinal=zeros(Nstim)

    hbias_d=param["hbias_d"][1]
    bias_d=param["bias_d"][1]
    # println(Nframes)
    # println("hola soc ",dt_sim)
    s=zeros(Nframes)
    hb=0 ###########canvia això per posar histroy bias
    #println(hb,hbias_d)
    Bound=param["B"]
    for itrial in 1:Nstim
        it=2
        x=param["x0"][1]+hb*param["x0"][1]

        model_f[:st_f](stim[itrial,:],s,param["st"])
        #println(s)
        for iframe in 1:Nframes
            for i in 1:NTframes

                if abs(x)>Bound
                    x=x
                else
                    x=x-(-(s[iframe]+hb*hbias_d+bias_d))*dt_sim+dt_sqrt*internal_noise[itrial,it-1]
                end
                it=it+1

            end
        end

        if model_f[:ro_f]==read_out_perfect_sim
            d[itrial]=model_f[:ro_f](x)
        else
            d[itrial]=model_f[:ro_f](x,param["ro"])
        end
        xfinal[itrial]=x
    end
    return d,xfinal
end


###### Simulations DDMR ####
function simulation_Nframe_general_DDMR(param,stim,internal_noise,T,tau,model_f)
    #println("hola")
    Nstim,Nframes=size(stim)
    aux2=size(internal_noise)
    NT=aux2[2]
    NTframes=NT/Nframes
    deltat=T/NT
    dt_sim=deltat/tau
    dt_sqrt=sqrt(dt_sim)
    d=zeros(Float64,Nstim)
    x=0.0
    xfinal=zeros(Nstim)

    hbias_d=param["hbias_d"][1]
    bias_d=param["bias_d"][1]
    # println(Nframes)
    # println("hola soc ",dt_sim)
    s=zeros(Nframes)
    hb=0 ###########canvia això per posar histroy bias
    #println(hb,hbias_d)
    Bound=param["B"]
    for itrial in 1:Nstim
        it=2
        x=param["x0"][1]+hb*param["x0"][1]

        model_f[:st_f](stim[itrial,:],s,param["st"])
        #println(s)
        for iframe in 1:Nframes
            for i in 1:NTframes

                if x>Bound
                    x=Bound
                elseif  x<-Bound
                    x=-Bound
                end

                x=x-(-(s[iframe]+hb*hbias_d+bias_d))*dt_sim+dt_sqrt*internal_noise[itrial,it-1]
                it=it+1

            end
        end

        if model_f[:ro_f]==read_out_perfect_sim
            d[itrial]=model_f[:ro_f](x)
        else
            d[itrial]=model_f[:ro_f](x,param["ro"])
        end
        xfinal[itrial]=x
    end
    return d,xfinal
end



function simulation_Nframe_general_DDMR_trace(param,stim,internal_noise,T,tau,model_f)
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
    hbias_d=param["hbias_d"][1]
    bias_d=param["bias_d"][1]
    # println(Nframes)
    # println("hola soc ",dt_sim)
    s=zeros(Nframes)
    hb=0 ###########canvia això per posar histroy bias
    #println(hb,hbias_d)
    Bound=param["B"]
    for itrial in 1:Nstim
        it=2
        x[itrial,1]=param["x0"][1]+hb*param["x0"][1]

        model_f[:st_f](stim[itrial,:],s,param["st"])
        #println(s)
        for iframe in 1:Nframes
            for i in 1:NTframes

                if x[itrial,it-1]>Bound
                    x[itrial,it-1]=Bound
                elseif  x[itrial,it-1]<-Bound
                    x[itrial,it-1]=-Bound
                end

                x[itrial,it]=x[itrial,it-1]-(-(s[iframe]+hb*hbias_d+bias_d))*dt_sim+dt_sqrt*internal_noise[itrial,it-1]
                it=it+1

            end
        end

        if model_f[:ro_f]==read_out_perfect_sim
            d[itrial]=model_f[:ro_f](x[itrial,end])
        else
            d[itrial]=model_f[:ro_f](x[itrial,end],param["ro"])
        end
    end
    return d,x
end
