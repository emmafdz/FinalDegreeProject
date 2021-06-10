function make_dict2(args,x)
    nargs = 0
    for i in [1:length(args);]
        if typeof(args[i])==String # if the entry in args is a string, then there's one corresponding scalar entry in x0
            nargs += 1
        else
            nargs += args[i][2]    # otherwise, the entry in args should be a  [varnamestring, nvals] vector,
            # indicating that the next nvals entries in x0 are all a single vector, belonging to variable
            # with name varnamestring.
        end
    end
    if nargs != length(x)
        error("Oy! args and x must indicate the same total number of variables!")
    end

    i=1
    param=Dict()
    for  (iarg,arg) in enumerate(args)
        if typeof(arg)==String
            param[arg]=x[i:i]
            i+=1
        else
            j=arg[2]
            param[arg[1]]=x[i:i+j-1]
            i+=j
        end
    end

    return param
end

function arg_labels(args)
    labels=[]
    for arg in args
        if arg[2]!=0
            if length(arg[2])==1
                append!(labels,[arg[1]])
            else
                for ip in arg[2]
                    append!(labels, [ arg[1]*string(ip) ])
                end
            end
        end
    end

    return labels
end


function make_dict(xargs,x,zargs,z)
    """
    Function to combine paramters to fit (args,x)
    and constants (zargs,z)
    """

    args=[]
    for iarg in 1:length(xargs)
        if xargs[iarg][2]==0
            ii=length(zargs[iarg][2])
            if ii==1
                push!( args,xargs[iarg][1] )
            else
                push!( args,( xargs[iarg][1],ii) )
            end

        elseif zargs[iarg][2]==0
            ii=length(xargs[iarg][2])
            if ii==1
                push!( args,xargs[iarg][1] )
            else
                push!( args,( xargs[iarg][1],ii) )
            end


        else
            push!( args,(xargs[iarg][1],length(zargs[iarg][2])+length(xargs[iarg][2])) )
        end

    end

    y=zeros(typeof(x[1]),length(x)+length(z))
    ix=1
    iz=1
    iy=1
    for iarg in 1:length(xargs)

        if xargs[iarg][2]==0

            ii=length(zargs[iarg][2])
            # for ind in 1:ii
            #     push!(y,z[iz:iz+ind-1])
            # end

            y[iy:iy+ii-1]=z[iz:iz+ii-1]
            iz+=ii
            iy+=ii

        elseif zargs[iarg][2]==0

            ii=length(xargs[iarg][2])

            # for ind in 1:ii
            #     push!(y,z[iz:iz+ind-1])
            # end

            y[iy:iy+ii-1]=x[ix:ix+ii-1]
            ix+=ii
            iy+=ii
        else
            ii=length(xargs[iarg][2])+length(zargs[iarg][2])
            iix=length(xargs[iarg][2])
            iiz=length(zargs[iarg][2])

            y[iy+xargs[iarg][2]-1]=x[ix:ix+iix-1]
            y[iy+zargs[iarg][2]-1]=z[iz:iz+iiz-1]

            ix+=iix
            iy+=ii
            iz+=iiz
        end

    end


    nargs = 0
    for i in [1:length(args);]
        if typeof(args[i])==String # if the entry in args is a string, then there's one corresponding scalar entry in x0
            nargs += 1
        else
            nargs += args[i][2]    # otherwise, the entry in args should be a  [varnamestring, nvals] vector,
            # indicating that the next nvals entries in x0 are all a single vector, belonging to variable
            # with name varnamestring.
        end
    end
    if nargs != length(y)
        error("Oy! args and x must indicate the same total number of variables!")
    end

    i=1
    param=Dict()
    for  (iarg,arg) in enumerate(args)
        if typeof(arg)==String
            param[arg]=y[i:i]
            i+=1
        else
            j=arg[2]
            param[arg[1]]=y[i:i+j-1]
            i+=j
        end
    end

    return param
end

xargs=[ ("c2",[1,2]),("c4",0),("st",[2,3]),("x0",[1,2]),("bias_d",[1]),("hbias_d",[1]),("sigma_a",[1])]
zargs=[("c2",0),("c4",[1]),("st",[1,4]),("x0",0),("bias_d",0),("hbias_d",0),("sigma_a",0)]

x=[1,2,5,6,8,9,10,11,12]
z=[3,4,7]

args=[]
for iarg in 1:length(xargs)
    if xargs[iarg][2]==0
        ii=length(zargs[iarg][2])
        if ii==1
            push!( args,xargs[iarg][1] )
        else
            push!( args,( xargs[iarg][1],ii) )
        end

    elseif zargs[iarg][2]==0
        ii=length(xargs[iarg][2])
        if ii==1
            push!( args,xargs[iarg][1] )
        else
            push!( args,( xargs[iarg][1],ii) )
        end


    else
        push!( args,(xargs[iarg][1],length(zargs[iarg][2])+length(xargs[iarg][2])) )
    end

end


function get_model_f(model,reg; simulations=false)
    global model_f

    if (model=="StLogisticAdaptatioAbDwX0MubReg"*string(reg) )

        model_f=Dict(:c2_f=>c2_const,:c4_f=>c4_const,:c6_f=>c6_zeros,:bias_d_f=>bias_d_const,
        :hbias_d_f=>history_bias_d_const,:st_f=>st_logistic_adaptation_bias,:x0_f=>initial_condition_bias_hbias,
        :ro_f=>read_out_perfect,:lambda=>reg)


    elseif (model=="StLinearPiReg"*string(reg) )
        model_f=Dict(:bias_d_f=>bias_d_const,:hbias_d_f=>history_bias_d_const,
        :st_f=>st_linear,:x0_f=>initial_condition_bias_hbias,
        :ro_f=>read_out_perfect,:lambda=>reg)


    elseif (model=="StLogisticPiReg"*string(reg) )

        model_f=Dict(:bias_d_f=>bias_d_const,:hbias_d_f=>history_bias_d_const,
        :st_f=>st_logistic,:x0_f=>initial_condition_bias_hbias,
        :ro_f=>read_out_perfect,:lambda=>reg)

    elseif (model=="StLogisticAdaptationDwReg"*string(reg))
        model_f=Dict(:c2_f=>c2_const,:c4_f=>c4_const,:c6_f=>c6_zeros,:bias_d_f=>bias_d_const,
        :hbias_d_f=>history_bias_d_const,:st_f=>st_logistic_adaptation,:x0_f=>initial_condition_bias_hbias,
        :ro_f=>read_out_perfect,:lambda=>reg)

    elseif (model=="StLinearAdaptationExpDwReg"*string(reg))
        model_f=Dict(:c2_f=>c2_const,:c4_f=>c4_const,:c6_f=>c6_zeros,:bias_d_f=>bias_d_const,
        :hbias_d_f=>history_bias_d_const,:st_f=>st_linear_adaptation_exp,:x0_f=>initial_condition_bias_hbias,
        :ro_f=>read_out_perfect,:lambda=>reg)

    elseif (model=="StLinearDwReg"*string(reg))
        model_f=Dict(:c2_f=>c2_const,:c4_f=>c4_const,:c6_f=>c6_zeros,:bias_d_f=>bias_d_const,
        :hbias_d_f=>history_bias_d_const,:st_f=>st_linear,:x0_f=>initial_condition_bias_hbias,
        :ro_f=>read_out_perfect,:lambda=>reg)

    elseif (model=="StLogisticDwReg"*string(reg))
        model_f=Dict(:c2_f=>c2_const,:c4_f=>c4_const,:c6_f=>c6_zeros,:bias_d_f=>bias_d_const,
        :hbias_d_f=>history_bias_d_const,:st_f=>st_logistic,:x0_f=>initial_condition_bias_hbias,
        :ro_f=>read_out_perfect,:lambda=>reg)

    elseif (model=="StLogisticAdaptationC4Reg"*string(reg))
        model_f=Dict(:c2_f=>c2_const,:c4_f=>c4_const,:c6_f=>c6_zeros,:bias_d_f=>bias_d_const,
        :hbias_d_f=>history_bias_d_const,:st_f=>st_logistic_adaptation,:x0_f=>initial_condition_bias_hbias,
        :ro_f=>read_out_perfect,:lambda=>reg)

    elseif (model=="StLogisticC4Reg"*string(reg))
        model_f=Dict(:c2_f=>c2_const,:c4_f=>c4_const,:c6_f=>c6_zeros,:bias_d_f=>bias_d_const,
        :hbias_d_f=>history_bias_d_const,:st_f=>st_logistic,:x0_f=>initial_condition_bias_hbias,
        :ro_f=>read_out_perfect,:lambda=>reg)


    else:
        println("not found")
    end
    if simulations
        model_f[:ro_f]=read_out_perfect_sim
    end
    return model_f

end


function get_args_consts(model,reg)
    args=[]
    consts=[]
    lb=[]
    ub=[]
    z=[]
    if (model=="StLogisticAdaptatioDwReg"*string(reg))
        args=[ ("c2",[1]),("c4",[1]),("st",[1,2,3]),("x0",0),("bias_d",0),("hbias_d",0),("sigma_a",0)]
        consts=[("c2",0),("c4",0),("st",0),("x0",[1,2]),("bias_d",[1]),
        ("hbias_d",[1]),("sigma_a",[1])]

        z=1.0*[0.0,0.0,0.0,0.0,1.0]
        lb=[-50., 0.0,  0.0,0.0,-50.0 ]
        ub=[50., 50.,  50., 50.0,  50.0]

    elseif (model=="StLogisticAdaptationC4Reg"*string(reg))
        args=[ ("c2",0),("c4",[1]),("st",[1,2,3]),("x0",0),("bias_d",0),("hbias_d",0),("sigma_a",0)]
        consts=[("c2",[1]),("c4",0),("st",0),("x0",[1,2]),("bias_d",[1]),
        ("hbias_d",[1]),("sigma_a",[1])]

        z=1.0*[0.0,0.0,0.0,0.0,0.0,1.0]
        lb=[0., 0.0,  0.0,-50.0]
        ub=[50., 50.,  50., 50.0]

    elseif (model=="StLogisticC4Reg"*string(lambda))
        args=[ ("c2",0),("c4",[1]),("st",[1,2]),("x0",0),("bias_d",0),("hbias_d",0),("sigma_a",0)]
        consts=[("c2",[1]),("c4",0),("st",0),("x0",[1,2]),("bias_d",[1]),
        ("hbias_d",[1]),("sigma_a",[1])]

        z=1.0*[0.0,0.0,0.0,0.0,0.0,1.0]
        lb=[0., 0.0,  0.0]
        ub=[50., 50.,  50.]

    else
        println("model not found")
    end

    return args,consts,z,lb,ub
end

function create_stimuli(mu,sigmas,N,Nframes)
    m=mu*ones(N)
    m[Int(floor(N/2)):end]=-m[Int(floor(N/2)):end]
    stim=randn(N,Nframes)
    sigmas_vec=[]
    i_sigmas=[]
    for i in 1:Int(N/length(sigmas))
        sigmas_vec=vcat(sigmas_vec,sigmas)
        i_sigmas=vcat(i_sigmas,0:length(sigmas)-1)
    end

    for i in 1:N
        stim[i,:]=m[i]+sigmas_vec[i]*(stim[i,:] -mean(stim[i,:]) )/std(stim[i,:])
    end
    return stim,convert(Array{Float64,1},sigmas_vec)[1:N],convert(Array{Float64,1},i_sigmas)[1:N],sign.(m)

end
