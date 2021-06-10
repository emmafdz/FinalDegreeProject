'''

@author: genis
'''

import numpy as np
from scipy.optimize import curve_fit
import numpy as np
import math
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression

from scipy.special import erf
import statsmodels.api as sm
from scipy.stats import norm


def kernel_logistic(stim, d,Nboot=500):
    '''
    Computes the PK and the standard error as a LogisticRegression.
    Standard error is computed using bootstrap.
    inputs:
    stim: 2D-Array of stimulus
    d: 1-D array of decisions (-1,1)

    outputs:
    pk and pk std error
    '''

    Nstim,Nframes=np.shape(stim)
    pk_boots=np.zeros((Nboot,len(stim[1])))

    INDEXS=np.random.randint(0,Nstim,(Nboot,Nstim))

    C=10000
    clf_l2_LR = LogisticRegression(C=C, penalty='l2')

    if not all(np.unique(d)==[0,1]):
        d=(d+1)/2


    for i,index in enumerate(INDEXS):
        clf_l2_LR.fit(stim[index],d[index])
        pk_boots[i]=clf_l2_LR.coef_[0]

    pk=np.mean(pk_boots,axis=0)
    pk_std=np.std(pk_boots,axis=0)

    return pk, pk_std

def kernel_error(stim,d,error=False,Nboot=500):
    '''
    Computes the kernel and the standard error with bootstrap.
    inputs:
    stim: 2D-Array of stimulus
    d: 1-D array of decisions (-1,1)

    outputs:
    kernel: Dictionary with kernel and error_kernel
    '''
    Nframe=len(stim[0])
    Nstim=len(stim)
    kernel={'kernel':np.zeros(Nframe),'error':np.zeros(Nframe)}

    if not error:
        aux_kernel=np.zeros(Nframe)
        for iframe in range(Nframe):
            fpr,tpr,_=roc_curve(d,stim[:,iframe])
            aux_kernel[iframe] = auc(fpr, tpr)
        kernel['kernel']=aux_kernel
        return kernel
    else:

        aux_kernel=np.zeros((Nframe,Nboot))
        indexs=np.random.randint(0,Nstim,(Nboot,Nstim))


        for iboot in range(Nboot):
            if iboot%100==0:
                print(iboot)
            for iframe in range(Nframe):
                fpr,tpr,_=roc_curve(d[indexs[iboot]],stim[indexs[iboot],iframe])
                aux_kernel[iframe][iboot] = auc(fpr, tpr)


        for iframe in range(Nframe):
            kernel['kernel'][iframe]=np.mean(aux_kernel[iframe])
            kernel['error'][iframe]=np.std(aux_kernel[iframe])

        return kernel


def kernel_shuffle(stim,d,Nboot=500):
    '''
    Computes the kernel and the standard error for
    shuffle choices.
    inputs:
    stim: 2D-Array of stimulus
    d: 1-D array of decisions (-1,1)

    outputs:
    kernel: Dictionary with kernel and error_kernel
    '''
    Nframe=len(stim[0])
    Nstim=len(stim)
    kernel={'kernel':np.zeros(Nframe),'error':np.zeros(Nframe)}
    aux_kernel=np.zeros((Nframe,Nboot))
    for ishuffle in range(Nboot):
        np.random.shuffle(d)
        if ishuffle%100==0:
            print(ishuffle)
        for iframe in range(Nframe):
            fpr,tpr,_=roc_curve(d,stim[:,iframe])
            aux_kernel[iframe][ishuffle] = auc(fpr, tpr)


    for iframe in range(Nframe):
        kernel['kernel'][iframe]=np.mean(aux_kernel[iframe])
        kernel['error'][iframe]=np.std(aux_kernel[iframe])

    return kernel




def PK_slope(kernel):
    '''
    Compute the slope of the PK:
    PRR=integral( kernel*f(t))
    with f(t)=1-a*t with a such as f(T)=-1 T stimulus duration
    positive recency
    zero flat
    negative primacy
    '''
    aux=np.linspace(1,-1,len(kernel))
    kernel=kernel-0.5
    aux_kernel=(kernel)/(np.sum(kernel))
    return -np.sum(aux_kernel*aux)

def total_area_kernel_PInormalize(kernel):
    '''
    Compute the PK area normalized by the area of a PI
    '''
    nframes=len(kernel)
    area_pi=nframes*( 0.5+2/np.pi*np.arctan(1/np.sqrt(2*nframes-1)) ) -0.5*nframes

    return np.sum(kernel-0.5)/area_pi


def mean_and_error(vector,z=1.0):
    '''
    mean and error according to binomial distribution
    '''
    m=np.mean(vector)
    return m,z*np.sqrt(m*(1-m)/len(vector))



def make_control_stim(mu,sigma,T,N):
    '''
    it returns stimulus with exact mean mu
    and std sigma
    '''
    if N==1:
        stim=np.random.randn(T)
        m=np.mean(stim)
        s=np.std(stim)
        stim=mu+((stim-m)/s)*sigma
    else:
        stim=np.random.randn(N,T)
        for itrial in range(N):
            m=np.mean(stim[itrial])
            s=np.std(stim[itrial])
            stim[itrial]=mu+((stim[itrial]-m)/s)*sigma
    return stim


def running_average(x,window,mode='valid'):
    y=np.convolve(x, np.ones((window))/window, mode=mode)
    return y




def spatial_kernel_sklearn(stim,d,bins=None,Nbins=21,Nboot=500):
    """
    Compute the spatial kernel.
    stim is a 2d array with the stimulus
    d is a 1d array with the choices associated with
    stimulus.
    """


    if not all(np.unique(d)==[0,1]):
        d=(d+1)/2

    if bins is None:
        max_stim=np.max(stim)
        min_stim=np.min(stim)
        b=np.max([max_stim,abs(min_stim)])
        bins=np.linspace(-b,b,Nbins+1)
        print("bins:",bins)

    bin_center=[ (bins[i]+bins[i+1])/2.0 for i in range(len(bins)-1) ]
    kernel_boots=np.zeros((Nboot,len(bin_center)))

    C=10000
    clf_l2_LR = LogisticRegression(C=C, penalty='l2')
    spatial_stim=np.array([np.histogram(stim[i],bins)[0] for i in range(len(stim))])
    #print "hola"
    for iN in range(Nboot):
         index_stims=np.random.randint(0,len(stim),Nboot)

         clf_l2_LR.fit(spatial_stim[index_stims], d[index_stims])

         kernel_boots[iN]=clf_l2_LR.coef_[0]

    kernel_mean=np.zeros(len(kernel_boots[0]))
    kernel_err=np.zeros(len(kernel_boots[0]))

    for it in range(len(kernel_mean)):
        kernel_mean[it]=np.mean(kernel_boots[:,it])
        kernel_err[it]=np.std(kernel_boots[:,it])

    return kernel_mean,kernel_err,bin_center,bins



def basic_psychophysics_results(data):


    Nframes=len(data['stim'][0])
    stim=np.array(data['stim'])
    stim=np.reshape(stim,(len(stim),len(stim[0])))
    stim_sign=np.array(data['stim_sign'])
    isigmas=np.array(data['i_sigmas'])
    choice=np.array(data['choice'])
    #correct=np.array(data['correct'])
    correct=( (stim_sign*choice)+1)/2


    mu=np.array(data['mu'])
    isigmas_values=np.unique(isigmas)
    Nsigmas=len(isigmas_values)

    kernel=np.zeros((Nsigmas,Nframes))
    kernel_err=np.zeros((Nsigmas,Nframes))

    kernel_shuff=np.zeros((Nsigmas,Nframes))
    kernel_shuff_err=np.zeros((Nsigmas,Nframes))


    Performance=np.zeros(Nsigmas)
    Performance_err=np.zeros(Nsigmas)


    PerformanceR=np.zeros(Nsigmas)
    PerformanceR_err=np.zeros(Nsigmas)


    PerformanceL=np.zeros(Nsigmas)
    PerformanceL_err=np.zeros(Nsigmas)

    PR=np.zeros(Nsigmas)
    PR_err=np.zeros(Nsigmas)

    PR_trial=np.zeros(Nsigmas)



    Nbins_spatial_kernel=11
    spatial_kernel=np.zeros((Nsigmas,Nbins_spatial_kernel))
    spatial_kernel_std=np.zeros((Nsigmas,Nbins_spatial_kernel))
    bin_centers=np.zeros((Nsigmas,Nbins_spatial_kernel))


    spatial_kernel_all,spatial_kernel_all_std,bin_centers_all,_=spatial_kernel_sklearn(stim,choice,Nbins=Nbins_spatial_kernel)


    # Nbins_spatial_kernel2=22
    # percentile=np.linspace(0,100,Nbins_spatial_kernel)
    # bins=np.percentile(np.hstack(stim),percentile)
    # spatial_kernel_all2,spatial_kernel_all_std2,bin_centers_all2,_=spatial_kernel_sklearn(stim,choice,bins=bins,Nbins=Nbins_spatial_kernel2)


    PR_total,PR_total_err=mean_and_error( (choice+1)/2 )
    PR_trial_total,PR_trial_total_err=mean_and_error( (stim_sign+1)/2 )


    Nboot=1000
    for isigma in isigmas_values:
        print("isigma: ",isigma)
        indices=np.where(isigmas==isigma)[0]
        Performance[isigma],Performance_err[isigma]=mean_and_error(correct[indices])
        indicesL=np.where( (isigmas==isigma) & (stim_sign==-1) )[0]
        indicesR=np.where( (isigmas==isigma) & (stim_sign==1) )[0]


        PerformanceR[isigma],PerformanceR_err[isigma]=mean_and_error(correct[indicesR])
        PerformanceL[isigma],PerformanceL_err[isigma]=mean_and_error(correct[indicesL])


        d=choice[indices]*stim_sign[indices]



        PR[isigma],PR_err[isigma]=mean_and_error( (choice[indices]+1)/2 )
        PR_trial[isigma]=np.mean( (stim_sign[indices]+1)/2 )



        stim_sigma=np.array([stim[i]*stim_sign[i]-mu[i] for i in indices ]) ### correct vs incorrecte
        d=choice[indices]*stim_sign[indices]   ### correct vs incorrecte
        aux_k=kernel_error(stim_sigma,d,error=True,Nboot=Nboot)
        kernel[isigma]=aux_k['kernel']
        kernel_err[isigma]=aux_k['error']


        aux_k=kernel_shuffle(stim_sigma,d,Nboot=Nboot)
        kernel_shuff[isigma]=aux_k['kernel']
        kernel_shuff_err[isigma]=aux_k['error']

    results={
    "kernel": kernel,"kernel_std": kernel_err,"kernel_shuffle": kernel_shuff,
    "kernel_shuffle_std": kernel_shuff_err,"Performance":Performance,"Performance_std":Performance_err,
    "PerformanceR":PerformanceR,"PerformanceR_std":PerformanceR_err,"PerformanceL":PerformanceL,
    "PerformanceL_std":PerformanceL_err,"PR":PR,"PR_std":PR_err,"PR_trial":PR_trial,
    "spatial_kernel_all":spatial_kernel_all,"spatial_kernel_all_std":spatial_kernel_all_std,
    "bin_centers_all":bin_centers_all}


    return results





def AIC(LL,Nparam,model_base):
	Delta_AIC={}
	for model in LL.keys():
		Delta_AIC[model]=[]
		for isubject in range(len(LL[model_base])):
			AIC_model=2*Nparam[model]+2*LL[model][isubject]
			AIC_model_base=2*Nparam[model_base]+2*LL[model_base][isubject]
			Delta_AIC[model].append(AIC_model-AIC_model_base)
	return Delta_AIC




def best_model(LL,Nparam):
    keys=list(LL.keys())
    Nsubject=len(LL[keys[0]])
    AIC=[ {} for  isubject in range(Nsubject) ]


    best_model=[]

    for isubject in range(Nsubject):
        for model in LL.keys():
            AIC[isubject][model]=2*Nparam[model]+2*LL[model][isubject]
        best=min(AIC[isubject], key=AIC[isubject].get)
        best_model.append(best)
    return best_model,AIC



def create_stim(mu,sigmas,N,Nframes):
    m=mu*np.random.choice([-1,1],N)
    stim=np.random.randn(N,Nframes)
    s=int((N/len(sigmas)) )*list(sigmas)
    s=np.array(s)
    for i in range(N):
        stim[i]=m[i]+s[i]*(stim[i] -np.mean(stim[i]) )/np.std(stim[i])


    return stim,m,s



def basic_psychophysics_results2(data):


    Nframes=len(data['stim'][0])
    stim=np.array(data['stim'])
    stim=np.reshape(stim,(len(stim),len(stim[0])))
    Nstim,Nframes=np.shape(stim)

    #stim_sign=np.array(data['stim_sign'])
    i_sigmas=np.array(data['i_sigmas'],dtype=int)
    choices=np.array(data['choices'])
    stim_sign=np.array(data["stim_sign"])
    correct=np.array( (choices*stim_sign+1)/2)
    print("correct",correct)
    ## temporal kernel ##
    logit,tpk_stderr=temporal_kernel_sm(stim,choices)
    tpk=logit.params

    pkslope,pkslope_std_err=temporal_pk_slope(tpk)


    ## spatial kernel ##
    logit,spk_stderr,bins_centers=spatial_kernel_sm(stim,choices,Nbins=11)
    spk=logit.params


    i_sigmas_values=np.unique(i_sigmas)
    Nsigmas=len(i_sigmas_values)


    Performance=np.zeros(Nsigmas)
    Performance_stderr=np.zeros(Nsigmas)
    PerformanceR=np.zeros(Nsigmas)
    PerformanceR_stderr=np.zeros(Nsigmas)
    PerformanceL=np.zeros(Nsigmas)
    PerformanceL_stderr=np.zeros(Nsigmas)

    for isigma in i_sigmas_values:
        print("isigma: ",isigma)
        indices=np.where(i_sigmas==isigma)[0]

        Performance[isigma],Performance_stderr[isigma]=mean_and_error(correct[indices])
        indicesL=np.where( (i_sigmas==isigma) & (stim_sign==-1) )[0]
        indicesR=np.where( (i_sigmas==isigma) & (stim_sign==1) )[0]
        PerformanceR[isigma],PerformanceR_stderr[isigma]=mean_and_error(correct[indicesR])
        PerformanceL[isigma],PerformanceL_stderr[isigma]=mean_and_error(correct[indicesL])

    results={
    "tpk":tpk,"tpk_stderr":tpk_stderr,"pkslope":pkslope,
    "pkslope_std_err":pkslope_std_err,"spk":spk,
    "spk_stderr":spk_stderr,"bins_centers":bins_centers,
    "Performance":Performance,"Performance_stderr":Performance_stderr,
    "PerformanceR":PerformanceR,"PerformanceR_stderr":PerformanceR_stderr,
    "PerformanceL":PerformanceL,"PerformanceL_stderr":PerformanceL_stderr
    }

    return results

def temporal_kernel_sm_correct_vs_error(stim,choices,stim_sign,stim_full=False):

    #compute mean of each stimulus
    Nstim,Nframes=np.shape(stim)
    mu=np.array([np.mean(stim[i]) for i in range(Nstim) ])
    mu=np.reshape(mu,(Nstim,1))
    #stim_sign=np.sign(mu[:,0])
    # if not all(np.unique(choices)==[0,1]):
    #     choices=(choices+1)/2
    correct=stim_sign*choices
    correct=(correct+1)/2
    m=np.abs(np.mean(stim[0]))
    if not stim_full:
        stim_fluctuation=np.array([ stim_sign[i]*(stim[i]) -m for i in range(Nstim)])
        #stim_fit=stim_fluctuation
        #stim_fit=np.append(stim_fluctuation,np.abs(mu), axis=1)
        #stim_fit=np.append(stim_fit,np.ones((Nstim,1)), axis=1) #add a bias
        stim_fit=np.append(stim_fluctuation,m*np.ones((Nstim,1)), axis=1) #add a bias

    else:
        stim_fit=sm.add_constant(stim)
    print(np.shape(stim_fit))
    print(np.shape(correct))
    log_reg=sm.Logit(correct,stim_fit)
    fit_res=log_reg.fit()
    #Laplace approximation for conf_interval
    hessian=log_reg.hessian(fit_res.params)
    hinv=np.linalg.inv(hessian)
    diag=np.diagonal(hinv)
    stderr=np.sqrt(-diag)

    return fit_res,stderr


def temporal_kernel_sm(stim,choices):

    Nstim,Nframes=np.shape(stim)
    #print(Nstim)

    if not all(np.unique(choices)==[0,1]):
        choices=(choices+1)/2

    stim_fit=sm.add_constant(stim)
    print(np.shape(stim_fit))
    log_reg=sm.Logit(choices,stim_fit)
    fit_res=log_reg.fit()
    #Laplace approximation for conf_interval
    hessian=log_reg.hessian(fit_res.params)
    hinv=np.linalg.inv(hessian)
    diag=np.diagonal(hinv)
    stderr=np.sqrt(-diag)

    return fit_res,stderr

def temporal_pk_slope(tpk):
    x=range(len(tpk))
    x=sm.add_constant(x)
    results=sm.OLS(tpk,x).fit()
    std_err=(results.conf_int()[1][1]-results.conf_int()[1][0])/1.96
    return results.params[1],std_err

def spatial_kernel_sm(stim,choices,bins=None,Nbins=21):
    """
    Compute the spatial kernel.
    stim is a 2d array with the stimulus
    d is a 1d array with the choices associated with
    stimulus.
    """

    Nstim,Nframes=np.shape(stim)

    if not all(np.unique(choices)==[0,1]):
        choices=(choices+1)/2

    if bins is None:
        max_stim=np.max(stim)
        min_stim=np.min(stim)
        b=np.max([max_stim,abs(min_stim)])
        bins=np.linspace(-b,b,Nbins+1)
        print("bins:",bins)

    bin_center=[ (bins[i]+bins[i+1])/2.0 for i in range(len(bins)-1) ]

    spatial_stim=np.array([np.histogram(stim[i],bins)[0] for i in range(len(stim))])

    #stim_fit=np.append(spatial_stim,np.ones((Nstim,1)), axis=1) #add a bias
    stim_fit=spatial_stim

    log_reg=sm.Logit(choices,stim_fit)
    fit_res=log_reg.fit()
    #Laplace approximation for conf_interval
    hessian=log_reg.hessian(fit_res.params)
    hinv=np.linalg.inv(hessian)
    diag=np.diagonal(hinv)
    stderr=np.sqrt(-diag)

    return fit_res,stderr,bin_center
