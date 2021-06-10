import numpy as np




def potential_TW(x,coef):
	return -coef[0]*x-0.50*coef[1]*x**2+0.250*coef[2]*x**4+(1./6.)*coef[3]*x**6


def st_logistic_bias(stim,st_p,model=6):
	Nframes=len(stim)
	y=np.zeros(Nframes)
	for iframe in range(Nframes):
		if stim[iframe]> 0:
			y[iframe]=2*(1./st_p[0])*(1./(1+np.exp(-((1./st_p[1])*(stim[iframe] ) ) ) ) -0.5 )
		else:
		#s_trans[iframe]=2*st_p[1]*(1./(1+exp(-(st_p[2]*(stim[iframe]-st_p[3] ) ) ) ) -0.5 )
			y[iframe]=2*((1./st_p[0])-st_p[2])*(1./(1+np.exp(-( ((1./st_p[1])-st_p[3])*(stim[iframe] ) ) ) ) -0.5 )
	return y


def st_logistic(stim,st_p):
	Nframes=len(stim)
	y=np.zeros(Nframes)
	for iframe in range(Nframes):
		y[iframe]=2*(1./st_p[0])*(1./(1+np.exp(-((1./st_p[1])*(stim[iframe] ) ) ) ) -0.5 )
	return y


def st_linear(stim,st_p):
	Nframes=len(stim)
	y=np.zeros(Nframes)
	for iframe in range(Nframes):
		y[iframe]=st_p[0]*stim[iframe]
	return y


def st_linear_time(x,st_p,iframe):
	y=(st_p[0]+st_p[1]*iframe)*x
	return y


def st_logistic_time(stim,st_p):
	Nframes=len(stim)
	y=np.zeros(Nframes)
	for iframe in range(Nframes):
		y[iframe]=2*(1./st_p[0])*(1./(1+ np.exp(-((1./(st_p[1]+(iframe-1)*st_p[2])*(stim[iframe] ) ) ) ) -0.5 ) )
	return y

def st_logistic_time_plot(x,st_p,iframe):
	y=2*(1./st_p[0])*(1./(1+ np.exp(-x/(st_p[1]+(iframe)*st_p[2])) ) -0.5  )
	return y


def st_logistic_plot(x,st_p):
	y=2*(1./st_p[0])*(1./(1+ np.exp(-x/st_p[1]) ) -0.5  )
	return y
