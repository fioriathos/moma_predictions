import sys
sys.path.append('/scicore/home/nimwegen/fiori/anaconda2/lib/python2.7/site-packages/')
import numpy as np
class ou_process(object):
    """dx = -gamma(x-mu)+sigma dW"""
    def __init__(self, mu, gamma, sigma):
        """Initialize a class instance"""
        self.mu = mu
        self.gamma = gamma
        self.sigma = sigma
    def _mu(self, x):
        return -self.gamma*(x-self.mu)
    def _dW(self, delta_t,leng=1):
        """Return normal rnd with mean 0 and variance delta_t"""
        return np.random.normal(loc = np.zeros(leng), scale = np.sqrt(delta_t))
    def _predict(self, t_fin,g_std, l_0 = None, t_0 = 0, dt = 3.,seed=None):
        """Predict the production using 3 min delay and starting at 0"""
        t_fin+=1000*dt#ensure to be in stationary phase
        if seed is not None: np.random.seed(seed)
        l_0 = l_0 if l_0 is not None else self.mu #initialize at the mean
        t = np.arange(t_0, t_fin+dt ,dt)
        dw = self._dW(dt,len(t))
        for i,j in enumerate(t):
            if i == 0:
                x = np.array([l_0])
            else:
                x = np.append(x, x[i-1] + self._mu(x[i-1])*dt +\
                              self.sigma*dw[i-1])
        t=t[1000:]-t[1000]; x=x[1000:]#cancel the added 100 points
        xnoise = np.random.normal(loc=x ,scale = g_std*np.ones_like(x))
        return t, x, xnoise
class integrated_ou_process(ou_process):#inherit
    """Math:
        dx = -gamma(x-mu)+sigma dW
        dy = dx (no bleaching)
        dy = dx - lambda y (bleaching)
        """
    def __init__(self, mu, gamma, sigma):
        super(integrated_ou_process, self).__init__(mu, gamma, sigma) # inherit
    def _predict(self,  l_0, t_fin, g_std, x_0=None, t_0 = 0, dt=\
                              3.,seed=None):
        """dL/dt = x take the EXP and add G_noise"""
        x_0 = x_0 if x_0 is not None else self.mu #initialize at the mean
        t,x,_ = super(integrated_ou_process,self)._predict(l_0=x_0,g_std=0., t_fin = t_fin,t_0 = t_0, dt=\
                                 dt,seed=seed)
        for i,j in enumerate(x):
            if i == 0:
                y = np.array([l_0])
            else:
                y = np.append(y, y[i-1]+x[i-1]*dt)
        if seed is not None: np.random.seed(seed)
        y = np.random.normal(loc=y ,scale = g_std*np.ones_like(y))
        return t, x, y
class integrated_ou_with_bleach(ou_process):
    """dg/dt = lambda_1 x_t-lambda_2 g_t, length considered perfect!"""
    def __init__(self,mu,gamma,sigma,lamb1,lamb2,length):
        super(integrated_ou_with_bleach, self).__init__(mu, gamma, sigma) # inherit
        self.lamb1=lamb1
        self.lamb2=lamb2
        assert type(lamb1)==type(lambd2)==type(length)==np.ndarray
    def _predict(self, length, l_0, t_fin, g_std,\
                 x_0=None, t_0 = 0, dt = 3.,seed = None):
        x_0 = x_0 if x_0 is not None else self.mu #initialize at the mean
        assert length[0] is not 0, "give reasonalbe measurments!"
        # The production is an OU
        t,x,_ = super(integrated_ou_with_bleach,self)._predict(l_0=x_0,g_std=0, t_fin = t_fin,t_0 = t_0,\
                                 dt=dt,seed=seed)
        # integrat
        for i,j in enumerate(x):
            if i == 0:
                y = np.array([l_0])
            else:
                y = np.append(y,y[i-1]+x[i-1]*self.lamb1[i-1]\
                              *dt-self.lamb2[i-1]*y[i-1]*dt)
        if seed is not None: np.random.seed(seed)
        if g_std is not None:
            y = np.random.normal(loc=y ,scale = g_std*np.ones_like(y))
        else:
            y = np.random.normal(loc=y ,scale = 1.1*np.sqrt(y/self.length))
        return t, x, y

