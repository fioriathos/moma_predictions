import numpy as np
from numba import jit

@jit(nopython=True)
def parameters(gamma,dt,mlam,sig_l_s):
    """return F,A, a as in Erik theory paper """
    A = np.empty((2,2))
    A[0,0] =  sig_l_s/(2*gamma**3)*\
            (2*gamma*dt-3+4*np.exp(-gamma*dt)-np.exp(-2*gamma*dt))
    A[1,1] = sig_l_s/(2*gamma)*(1-np.exp(-2*gamma*dt))
    A[0,1] = sig_l_s/(2*gamma**2)*(1-np.exp(-gamma*dt))**2
    A[1,0] = A[0,1]

    F = np.empty((2,2))
    F[0,0] = 1
    F[0,1] = (1-np.exp(-gamma*dt))/gamma
    F[1,0] = 0
    F[1,1] = np.exp(-gamma*dt)

    a = np.empty((2,1))
    a[0,0] = mlam/gamma*(np.exp(-gamma*dt)-(1-gamma*dt))
    a[1,0] = mlam*(1-np.exp(-gamma*dt))

    return F, A, a
@jit(nopython=True)
def new_mean_cov(mu, M,F,A,a):
    """From old mean mu and cov M return new one"""
    # No numpy modules for speed up in numba
    #mu_ = a + np.matmul(F,mu)
    #M_ = A+np.matmul(F,np.matmul(M,F.T))
    m = a + np.array([[F[0,0]*mu[0,0]+F[0,1]*mu[1,0]],\
                        [F[1,0]*mu[0,0]+F[1,1]*mu[1,0]]])
    T =\
    np.array([[M[0,0]*F[0,0]+M[0,1]*F[0,1],M[0,0]*F[1,0]+M[0,1]*F[1,1]],\
              [M[1,0]*F[0,0]+M[1,1]*F[0,1],M[1,0]*F[1,0]+M[1,1]*F[1,1 ]]])
    T_ = np.array([[F[0,0]*T[0,0]+F[0,1]*T[1,0],F[0,0]*T[0,1]+F[0,1]*T[1,1]],\
                  [F[1,0]*T[0,0]+F[1,1]*T[1,0],F[1,0]*T[0,1]+F[1,1]*T[1,1]]])
    Q = A + T_
    return m, Q
@jit(nopython=True)
def posteriori_matrices(x,m,Q,sig_m_s):
    """From updated mu, M return mean and cov of posterior (wich is normal dist)"""
    den = sig_m_s+Q[0,0]
    mu_ = np.zeros_like(m);
    M_ = np.zeros_like(Q)
    mu_[1,0] = m[1,0]+Q[0,1]/den*(x-m[0,0]) #mean lambda t+dt
    mu_[0,0] = (sig_m_s*m[0,0]+Q[0,0]*x)/den #mean x t+dt
    M_[0,0] = sig_m_s*Q[0,0]/den    #var x t+dt
    M_[1,1] = Q[1,1] - Q[0,1]*Q[0,1]/den # var lam t+dt
    M_[0,1] = Q[0,1]*sig_m_s/den
    M_[1,0] = M_[0,1]
    return mu_, M_
@jit(nopython=True)
def log_likelihood(x,mu,M,sig_m_s):
    """Return P(x_{t+dt}|D_t) in log """
    den = sig_m_s+M[0,0]
    return -(x-mu[0,0])**2/(2*den)-0.5*(np.log(den)+np.log(2*np.pi))
@jit(nopython=True)
def objective_1cc(W,mlam,gamma,sig_l_s,sig_m_s,dt,m0,M0):
    """Give -log_lik after one cell cycle and the next inital matrix"""
    assert W.shape[0] == 1
    F, A, a = parameters(gamma,dt,mlam,sig_l_s)
    # log likelihood at time zero
    ll=-(W[0,0]-m0[0,0])**2/(2*sig_m_s)-0.5*np.log(sig_m_s)-np.log(2*np.pi)
    m,Q = new_mean_cov(m0, M0,F,A,a)
    for j in range(W.shape[1]):
        ll += log_likelihood(W[0,j],m,Q,sig_m_s)
        mu, M = posteriori_matrices(W[0,j],m,Q,sig_m_s)
        m,Q = new_mean_cov(mu, M,F,A,a)
    return -ll, np.array([[m[0,0]/2.,m[1,0]]]),np.array([[Q[0,0]/4.,Q[0,1]/2],[Q[0,1]/2,Q[1,1]]])
@jit(nopython=True)
def predictions_1cc(W,mlam,gamma,sig_l_s,sig_m_s,dt,m0,M0):
    """Return the optimal length and growth and their errors and next matrix"""
    assert W.shape[0] == 1
    F, A, a = parameters(gamma,dt,mlam,sig_l_s)
    z = []; err_z = []
    m,Q = new_mean_cov(m0, M0,F,A,a)
    for j in range(W.shape[1]):
        mu, M = posteriori_matrices(W[0,j],m,Q,sig_m_s)
        z.append(np.array(mu)); err_z.append(np.sqrt(np.array([M[0,0],M[1,1]])))
        m,Q = new_mean_cov(mu, M,F,A,a)
    return np.hstack(z), np.vstack(err_z).T,np.array([[m[0,0]/2.,m[1,0]]]),np.array([[Q[0,0]/4.,Q[0,1]/2],[Q[0,1]/2,Q[1,1]]])
################################################################################################
############################### TO BE CANCELLED  ###############################################
################################################################################################
def ornstein_uhlenbeck(mlam,gamma,sig,length=30,ncel=10,dt=3.):
    mat = np.zeros((ncel,length))
    dW = np.random.normal(loc=mat,scale=np.sqrt(dt))
    mat[:,0]=mlam+sig*dW[:,0]
    for k in range(1,length):
        mat[:,k]=mat[:,k-1]-gamma*(mat[:,k-1]-mlam)*dt+sig*dW[:,k-1]
    return mat
def integrated_ou(mlam,gamma,sig,sigm2,X0=1,sx0=0.1,length=30,ncel=10,dt=3.):
    X = ornstein_uhlenbeck(mlam,gamma,sig,length,ncel,dt)
    X0 = np.random.normal(loc=np.ones((ncel,1)),scale=sx0)
    return np.random.normal(loc=np.hstack([X0,np.cumsum(X,axis=1)*dt+X0]),scale=np.sqrt(sigm2))[:,:-1], X
def W_er(st): 
    diffW= abs(W-z[0,:])
    percW = sum(sum(diffW>st*err_z[0,:]))/diffW.shape[1]
    return percW
def X_er(st):
    diffX= abs(X-z[1,:])
    percX = sum(sum(diffX>st*err_z[1,:]))/diffX.shape[1]
    return percX
if  __name__=="__main__":
    import time
    np.random.seed(1)
    W,X = integrated_ou(mlam=1.,gamma=0.02,sig=.03,sigm2=0.08,length=100000,ncel=1,dt=3.)
    m0 = np.array([[np.mean(W[:,0])],[np.mean(X[:,0])]])
    M0 = np.array([[np.var(W[:,0]),0],[0,np.var(X[:,0])]])
    t1 = time.time()
    print(objective_1cc(W,mlam=1.,gamma=0.02,sig_l_s=.03**2,sig_m_s=0.08,dt=3.,m0=m0,M0=M0))
    print(time.time()-t1)
    #z, err_z,_,_ = predictions_1cc(W,mlam=1.,gamma=0.02,sig_l_s=.03**4,sig_m_s=0.08,dt=3.,m0=m0,M0=M0)
    #print(W_er(1),W_er(2))
    #print(X_er(1),X_er(2))
