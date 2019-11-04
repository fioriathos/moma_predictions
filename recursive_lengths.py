import numpy as np
from numba import jit
#### Derivatives works ;)
#@jit(nopython=True)
def parameters(gamma,dt,mlam,sig_l_s):
    """return F,A, a as in Erik theory paper OK """
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
#@jit(nopython=False)
def grad_parameters(gamma,dt,mlam,sl2):
    """The non zero derivatives OK"""
    ############################################
    # For F
    ############################################
    emg = np.exp(-gamma*dt)
    tmp = dt*emg/gamma-(1-emg)/gamma**2
    F_gamma = np.array([[0,tmp],[0,-dt*np.exp(-gamma*dt)]])
    ############################################
    # For a
    ############################################
    a_mlam = np.empty((2,1))
    a_mlam[0,0] = 1/gamma*(np.exp(-gamma*dt)-(1-gamma*dt))
    a_mlam[1,0] = (1-np.exp(-gamma*dt))
    #-------------
    a_gamma = np.empty((2,1))
    a_gamma[0,0] = mlam*((-dt*emg+dt)/gamma-(emg-(1-gamma*dt))/gamma**2)
    a_gamma[1,0] = mlam*(dt*np.exp(-gamma*dt))
    ############################################
    #For A
    ############################################
    A_sl2 = np.empty((2,2))
    A_sl2[0,0] =  1/(2*gamma**3)*\
            (2*gamma*dt-3+4*np.exp(-gamma*dt)-np.exp(-2*gamma*dt))
    A_sl2[1,1] = 1/(2*gamma)*(1-np.exp(-2*gamma*dt))
    A_sl2[0,1] = 1/(2*gamma**2)*(1-np.exp(-gamma*dt))**2
    A_sl2[1,0] = A_sl2[0,1]
    #--------------
    A_gamma = np.zeros((2,2))
    A_gamma[0,0] = sl2/(2*gamma**3)*((2*dt-4*dt*emg+2*dt*emg**2)-3/gamma*(2*gamma*dt-3+4*emg-emg**2))
    A_gamma[1,1] = sl2/2*(2*dt*emg**2/gamma-1/gamma**2*(1-emg**2))
    A_gamma[0,1] = sl2/2*(2*(1-emg)/gamma**2*dt*emg-2/gamma**3*(1-emg)**2)
    A_gamma[1,0] = A_gamma[0,1]
    return {'F_gamma':F_gamma,'a_mlam':a_mlam,'a_gamma':a_gamma,'A_sl2':A_sl2,'A_gamma':A_gamma}
#@jit(nopython=True)
def new_mean_cov(b, B,F,A,a):
    """From old mean mu and cov M return new one"""
    # No numpy modules for speed up in numba
    m = a + np.dot(F,b)
    Q = A+np.dot(F,np.dot(B,F.T))
    return m, Q
#@jit(nopython=False)
def grad_new_mean_cov(b,B,F,A,a,grad_para,grad_mat_b):
    """Grad m and Q TO CHECK"""
    F_gamma=grad_para['F_gamma']
    #######################################################
    # Grad m
    ######################################################
    fdgm = lambda x: np.dot(F,grad_mat_b['{}'.format(x)])
    m_mlam = grad_para['a_mlam']+fdgm('b_mlam')
    m_gamma = grad_para['a_gamma'] + fdgm('b_gamma')+np.dot(F_gamma,b)
    m_sl2 = fdgm('b_sl2')
    m_sm2 = fdgm('b_sm2')
    #######################################################
    # Grad Q grad_mat['
    ######################################################
    fdgf = lambda x: np.dot(F,np.dot(grad_mat_b['{}'.format(x)],F.T))
    Q_mlam = fdgf('B_mlam')
    Q_gamma = grad_para['A_gamma'] + np.dot(F_gamma,np.dot(B,F.T))\
              + fdgf('B_gamma')\
              + np.dot(F,np.dot(B,F_gamma.T))
    Q_sl2 = grad_para['A_sl2'] + fdgf('B_sl2')
    Q_sm2 = fdgf('B_sm2')
    return {'m_mlam':m_mlam,'m_gamma':m_gamma,'m_sl2':m_sl2,'m_sm2':m_sm2,\
            'Q_mlam':Q_mlam,'Q_gamma':Q_gamma,'Q_sl2':Q_sl2,'Q_sm2':Q_sm2}
#@jit(nopython=True)
def posteriori_matrices(x,m,Q,sig_m_s):
    """From updated m, Q return mean and cov of posterior b',B' in my paper (wich is normal dist)"""
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
#@jit(nopython=False)
def grad_posteriori_matrices(x,m,Q,sm2,grad_mat):
    """Grad b and B TO CHECK"""
    den = sm2+Q[0,0]
    #########################################################
    # GRAD B
    ###########################################################
    def dB(Q_d,ds=0):
        """Return the gradient out than for sm2"""
        B_d = np.zeros((2,2))
        dQ = grad_mat['{}'.format(Q_d)]
        B_d[0,0] = sm2*dQ[0,0]/den-sm2*Q[0,0]/den**2*(dQ[0,0]+ds)
        B_d[1,1] = dQ[1,1]-2*Q[0,1]*dQ[0,1]/den+Q[0,1]**2/den**2*(dQ[0,0]+ds)
        B_d[0,1] = sm2*dQ[0,1]/den-sm2*Q[0,1]/den**2*(dQ[0,0]+ds)
        B_d[1,0] = B_d[0,1]
        return B_d
    #-------------------------------
    B_sm2 = dB('Q_sm2',1)
    B_sm2 = B_sm2 + np.array([[Q[0,0],Q[0,1]],[Q[0,1],0]])/den
   #################################################################
    # Grad b
    ################################################################
    def db(m_d,Q_d,sd=0):
        """Returnthe gradient out of sm2"""
        b_d = np.zeros((2,1))
        dm = grad_mat['{}'.format(m_d)]
        dQ = grad_mat['{}'.format(Q_d)]
        b_d[0,0] = dm[0,0]*sm2/den + x*dQ[0,0]/den\
                   -(m[0,0]*sm2+Q[0,0]*x)/den**2*(dQ[0,0]+sd)
        b_d[1,0] = dm[1,0]-(dQ[0,1]*(m[0,0]-x)+Q[0,1]*dm[0,0])/den\
                   +Q[0,1]*(m[0,0]-x)/den**2*(dQ[0,0]+sd)
        return b_d
    b_sm2 = db('m_sm2','Q_sm2',1) + np.array([[m[0,0]/den],[0]])
    return {'B_gamma':dB('Q_gamma'),'B_mlam':dB('Q_mlam'),'B_sl2':dB('Q_sl2'),\
            'B_sm2':B_sm2,'b_sm2':b_sm2,\
            'b_gamma':db('m_gamma','Q_gamma'),'b_mlam':db('m_mlam','Q_mlam'),'b_sl2':db('m_sl2','Q_sl2')}
#@jit(nopython=False)
def grad_post_initial():
    """They do not depend on the parameters so all zeros"""
    mat = np.zeros((2,2)); vec = np.zeros((2,1))
    return {'B_gamma':mat,'B_mlam':mat,'B_sl2':mat,\
            'B_sm2':mat,'b_sm2':vec,\
            'b_gamma':vec,'b_mlam':vec,'b_sl2':vec}
#@jit(nopython=True)
def log_likelihood(x,m,Q,sm2):
    """Return P(x_{t+dt}|D_t) in log """
    den = sm2+Q[0,0]
    return -(x-m[0,0])**2/(2*den)-0.5*(np.log(den)+np.log(2*np.pi))
#@jit(nopython=False)
def grad_log_likelihood(x,m,Q,sm2,grad_mat):
    """Give gradient of log likelihood TO CHECK"""
    den = sm2 + Q[0,0]
    def grad(m_d,Q_d,sd=0):
        dm0 = grad_mat['{}'.format(m_d)][0,0]
        dQ0 = grad_mat['{}'.format(Q_d)][0,0]
        return   (x-m[0,0])*dm0/den\
                +(x-m[0,0])**2/(2*den**2)*(dQ0+sd)\
                -0.5/den*(dQ0+sd)
    return np.array([[grad('m_mlam','Q_mlam')],\
                    [grad('m_gamma','Q_gamma')],\
                    [grad('m_sl2','Q_sl2')],\
                    [grad('m_sm2','Q_sm2',1)]])
#@jit(nopython=True)
def objective_1cc(W,mlam,gamma,sl2,sm2,dt,m0,M0):
    """Give -log_lik after one cell cycle and the next inital matrix"""
    assert W.shape[0] == 1
    F, A, a = parameters(gamma,dt,mlam,sl2)
    # log likelihood at time zero
    ll=-(W[0,0]-m0[0,0])**2/(2*sm2)-0.5*np.log(sm2)-np.log(2*np.pi)
    m,Q = new_mean_cov(m0, M0,F,A,a)
    for j in range(1,W.shape[1]):
        ll += log_likelihood(W[0,j],m,Q,sm2)
        mu, M = posteriori_matrices(W[0,j],m,Q,sm2)
        m,Q = new_mean_cov(mu, M,F,A,a)
    return -ll, np.array([[m[0,0]/2.,m[1,0]]]),np.array([[Q[0,0]/4.,Q[0,1]/2],[Q[0,1]/2,Q[1,1]]])
#@jit(nopython=False)
def obj_and_grad_1cc(W,mlam,gamma,sl2,sm2,dt,m0,M0,grad_mat0 = grad_post_initial()):
    """To check"""
    F, A, a = parameters(gamma,dt,mlam,sl2)
    grad_param = grad_parameters(gamma,dt,mlam,sl2)
    ll=-(W[0,0]-m0[0,0])**2/(2*sm2)-0.5*np.log(sm2)-np.log(2*np.pi)
    gll = np.zeros((4,1))
    gll[3,0] = +(W[0,0]-m0[0,0])**2/(2*sm2**2)-0.5/(sm2)
    m,Q = new_mean_cov(m0, M0,F,A,a)
    grad_mat = grad_new_mean_cov(m0,M0,F,A,a,grad_param,grad_mat0)
    for j in range(1,W.shape[1]):
        ll += log_likelihood(W[0,j],m,Q,sm2)
        gll += grad_log_likelihood(W[0,j],m,Q,sm2,grad_mat)
        mu, M = posteriori_matrices(W[0,j],m,Q,sm2)
        grad_mat_b = grad_posteriori_matrices(W[0,j],m,Q,sm2,grad_mat)
        m,Q = new_mean_cov(mu, M,F,A,a)
        grad_mat = grad_new_mean_cov(mu,M,F,A,a,grad_param,grad_mat_b)
    return -ll, -gll
#@jit(nopython=True)
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
    W,X = integrated_ou(mlam=1.,gamma=0.02,sig=.03,sigm2=0.08,length=30000,ncel=1,dt=3.)
    m0 = np.array([[np.mean(W[:,0])],[np.mean(X[:,0])]])
    M0 = np.array([[np.var(W[:,0]),0],[0,np.var(X[:,0])]])
    t1 = time.time()
    print(obj_and_grad_1cc(W,mlam=1.,gamma=0.02,sl2=.03**2,sm2=0.08,dt=3.,m0=m0,M0=M0))
    print(time.time()-t1)
    #z, err_z,_,_ = predictions_1cc(W,mlam=1.,gamma=0.02,sig_l_s=.03**4,sig_m_s=0.08,dt=3.,m0=m0,M0=M0)
    #print(W_er(1),W_er(2))
    #print(X_er(1),X_er(2))
