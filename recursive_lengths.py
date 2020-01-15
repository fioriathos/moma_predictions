import numpy as np
from numba import jit,prange
from copy import deepcopy
import pandas as pd
import time
from pathos.multiprocessing import ProcessingPool as Pool
########################################################################################
#############################INFERENCE METHODS##########################################
########################################################################################
def parameters(gamma,dt,mlam,sig_l_s):
    """ P(z_{t+dt}|z_t) = N(a+F z_t; A) """
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
#
def new_mean_cov(b, B,F,A,a):
    """Start from P(z_t|D_t)= N(b;B) and find P(z_{t+dt}|D_t) =
    N(a+Fb;A+FBF.T):= N(m,Q)  """
    # No numpy modules for speed up in numba
    m = a + np.dot(F,b)
    Q = A+np.dot(F,np.dot(B,F.T))
    return m, Q
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
#
def posteriori_matrices(x,m,Q,sm2):
    """    P(z_{t+dt}|D_{t+dt})=P(x_{t+dt}^m|z_{t+dt})P(z_{t+dt}|D_t)
    =N(x_{t+dt},sm2)N(m,Q)=N(b',B') """
    den = sm2+Q[0,0]
    b_ = np.zeros_like(m);
    B_ = np.zeros_like(Q)
    b_[1,0] = m[1,0]+Q[0,1]/den*(x-m[0,0]) #mean lambda t+dt
    b_[0,0] = (sm2*m[0,0]+Q[0,0]*x)/den #mean x t+dt
    B_[0,0] = sm2*Q[0,0]/den    #var x t+dt
    B_[1,1] = Q[1,1] - Q[0,1]*Q[0,1]/den # var lam t+dt
    B_[0,1] = Q[0,1]*sm2/den
    B_[1,0] = B_[0,1]
    return b_, B_
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
#
def log_likelihood(x,m,Q,sm2):
    """Return P(x_{t+dt}|D_t) in log """
    den = sm2+Q[0,0]
    # if den gets too small..
    #np.seterr(invalid='raise') # if log negative stop everything
    assert den>0, "wrong state space Q[00]={} and sm2 + {}".format(Q[0,0],sm2)
    if den<1e-08:
        tmp = -(x-m[0,0])**2/(2*den)-0.5*(np.log(1e10*den)-np.log(1e10)+np.log(2*np.pi))
    else:
        tmp =  -(x-m[0,0])**2/(2*den)-0.5*(np.log(den)+np.log(2*np.pi))
    return tmp
def grad_log_likelihood(x,m,Q,sm2,grad_mat):
    """Give gradient of log likelihood """
    den = sm2 + Q[0,0]
    def grad(m_d,Q_d,sd=0):
        dm0 = grad_mat['{}'.format(m_d)][0,0]
        dQ0 = grad_mat['{}'.format(Q_d)][0,0]
        return   (x-m[0,0])*dm0/den\
                +(x-m[0,0])**2/(2*den**2)*(dQ0+sd)\
                -0.5/den*(dQ0+sd)
    return  np.array([[grad('m_mlam','Q_mlam')],\
                    [grad('m_gamma','Q_gamma')],\
                    [grad('m_sl2','Q_sl2')],\
                    [grad('m_sm2','Q_sm2',1)]])
#------------------ OBJECTIVE OVER CELL CYCLE/ LANE AND TOTAL-----------------------------------
def obj_and_grad_1cc(W,mlam,gamma,sl2,sm2,dt,s,S,grad_matS,rescale):
    """To check"""
    ##### likelihood and gradient at initial conditions
    ll = log_likelihood(W[0,0],s,S,sm2)
    gll = grad_log_likelihood(W[0,0],s,S,sm2,grad_matS)
    #### Initialize parameters for recurrence
    F, A, a = parameters(gamma,dt,mlam,sl2)
    grad_param = grad_parameters(gamma,dt,mlam,sl2)
    ##### P(z_0|x_0^m)
    b,B = posteriori_matrices(W[0,0],s,S,sm2)
    grad_mat_b = grad_posteriori_matrices(W[0,0],s,S,sm2,grad_matS)
    for j in range(1,W.shape[1]):
        ###### P(z_{t+dt}|D_t) = N(m,Q))
        m,Q = new_mean_cov(b,B,F,A,a)
        grad_mat_Q = grad_new_mean_cov(b,B,F,A,a,grad_param,grad_mat_b)
        ##### P(z_{t+dt}|D_{t+dt}) = N(b',B')
        b,B = posteriori_matrices(W[0,j],m,Q,sm2)
        grad_mat_b = grad_posteriori_matrices(W[0,j],m,Q,sm2,grad_mat_Q)
        ##### Likelihood
        ll += log_likelihood(W[0,j],m,Q,sm2)
        gll += grad_log_likelihood(W[0,j],m,Q,sm2,grad_mat_Q)
    # Predict for daughter cell
    m,Q = new_mean_cov(b,B,F,A,a)
    grad_mat_Q = grad_new_mean_cov(b,B,F,A,a,grad_param,grad_mat_b)
    # Find next cell initial conditions
    s = np.array([[m[0,0]-rescale*np.log(2)],[m[1,0]]])
    S = np.array([[Q[0,0],Q[0,1]],[Q[1,0],Q[1,1]]])
    grad_matS = {}
    def gv(x):
        grad_matS['{}'.format(x)] = \
        np.array([[grad_mat_Q['{}'.format(x)][0,0]],[grad_mat_Q['{}'.format(x)][1,0]]])
    def gm(x):
        mat = grad_mat_Q['{}'.format(x)]
        grad_matS['{}'.format(x)] = \
        np.array([[mat[0,0],mat[0,1]],[mat[1,0],mat[1,1]]])
    gv('m_mlam');gv('m_gamma');gv('m_sl2');gv('m_sm2')
    gm('Q_mlam');gm('Q_gamma');gm('Q_sl2');gm('Q_sm2')
    return -ll, -gll, s, S, grad_matS
#
def grad_obj_1lane(reind_,dat_,mlam,gamma,sl2,sm2,\
                   S,s,dt,grad_matS,rescale,nparr=False):
    """Compute the ll and gradient for 1 lane"""
    reind = deepcopy(reind_); dat = deepcopy(dat_)
    obj = 0; gobj = 0           # total objective and gradient
    for i in range(len(dat)):
        # Every time a cell do not have a "mother" (happen for first cells) intialize to initial conditions 
        if type(dat[i][0])!=tuple:
            dat[i][0] = [s,S,grad_matS]
        else:
            s,S,grad_matS = dat[i][0] # Unpack initial conditions
        # Find ll over 1 cell cycle for 1 daughter
        tmp =\
        obj_and_grad_1cc(W=dat[i][1][0],mlam=mlam,gamma=gamma,sl2=sl2,sm2=sm2,dt=dt,s=s,S=S,grad_matS=grad_matS,rescale=rescale) # calculate obj,gobj,p0 for one daugter
        obj += tmp[0]; gobj += tmp[1] #update obj, gobj
    # give the inital condition to the right cell lane
        if np.isnan(reind[i,0]) == False:
            dat[int(reind[i,0])][0] = tmp[2:] # s,S,grad_S
    #If the second cell exists do the same
        if np.sum(np.isnan(dat[i][1][1]))==0:
            tmp =\
            obj_and_grad_1cc(W=dat[i][1][1],mlam=mlam,gamma=gamma,sl2=sl2,sm2=sm2,dt=dt,s=s,S=S,grad_matS=grad_matS,rescale=rescale)
            obj += tmp[0]; gobj += tmp[1]
            if np.isnan(reind[i,1]) == False:
                dat[int(reind[i,1])][0] = tmp[2:]
    #Return obj and gradobj
    if nparr:
        return np.append(np.array([obj]),gobj)
    else:
        return obj, gobj
#@jit(parallel=True)
#def grad_obj_total(mlam,gamma,sl2,sm2,reind_v,dat_v,s,S,grad_matS,dt,rescale):
#    """Apply in parallel on all lane ID"""
#    tot_obj = 0; tot_grad = 0
#    for i in range(len(dat_v)):
#        reind = reind_v[i]; dat = dat_v[i]
#        obj, gobj =\
#        grad_obj_1lane(reind,dat,mlam,gamma,sl2,sm2,S,s,dt,grad_matS,rescale)
#        tot_obj += obj; tot_grad += gobj
#    return tot_obj, tot_grad

def grad_obj_total(mlam,gamma,sl2,sm2,reind_v,\
                            dat_v,s,S,grad_matS,dt,rescale,nproc=10):
    """Apply in parallel on all lane ID"""
    p = Pool(nproc)
    fun = lambda x:\
        grad_obj_1lane(x[0],x[1],mlam,gamma,sl2,sm2,S,s,dt,grad_matS,rescale,True)
    ret = p.map(fun,zip(reind_v,dat_v))
    ret = np.sum(np.vstack(ret),axis=0)
    return ret[0],ret[1:]
    #-------------------PREDICTIONS OVER CC/LANE AND TOTAL-----------------------------------------
def grad_obj_wrap(x,in_dic):
    mlam,gamma,sl2,sm2 = x
    reind_v,dat_v,grad_matS,s,S,dt,lane_ID_v,val_v,rescale =\
    in_dic['reind_v'],in_dic['dat_v'],in_dic['grad_matS'],in_dic['s'],in_dic['S'],in_dic['dt'],in_dic['lane_ID_v'],in_dic['val_v'],in_dic['rescale']
    return grad_obj_total(mlam,gamma,sl2,sm2,reind_v, dat_v,s,S,grad_matS,dt,rescale,nproc=10)

def predictions_1cc(W,mlam,gamma,sl2,sm2,dt,s,S,rescale):
    """Return optiman length and growth (z) and std """
    z = []; err_z=[]
    #### Initialize parameters for recurrence
    F, A, a = parameters(gamma,dt,mlam,sl2)
    ##### P(z_0|x_0^m)
    b,B = posteriori_matrices(W[0,0],s,S,sm2)
    z.append(np.array(b)); err_z.append(np.sqrt(np.array([B[0,0],B[1,1]])))
    for j in range(1,W.shape[1]):
       ###### P(z_{t+dt}|D_t) = N(m,Q))
        m,Q = new_mean_cov(b,B,F,A,a)
        ##### P(z_{t+dt}|D_{t+dt}) = N(b',B')
        b,B = posteriori_matrices(W[0,j],m,Q,sm2)
        ##### Optimal predicitons
        z.append(np.array(b)); err_z.append(np.sqrt(np.array([B[0,0],B[1,1]])))
    # Find next cell intial conditions
    m,Q = new_mean_cov(b,B,F,A,a)
    # Find next cell initial conditions
    s = np.array([[m[0,0]-rescale*np.log(2)],[m[1,0]]])
    S = np.array([[Q[0,0],Q[0,1]],[Q[1,0],Q[1,1]]])
    return z, err_z, s, S
#
def predictions_1lane(reind_,dat_,mlam,gamma,sl2,sm2,S,s,dt,lane_ID,val,rescale):
    """Return best_predictiona and error for every cell in form
    [laneID+id,[z,z_err]]"""
    from IPython.core.debugger import set_trace
    reind = deepcopy(reind_); dat = deepcopy(dat_)
    ret = [] # Return
    for i in range(len(dat)):
        # Every time a cell do not have a "mother" (happen for first cells) intialize to initial conditions
        if type(dat[i][0])!=tuple:
            dat[i][0] = [s,S]
        else:
            s,S= dat[i][0] # Unpack initial conditions
        # Find prediction over 1 cell cycle for 1 daughter
        tmp =\
        predictions_1cc(W=dat[i][1][0],mlam=mlam,gamma=gamma,sl2=sl2,sm2=sm2,dt=dt,s=s,S=S,rescale=rescale)
        z = tmp[0]; err_z = tmp[1]
        ret.append([lane_ID+'_'+str(val[i,1]),z,err_z])
        #print(np.mean(np.hstack(z).T[:,0]-dat[i][1][0]))
    # give the inital condition to the right cell lane
        if np.isnan(reind[i,0]) == False:
            dat[int(reind[i,0])][0] = tmp[2:] # s,S,grad_S
    #If the second cell exists do the same
        if np.sum(np.isnan(dat[i][1][1]))==0:
            tmp =\
            predictions_1cc(W=dat[i][1][1],mlam=mlam,gamma=gamma,sl2=sl2,sm2=sm2,dt=dt,s=s,S=S,rescale=rescale)
            z = tmp[0]; err_z = tmp[1]
            ret.append([lane_ID+'_'+str(val[i,2]),z,err_z])
            #print(np.mean(np.hstack(z).T[:,0]-dat[i][1][1]))
            if np.isnan(reind[i,1]) == False:
                dat[int(reind[i,1])][0] = tmp[2:]
        #set_trace()
    #Return obj and gradobj
    return ret
#
#def prediction_total(mlam,gamma,sl2,sm2,reind_v,dat_v,s,S,dt,lane_ID_v,val_v,rescale):
#    """Apply in parallel on all lane ID and return a np.array"""
#    tmp = []
#    for i in range(len(dat_v)):
#        reind = reind_v[i]; dat = dat_v[i]
#        tmp.append(predictions_1lane(reind,dat,mlam,gamma,sl2,sm2,S,s,dt,lane_ID_v[i],val_v[i],rescale=rescale))
#    # Return a nice behave np array with cell_ID, z[0],z[1],err_z[0],err_z[1]
#    foo = []
#    for lan in tmp:
#        for cid in lan:
#            foo.append(np.hstack((np.hstack([cid[0]]*len(cid[1]))[:,None],np.hstack(cid[1]).T,np.vstack(cid[2]))))
#    return np.vstack(foo)
def prediction_total(mlam,gamma,sl2,sm2,reind_v,dat_v,s,S,dt,lane_ID_v,val_v,rescale,nproc=10):
    """Apply in parallel on all lane ID and return a np.array"""
    p = Pool(nproc)
    fun = lambda x:\
        predictions_1lane(x[0],x[1],mlam,gamma,sl2,sm2,S,s,dt,x[2],x[3],rescale)
    tmp = p.map(fun,zip(reind_v,dat_v,lane_ID_v,val_v))
    # Return a nice behave np array with cell_ID, z[0],z[1],err_z[0],err_z[1]
    foo = []
    for lan in tmp:
        for cid in lan:
            foo.append(np.hstack((np.hstack([cid[0]]*len(cid[1]))[:,None],np.hstack(cid[1]).T,np.vstack(cid[2]))))
    return np.vstack(foo)
def cost_function(x,in_dic,r2=True):
    """Return the r2 between predicted and observed data"""
    mlam,gamma,sl2,sm2 = x
    reind_v,dat_v,vec_dat_v,s,S,dt,lane_ID_v,val_v,rescale =\
    in_dic['reind_v'],in_dic['dat_v'],in_dic['vec_dat_v'],in_dic['s'],in_dic['S'],in_dic['dt'],in_dic['lane_ID_v'],in_dic['val_v'],in_dic['rescale']
    predicted =\
    prediction_total(mlam,gamma,sl2,sm2,reind_v,dat_v,s,S,dt,lane_ID_v,val_v,rescale)[:,1:2]
    # nan is used to keep same structure eventough we do not have gradient
    if r2:
        return\
    1-np.sum((predicted.astype(np.float)-vec_dat_v)**2)/np.sum((vec_dat_v-np.mean(vec_dat_v))**2),\
            np.array([np.nan,np.nan,np.nan,np.nan])
    else:
        return np.log(np.sum((predicted.astype(np.float)-vec_dat_v)**2)),\
                np.array([np.nan,np.nan,np.nan,np.nan])

def predict(min_dic, in_dic):
    """It is just a  = xwrapper to keep nice data structure where min_dic is the dict
    returned by minimize and in dic the one returned by build_data_strucure"""
    md = min_dic['best_param']
    return prediction_total(md['mlam'],md['gamma'],md['sl2'],md['sm2'],\
                            in_dic['reind_v'],in_dic['dat_v'],in_dic['s'],\
                           in_dic['S'],in_dic['dt'],in_dic['lane_ID_v'],\
                            in_dic['val_v'],in_dic['rescale'])
################################################################################################
############################## DATA TREATEMENT #################################################
################################################################################################
def build_intial_mat(df,leng):
    """ Build the intiala matrix s,S and intial gradient grad_S """
    from scipy.stats import linregress
    # Computation done in minutes (that's why *60)
    def sleres(t,y):
        """Return slope (lambda), intercept (x0) and residuals (form sm2)"""
        t = t - t.iloc[0]
        r = linregress(t,y)
        return r.slope,r.intercept, y-(r.intercept+t*r.slope)
    G = df.groupby('cell').apply(lambda x: sleres(x.time_sec/60.,x['{}'.format(leng)]))
    G = np.vstack(G)
    lam = G[:,0]; x0 = G[:,1]; res = G[:,2]
    res = np.array([k for h in res for k in h])
    sm2 = np.var(res)
    ####### They do not depend on the parameters
    mat = np.zeros((2,2)); vec = np.zeros((2,1))
    grad_matS = {'m_mlam':vec,'m_gamma':vec,'m_sl2':vec,'m_sm2':vec,\
                 'Q_mlam':mat,'Q_gamma':mat,'Q_sl2':mat,'Q_sm2':mat}
    s = np.array([[np.mean(x0)],[np.mean(lam)]])
    S = np.array([[np.var(x0),0],[0,np.var(lam)]])
    return s, S, grad_matS,sm2
#
def build_mat(mother,daugther,dfl,leng):
    """Find the correct structure. It will return [mum_ind,daughter1_ind,daughter2_ind] and [mum_ind,[Len(dau1),Len(daug2)]]"""
    tmp = []; tmp1=[]
    for i,j in zip(mother,daugther):
        which = lambda x: dfl.loc[dfl['id']==j[x]]['{}'.format(leng)].values[None,:]
        if j.shape[0]==1:
            tmp.append([i,j[0],np.nan])
            tmp1.append([i,[which(0),np.nan]])
        else:
            tmp1.append([i,[which(0),which(1)]])
            tmp.append([i,j[0],j[1]])
    return np.vstack(tmp),tmp1
#
def who_goes_where(val):
    """ Connects the geanology e.g. if val [[19,[20,21]],[20,[22,nan]],[21,[23,nan]] ] it will return [[1,2],..] saying that division of cell 20 (daughter of 19) is at index 1 """
    tmp1 = [np.where(val[:,:1]==k)[0] for k in val [:,1:2]]
    tmp2 = [np.where(val[:,:1]==k)[0] for k in val [:,2:3]]
    fun = lambda x: np.nan if x.size==0 else x
    tmp1 = np.vstack(list(map(fun,tmp1)))
    tmp2 = np.vstack(list(map(fun,tmp2)))
    return np.hstack([tmp1,tmp2])
#
def build_data_strucutre(df,leng,rescale):
    """Return for every lane the data with respective daughteres and initial conditions"""
    #Sometimes cells with 1 data point are present and we don't want them
    df = df.groupby('cell').filter(lambda x: x.values.shape[0]>1) #
    dt = (df['time_sec'].iloc[1]-df['time_sec'].iloc[0])/60
    assert dt == 3 or dt == 6, "look if dt make sense"
    #rescae
    df['log_resc_'+leng] = np.log(df['{}'.format(leng)])*rescale
    #print("The variable to use is: log_resc_{}".format(leng))
    s,S,grad_matS,sm = build_intial_mat(df,leng='log_resc_'+leng)
	#n_point = df.shape[0]
    n_point = df.shape[0]
    dat_v = []; reind_v = [];
    val_v = []; lane_ID_v = []
    for lid in df['lane_ID'].unique():
        dfl = df.loc[df['lane_ID']==lid] # select one lande id
        daugther = [dfl.loc[dfl['parent_id']==k].id.unique() for k in dfl.parent_id.unique()] # find its daughters
        mothers = dfl.parent_id.unique() # respective mothers
        # Correct data strucutre
        val, dat = build_mat(mothers,daugther,dfl,'log_resc_'+leng)
        reind = who_goes_where(val)
        dat_v.append(dat); reind_v.append(reind)
        val_v.append(val); lane_ID_v.append(lid)
    vec_dat_v = []
    for k in dat_v:
            for j in k:
                vec_dat_v.append(j[1][0])
                if np.any(np.isnan(j[1][1])):
                    continue
                else:
                    vec_dat_v.append(j[1][1])
    vec_dat_v = np.hstack(vec_dat_v).T
    return df,{'n_point':n_point,'dt':dt,'s':s,'S':S,'grad_matS':grad_matS,\
            'reind_v':reind_v,'dat_v':dat_v, 'val_v':val_v,\
               'lane_ID_v':lane_ID_v,'rescale':rescale,'sm2':sm,\
              'vec_dat_v':vec_dat_v}
#
def merge_df_pred(df,pred_mat):
    """Merge the output from predict with the initial dataframe"""
    # From mat to dataframe
    dft = pd.DataFrame(pred_mat,columns=\
                       ('cell_','pred_z0','pred_growth_rate',\
                        'err_z0','err_growth_rate'))
    # Give numerical values
    dft[['pred_z0','pred_growth_rate','err_z0','err_growth_rate']] = \
            dft[['pred_z0','pred_growth_rate','err_z0','err_growth_rate']].apply(pd.to_numeric)
    #Create subindex for merging
    dft['sub_ind'] = dft.groupby('cell_')['pred_z0'].transform(lambda x:\
                                                               np.arange(len(x)))
    # Create same indexing in df
    df['cell_'] = df['lane_ID']+df['id'].apply(lambda x: '_'+str(x)+'.0')
    df['sub_ind'] = df.groupby('cell_')['time_sec'].transform(lambda x:\
                                                              np.arange(len(x)))
    #Concat, reindex, delete column used for mergin 
    dff = \
    pd.concat([df.set_index(['cell_','sub_ind']),dft.set_index(['cell_','sub_ind'])],axis=1)
    dff = dff.reset_index()
    return dff.drop(['cell_','sub_ind'],axis=1)
def find_best_lengths(files,pwd='/scicore/home/nimwegen/fiori/MoMA_predictions/predictions/',lengths=['length_um','length_erik','length_moma','length_raw','length_pixel']):
    """Find best length measurment """
    tmp=[]
    for k in lengths:
        df = pd.read_csv(pwd+files+'_'+k+'.csv')
        err = np.mean(np.abs(df['err_growth_rate']/df['pred_growth_rate']))
        tmp.append([files,k,err])
    tmp = np.vstack(tmp)
    return tmp[tmp[:,2]==min(tmp[:,2])]
################################################################################################
############################### The stocastics models  #########################################
################################################################################################
def ornstein_uhlenbeck(mlam,gamma,sl2,length=30,ncel=10,dt=3.,dtsim=1):
    sam = dt/dtsim
    lengthsim = length*sam
    assert (sam).is_integer(), "not sam integer"
    assert (lengthsim).is_integer(), "no lan integer"
    sam = int(sam); lengthsim=int(lengthsim)
    mat = np.zeros((ncel,lengthsim))
    sig = np.sqrt(sl2)
    dW = np.random.normal(loc=mat,scale=np.sqrt(dtsim))
    add = sig*dW*dtsim
    mat[:,0]=add[:,0]+mlam
    for k in range(1,lengthsim):
        mat[:,k]=mat[:,k-1]-gamma*(mat[:,k-1]-mlam)*dtsim+add[:,k]
    return mat[:,::sam]
def integrated_ou(mlam,gamma,sl2,sm2,X0=1,sx0=0.1,length=30,ncel=10,dt=3.,dtsim=1):
    X = ornstein_uhlenbeck(mlam,gamma,sl2,length,ncel,dt,dtsim)
    X0 = np.random.normal(loc=np.ones((ncel,1)),scale=sx0)
    return np.random.normal(loc=np.hstack([X0,np.cumsum(X,axis=1)*dt+X0]),scale=np.sqrt(sm2))[:,:-1], X
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
    #print(obj_and_grad_1cc(W,mlam=1.,gamma=0.02,sl2=.03**2,sm2=0.08,dt=3.,m0=m0,M0=M0,rescale=rescale))
    print(time.time()-t1)
    #z, err_z,_,_ = predictions_1cc(W,mlam=1.,gamma=0.02,sig_l_s=.03**4,sig_m_s=0.08,dt=3.,m0=m0,M0=M0)
    #print(W_er(1),W_er(2))
    #print(X_er(1),X_er(2))
