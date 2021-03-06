import numpy as np
from numba import jit,prange
from copy import deepcopy
import pandas as pd
import time
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.stats import linregress
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
def hessian_parameters(g,dt,ml,sl2):
    """The hessian and gradient of the parameters, g=gamma,ml=mlam"""
    ret = {}
    for k in ['m','g','s','e']:
        ret['A_{}'.format(k)] = np.zeros((2,2))
        ret['a_{}'.format(k)] = np.zeros((2,1))
        ret['F_{}'.format(k)] = np.zeros((2,2))
        for j in ['m','g','s','e']:
            ret['A_{}{}'.format(k,j)] = np.zeros((2,2))
            ret['a_{}{}'.format(k,j)] = np.zeros((2,1))
            ret['F_{}{}'.format(k,j)] = np.zeros((2,2))
    # Reformat derivatives
    tmp = grad_parameters(g,dt,ml,sl2)
    ret['F_g'] = tmp['F_gamma']
    ret['a_m'] = tmp['a_mlam']
    ret['a_g'] = tmp['a_gamma']
    ret['A_s'] = tmp['A_sl2']
    ret['A_g'] = tmp['A_gamma']
    # The derivatives of A
    A_gg = np.zeros((2,2))
    A_gsl = np.zeros((2,2))
    A_gg[0,0] =  (2*(-3 - 3*dt*g - dt**2*g**2 + 3*np.exp(2*dt*g)*(-3 + dt*g) +\
                np.exp(dt*g)*(12 + 6*dt*g + dt**2*g**2))*sl2)/(np.exp(2*dt*g)*g**5)
    A_gg[0,1] = ((3 + 3*np.exp(2*dt*g) + 4*dt*g + 2*dt**2*g**2 -\
                np.exp(dt*g)*(6 + 4*dt*g + dt**2*g**2))*sl2)/(np.exp(2*dt*g)*g**4)
    A_gg[1,1] = ((-1 + np.exp(2*dt*g) - 2*dt*g - 2*dt**2*g**2)*sl2)/(np.exp(2*dt*g)*g**3)
    A_gg[1,0] = A_gg[0,1]

    A_gsl[0,0] = (3 + 2*dt*g + np.exp(2*dt*g)*(9 - 4*dt*g) - 4*np.exp(dt*g)*(3 + dt*g))/(2.*np.exp(2*dt*g)*g**4)
    A_gsl[0,1] =  -(((-1 + np.exp(dt*g))*(-1 + np.exp(dt*g) - dt*g))/(np.exp(2*dt*g)*g**3))
    A_gsl[1,1] = -(-1 + np.exp(2*dt*g) - 2*dt*g)/(2.*np.exp(2*dt*g)*g**2)
    A_gsl[1,0] = A_gsl[0,1]
    # The derivatives of F
    F_gg = np.zeros((2,2))
    F_gg[0,1] = (-2 + 2*np.exp(dt*g) - 2*dt*g - dt**2*g**2)/(np.exp(dt*g)*g**3)
    F_gg[1,1] = dt**2/np.exp(dt*g)
    # The derivatives of a
    a_gg = np.zeros((2,1))
    a_gml = np.zeros((2,1))
    a_gg[0,0] = ((2 - 2*np.exp(dt*g) + 2*dt*g + dt**2*g**2)*ml)/(np.exp(dt*g)*g**3)
    a_gg[1,0] = -((dt**2*ml)/np.exp(dt*g))
    a_gml[0,0] = (-1 + np.exp(dt*g) - dt*g)/(np.exp(dt*g)*g**2)
    a_gml[1,0] = dt/np.exp(dt*g)
    #Reformat
    ret['A_gg']=A_gg;ret['A_gs']=A_gsl;ret['F_gg']=F_gg;ret['a_gg']=a_gg;ret['a_gm']=a_gml
    ret['A_sg']=ret['A_gs'];ret['a_mg']=ret['a_gm']
    return ret
def new_mean_cov(b, B,F,A,a):
    """Start from P(z_t|D_t)= N(b;B) and find P(z_{t+dt}|D_t) =
    N(a+Fb;A+FBF.T):= N(m,Q)  """
    # No numpy modules for speed up in numba
    m = a + np.dot(F,b)
    Q = A+np.dot(F,np.dot(B,F.T))
    return m, Q
def grad_new_mean_cov(b,B,F,A,a,grad_para,grad_mat_b):
    """Grad m and Q """
    F_gamma=grad_para['F_gamma']
    #######################################################
    # Grad m
    ######################################################
    fdgm = lambda x: np.dot(F,grad_mat_b['{}'.format(x)])
    m_mlam = grad_para['a_mlam']+fdgm('b_mlam')
    m_gamma = grad_para['a_gamma'] + fdgm('b_gamma')+np.dot(F_gamma,b)
    m_sl2 = fdgm('b_sl2')
    m_sm2 = fdgm('b_sm2')
    m_sd2 = fdgm('b_sd2')
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
    Q_sd2 = fdgf('B_sd2')
    return {'m_mlam':m_mlam,'m_gamma':m_gamma,'m_sl2':m_sl2,'m_sm2':m_sm2,\
            'Q_mlam':Q_mlam,'Q_gamma':Q_gamma,'Q_sl2':Q_sl2,'Q_sm2':Q_sm2,\
            'm_sd2':m_sd2,'Q_sd2':Q_sd2}
def hessian_new_mean_cov(b,B,F,A,a,H_para,H_mat_b):
    """Hessian and gradient of m and Q """
    trmul = lambda x,y,z: np.dot(x,np.dot(y,z.T))
    #######################################################
    # THE GRADIENTS
    #######################################################
    m_k = lambda k: H_para['a_{}'.format(k)]+ np.dot(H_para['F_{}'.format(k)],b)+np.dot(F,H_mat_b['b_{}'.format(k)])
    Q_k = lambda k: H_para['A_{}'.format(k)]+\
            trmul(H_para['F_{}'.format(k)],B,F)+\
            trmul(F,H_mat_b['B_{}'.format(k)],F)+\
            trmul(F,B,H_para['F_{}'.format(k)])
    #######################################################
    # THE HESSIAN
    #######################################################
    m_kj = lambda k,j:\
        H_para['a_{}{}'.format(k,j)]+np.dot(H_para['F_{}{}'.format(k,j)],b)+np.dot(F,H_mat_b['b_{}{}'.format(k,j)])+\
         np.dot(H_para['F_{}'.format(k)],H_mat_b['b_{}'.format(j)]) +\
         np.dot(H_para['F_{}'.format(j)],H_mat_b['b_{}'.format(k)])
    Q_kj = lambda k,j: H_para['A_{}{}'.format(k,j)]+\
            trmul(H_para['F_{}{}'.format(k,j)],B,F)+\
            trmul(H_para['F_{}'.format(k)],H_mat_b['B_{}'.format(j)],F)+\
            trmul(H_para['F_{}'.format(k)],B,H_para['F_{}'.format(j)])+\
            trmul(H_para['F_{}'.format(j)],H_mat_b['B_{}'.format(k)],F)+\
            trmul(F,H_mat_b['B_{}{}'.format(k,j)],F)+\
            trmul(F,H_mat_b['B_{}'.format(k)],H_para['F_{}'.format(j)])+\
            trmul(F,B,H_para['F_{}{}'.format(k,j)])+\
            trmul(F,H_mat_b['B_{}'.format(j)],H_para['F_{}'.format(k)])+\
            trmul(H_para['F_{}'.format(j)],B,H_para['F_{}'.format(k)])
    ########################################################
    # FILL THE MATRIX
    ########################################################
    ret = {}
    for k in ['m','g','s','e']:
        ret['m_{}'.format(k)] = m_k(k)
        ret['Q_{}'.format(k)] = Q_k(k)
        assert np.allclose(Q_k(k),Q_k(k).T)
        for j in ['m','g','s','e']:
            ret['m_{}{}'.format(k,j)] = m_kj(k,j)
            ret['Q_{}{}'.format(k,j)] = Q_kj(k,j)
#    for k in ['m','g','s','e']:
#        for j in ['m','g','s','e']:
#            assert np.allclose(ret['Q_{}{}'.format(k,j)] ,\
#                          ret['Q_{}{}'.format(j,k)]),'{}{}{}{}'.format(k,j,ret['Q_{}{}'.format(k,j)],ret['Q_{}{}'.format(j,k)])
#            assert np.allclose( ret['Q_{}{}'.format(k,j)] ,ret['Q_{}{}'.format(k,j)])
    return ret
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
    """Grad b and B """
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
            'B_sd2':dB('Q_sd2'),'b_sd2':db('m_sd2','Q_sd2'),'B_sm2':B_sm2,'b_sm2':b_sm2,\
            'b_gamma':db('m_gamma','Q_gamma'),'b_mlam':db('m_mlam','Q_mlam'),'b_sl2':db('m_sl2','Q_sl2')}
def hessian_posteriori_matrices(xm,m,Q,sm2,H_mat):
    """Return the gradient and the hessian of the posteriori matrices b and B. xm is the datapoint"""
    ########################################################
    # GRADIENT AND HESSIAN OF B AND b
    ########################################################
    def B00_x(x):
        if x=='e': sd=1
        else: sd=0
        dQ = H_mat['Q_{}'.format(x)][0,0]
        Q0=Q[0,0]; den=Q0+sm2
        return ((sd*Q0+sm2*dQ)*den-sm2*Q0*(dQ+sd))/den**2
    def B00_xy(x,y):
        if x=='e': dxe=1
        else: dxe=0
        if y=='e': dye=1
        else: dye=0
        dQx = H_mat['Q_{}'.format(x)][0,0]
        dQy = H_mat['Q_{}'.format(y)][0,0]
        dQxy = H_mat['Q_{}{}'.format(x,y)][0,0]
        Q00=Q[0,0]
        return (-2*dxe*dye*Q00**2 + (2*dQy*dxe + 2*dQx*dye)*Q00*sm2 +\
        (-2*dQx*dQy + dQxy*Q00)*sm2**2 + dQxy*sm2**3)/(Q00 + sm2)**3
    def B01_x(x):
        if x=='e': dxe=1
        else: dxe=0
        dQ = H_mat['Q_{}'.format(x)]
        dQ00x = dQ[0,0];dQ01x = dQ[0,1]
        Q00=Q[0,0]; Q01=Q[0,1]
        return (dQ01x*sm2*(Q00 + sm2) + (-(dQ00x*sm2) + Q00*dxe)*Q01)/(Q00 + sm2)**2
    def B01_xy(x,y):
        if x=='e': dxe=1
        else: dxe=0
        if y=='e': dye=1
        else: dye=0
        Q00=Q[0,0];Q01=Q[0,1]
        dQx = H_mat['Q_{}'.format(x)]
        dQy = H_mat['Q_{}'.format(y)]
        dQxy = H_mat['Q_{}{}'.format(x,y)]
        dQ00x = dQx[0,0];dQ00y = dQy[0,0]
        dQ01x = dQx[0,1];dQ01y = dQy[0,1]
        dQ00xy= dQxy[0,0];dQ01xy= dQxy[0,1];dQ11xy= dQxy[1,1]

        return    (Q01*((-(dQ00y*dxe) - (dQ00x + 2*dxe)*dye)*Q00 + (dQ00y*(2*dQ00x + dxe) + dQ00x*dye - dQ00xy*Q00)*sm2 -\
            dQ00xy*sm2**2) + (Q00 + sm2)*((dQ01y*dxe + dQ01x*dye)*Q00 - (dQ00y*dQ01x + dQ00x*dQ01y - dQ01xy*Q00)*sm2 +\
            dQ01xy*sm2**2))/(Q00 + sm2)**3
    def B11_x(x):
        if x=='e': dxe=1
        else: dxe=0
        Q00=Q[0,0];Q01=Q[0,1]
        dQx = H_mat['Q_{}'.format(x)]
        dQ00x=dQx[0,0]; dQ01x=dQx[0,1]; dQ11x=dQx[1,1]
        return dQ11x + ((dQ00x + dxe)*Q01**2)/(Q00 + sm2)**2 - (2*dQ01x*Q01)/(Q00 + sm2)
    def B11_xy(x,y):
        if x=='e': dxe=1
        else: dxe=0
        if y=='e': dye=1
        else: dye=0
        Q00=Q[0,0];Q01=Q[0,1]
        dQx = H_mat['Q_{}'.format(x)]
        dQy = H_mat['Q_{}'.format(y)]
        dQxy = H_mat['Q_{}{}'.format(x,y)]
        dQ00x = dQx[0,0];dQ00y = dQy[0,0]
        dQ01x = dQx[0,1];dQ01y = dQy[0,1]
        dQ00xy= dQxy[0,0];dQ01xy= dQxy[0,1];dQ11xy= dQxy[1,1]
        return   dQ11xy - (2*(dQ00x + dxe)*(dQ00y + dye)*Q01**2)/(Q00 + sm2)**3 +\
        (2*dQ01y*(dQ00x + dxe)*Q01)/(Q00 + sm2)**2 +\
        (2*dQ01x*(dQ00y + dye)*Q01)/(Q00 + sm2)**2 + (dQ00xy*Q01**2)/(Q00 + sm2)**2 -\
        (2*dQ01x*dQ01y)/(Q00 + sm2) - (2*dQ01xy*Q01)/(Q00 + sm2)
    def b0_x(x):
        if x=='e': dxe=1
        else: dxe=0
        dQ00x = H_mat['Q_{}'.format(x)][0,0]
        dm00x = H_mat['m_{}'.format(x)][0,0]
        Q00=Q[0,0];m00=m[0,0]
        return (Q00*(dm00x*sm2 + dxe*(m00 - xm)) + sm2*(dm00x*sm2 + dQ00x*(-m00 + xm)))/(Q00 + sm2)**2
    def b0_xy(x,y):
        if x=='e': dxe=1
        else: dxe=0
        if y=='e': dye=1
        else: dye=0
        dQ00x = H_mat['Q_{}'.format(x)][0,0]
        dQ00xy = H_mat['Q_{}{}'.format(x,y)][0,0]
        dQ00y = H_mat['Q_{}'.format(y)][0,0]
        dm00x = H_mat['m_{}'.format(x)][0,0]
        dm00y = H_mat['m_{}'.format(y)][0,0]
        dm00xy = H_mat['m_{}{}'.format(x,y)][0,0]
        Q00=Q[0,0];m00=m[0,0]
        return -(((dQ00y + dye)*(Q00 + sm2)*(dxe*m00 + dm00x*sm2 + dQ00x*xm) -\
           (Q00 + sm2)**2*(dm00y*dxe + dm00x*dye + dm00xy*sm2 + dQ00xy*xm) +\
           (dQ00x + dxe)*(Q00 + sm2)*(dye*m00 + dm00y*sm2 + dQ00y*xm) -\
           2*(dQ00x + dxe)*(dQ00y + dye)*(m00*sm2 + Q00*xm) +\
           dQ00xy*(Q00 + sm2)*(m00*sm2 + Q00*xm))/(Q00 + sm2)**3)
    def b1_x(x):
        if x=='e': dxe=1
        else: dxe=0
        dQx = H_mat['Q_{}'.format(x)]
        dQ00x = dQx[0,0]
        dQ01x = dQx[0,1]
        dmx = H_mat['m_{}'.format(x)]
        dm10x = dmx[1,0];dm00x = dmx[0,0]
        Q00=Q[0,0];m00=m[0,0]
        Q01=Q[0,1];m10=m[1,0]
        return dm10x - (dm00x*Q01)/(Q00 + sm2) -\
              ((dQ00x + dxe)*Q01*(-m00 + xm))/(Q00 + sm2)**2 + (dQ01x*(-m00 + xm))/(Q00 + sm2)
    def b1_xy(x,y):
        if x=='e': dxe=1
        else: dxe=0
        if y=='e': dye=1
        else: dye=0
        Q00=Q[0,0];Q01=Q[0,1];m00=m[0,0]
        dQx = H_mat['Q_{}'.format(x)]
        dQy = H_mat['Q_{}'.format(y)]
        dQxy = H_mat['Q_{}{}'.format(x,y)]
        dQ00x = dQx[0,0];dQ00y = dQy[0,0]
        dQ01x = dQx[0,1];dQ01y = dQy[0,1]
        dQ00xy= dQxy[0,0];dQ01xy= dQxy[0,1];dQ11xy= dQxy[1,1]
        dmx = H_mat['m_{}'.format(x)]
        dmy = H_mat['m_{}'.format(y)]
        dmxy = H_mat['m_{}{}'.format(x,y)]
        dm10x = dmx[1,0];dm00x = dmx[0,0]
        dm10y = dmy[1,0];dm00y = dmy[0,0]
        dm10xy = dmxy[1,0];dm00xy = dmxy[0,0]
        return -((-(dm00y*(dQ00x + dxe)*Q01*(Q00 + sm2)) -\
           dm00x*(dQ00y + dye)*Q01*(Q00 + sm2) + dm00y*dQ01x*(Q00 + sm2)**2 +\
           dm00x*dQ01y*(Q00 + sm2)**2 + dm00xy*Q01*(Q00 + sm2)**2 -\
           dm10xy*(Q00 + sm2)**3 - 2*(dQ00x + dxe)*(dQ00y + dye)*Q01*(-m00 + xm) +\
           dQ01y*(dQ00x + dxe)*(Q00 + sm2)*(-m00 + xm) +\
           dQ01x*(dQ00y + dye)*(Q00 + sm2)*(-m00 + xm) +\
           dQ00xy*Q01*(Q00 + sm2)*(-m00 + xm) - dQ01xy*(Q00 + sm2)**2*(-m00 + xm))/\
            (Q00 + sm2)**3)
    ########################################################
    # FILL THE MATRIX
    ########################################################
    ret = {}
    build_vec = lambda x,y: np.array([[x],[y]])
    build_mat= lambda x,y,z: np.array([[x,y],[y,z]])
    for k in ['m','g','s','e']:
        ret['b_{}'.format(k)] = build_vec(b0_x(k),b1_x(k))
        ret['B_{}'.format(k)] = build_mat(B00_x(k),B01_x(k),B11_x(k))
        for j in ['m','g','s','e']:
            ret['b_{}{}'.format(k,j)] = build_vec(b0_xy(k,j),b1_xy(k,j))
            ret['B_{}{}'.format(k,j)] =\
            build_mat(B00_xy(k,j),B01_xy(k,j),B11_xy(k,j))
    return ret
def log_likelihood(x,m,Q,sm2):
    """Return P(x_{t+dt}|D_t) in log """
    # if den gets too small..
    #np.seterr(invalid='raise') # if log negative stop everything
    den = Q[0,0]+sm2
    assert Q[0,0]>=0, "Q[00]={} must be positivie! The paramters are non\
    physicals conditions or increase the rescaling".format(Q[0,0])
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
                    [grad('m_sm2','Q_sm2',1)],\
                    [grad('m_sd2','Q_sd2')],\
                     ])

def hessian_log_likelihood(xm,m,Q,sm2,H_mat):
    """Give gradient and hessian of log lik"""
    def G(x):
        if x=='e': dxe=1
        else: dxe=0
        dQ00x = H_mat['Q_{}'.format(x)][0,0]
        dm00x = H_mat['m_{}'.format(x)][0,0]
        Q00=Q[0,0];m00=m[0,0]
        return (-((dQ00x + dxe)*(Q00 + sm2)) + 2*dm00x*(Q00 + sm2)*(-m00 + xm)\
                + (dQ00x + dxe)*(-m00 + xm)**2)/(2.*(Q00 + sm2)**2)
    def H(x,y):
        if x=='e': dxe=1
        else: dxe=0
        if y=='e': dye=1
        else: dye=0
        dQ00x = H_mat['Q_{}'.format(x)][0,0]
        dQ00xy = H_mat['Q_{}{}'.format(x,y)][0,0]
        dQ00y = H_mat['Q_{}'.format(y)][0,0]
        dm00x = H_mat['m_{}'.format(x)][0,0]
        dm00y = H_mat['m_{}'.format(y)][0,0]
        dm00xy = H_mat['m_{}{}'.format(x,y)][0,0]
        Q00=Q[0,0];m00=m[0,0]
        return         ((dQ00x + dxe)*(dQ00y + dye)*(Q00 + sm2) - 2*dm00x*dm00y*(Q00 + sm2)**2 -\
         dQ00xy*(Q00 + sm2)**2 - 2*dm00y*(dQ00x + dxe)*(Q00 + sm2)*(-m00 + xm) -\
         2*dm00x*(dQ00y + dye)*(Q00 + sm2)*(-m00 + xm) + 2*dm00xy*(Q00 + sm2)**2*(-m00 + xm) -\
         2*(dQ00x + dxe)*(dQ00y + dye)*(-m00 + xm)**2 + dQ00xy*(Q00 + sm2)*(-m00 + xm)**2)/\
       (2.*(Q00 + sm2)**3)
    grad = np.array([G(x) for x in ['m','g','s','e']])[:,None]
    hess = np.array([H(x,y) for y in ['m','g','s','e'] for x in\
                     ['m','g','s','e']]).reshape(4,4)
    return grad, hess
def cell_division_likelihood_and_grad(m,Q,grad_mat_Q,sd2,rescale,grad=True):
    grad_matS = {}
    S = np.array([[Q[0,0]+sd2,Q[0,1]],[Q[0,1],Q[1,1]]])
    s = np.array([[m[0,0]-rescale*np.log(2)],[m[1,0]]])
    grad_matS['Q_sd2'] =\
        np.array([[1,0],[0,0]])
    grad_matS['m_sd2'] =np.array([[0],[0]])
    def gm(x):
        GQ = grad_mat_Q['{}'.format(x)]
        grad_matS['{}'.format(x)] = \
                np.array([[GQ[0,0],GQ[0,1]],[GQ[0,1],GQ[1,1]]])
    def gv(x):
        mat = grad_mat_Q['{}'.format(x)]
        grad_matS['{}'.format(x)] = \
        np.array([[mat[0,0]],[mat[1,0]]])
    if grad:
        gv('m_mlam');gv('m_gamma');gv('m_sl2');gv('m_sm2')
        gm('Q_mlam');gm('Q_gamma');gm('Q_sl2');gm('Q_sm2')
        return s, S, grad_matS
    else:
        return s,S
def hess_cell_division_likelihood(m,Q,grad_mat_Q,sd2,rescale):
    """With the new defintion of the division we do not change gradient"""
    S = np.array([[Q[0,0]+sd2,Q[0,1]],[Q[0,1],Q[1,1]]])
    s = np.array([[m[0,0]-rescale*np.log(2)],[m[1,0]]])
    return s,S,grad_mat_Q
def obj_and_grad_1cc(W,mlam,gamma,sl2,sm2,dt,s,S,grad_matS,rescale,sd2):
    """Objective and gradient over 1 cell cycle"""
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
    # Find next cell initial conditions (9% asym div)
    s, S, grad_matS = cell_division_likelihood_and_grad(m,Q,grad_mat_Q,sd2,rescale)
    return -ll, -gll, s, S, grad_matS
def is_pos_def(x):
    """Check hessian x positive defined"""
    return x==x.T and  np.all(np.linalg.eigvals(np.linalg.inv(-x))>0)
def hessian_1cc(W,mlam,gamma,sl2,sm2,dt,s,S,grad_matS,rescale,sd2):
    """grad and Hessian 1 cell cycle"""
    ##### likelihood and gradient at initial conditions
    gll,hll = hessian_log_likelihood(W[0,0],s,S,sm2,grad_matS)
    #### Initialize parameters for recurrence
    F, A, a = parameters(gamma,dt,mlam,sl2)
    grad_param = hessian_parameters(gamma,dt,mlam,sl2)
    ##### P(z_0|x_0^m)
    b,B = posteriori_matrices(W[0,0],s,S,sm2)
    grad_mat_b = hessian_posteriori_matrices(W[0,0],s,S,sm2,grad_matS)
    for j in range(1,W.shape[1]):
        ###### P(z_{t+dt}|D_t) = N(m,Q))
        m,Q = new_mean_cov(b,B,F,A,a)
        grad_mat_Q = hessian_new_mean_cov(b,B,F,A,a,grad_param,grad_mat_b)
        ##### P(z_{t+dt}|D_{t+dt}) = N(b',B')
        b,B = posteriori_matrices(W[0,j],m,Q,sm2)
        grad_mat_b = hessian_posteriori_matrices(W[0,j],m,Q,sm2,grad_mat_Q)
        ##### Likelihood
        G,H = hessian_log_likelihood(W[0,j],m,Q,sm2,grad_mat_Q)
        gll+=G; hll=+H
    # Predict for daughter cell
    m,Q = new_mean_cov(b,B,F,A,a)
    grad_mat_Q = hessian_new_mean_cov(b,B,F,A,a,grad_param,grad_mat_b)
    # Find next cell initial conditions (9% asym div)
    s, S, grad_matS = hess_cell_division_likelihood(m,Q,grad_mat_Q,sd2,rescale)
    return hll, gll, s, S, grad_matS
def grad_obj_1lane(reind_,dat_,mlam,gamma,sl2,sm2,\
                   S,s,dt,grad_matS,rescale,sd2,\
                   nparr=False,fun=obj_and_grad_1cc):
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
        fun(W=dat[i][1][0],mlam=mlam,gamma=gamma,sl2=sl2,sm2=sm2,dt=dt,s=s,S=S,grad_matS=grad_matS,rescale=rescale,sd2=sd2) # calculate obj,gobj,p0 for one daugter
        obj += tmp[0]; gobj += tmp[1] #update obj, gobj
    # give the inital condition to the right cell lane
        if np.isnan(reind[i,0]) == False:
            dat[int(reind[i,0])][0] = tmp[2:] # s,S,grad_S
    #If the second cell exists do the same
        if np.sum(np.isnan(dat[i][1][1]))==0:
            tmp =\
            fun(W=dat[i][1][1],mlam=mlam,gamma=gamma,sl2=sl2,sm2=sm2,dt=dt,s=s,S=S,grad_matS=grad_matS,rescale=rescale,sd2=sd2)
            obj += tmp[0]; gobj += tmp[1]
            if np.isnan(reind[i,1]) == False:
                dat[int(reind[i,1])][0] = tmp[2:]
    #Return obj and gradobj
    if nparr:
        return np.append(np.array([obj]),gobj)
    else:
        return obj, gobj
def grad_obj_total(mlam,gamma,sl2,sm2,reind_v,\
                dat_v,s,S,grad_matS,dt,rescale,\
                   sd2,nproc=10):
    """Apply in parallel on all lane ID"""
    p = Pool(nproc)
    fun = lambda x:\
        grad_obj_1lane(x[0],x[1],mlam,gamma,sl2,sm2,S,s,dt,grad_matS,rescale,sd2,True,obj_and_grad_1cc)
    ret = p.map(fun,zip(reind_v,dat_v))
    ret = np.sum(np.vstack(ret),axis=0)
    return ret[0],ret[1:]
def grad_obj_wrap(x,in_dic):
    mlam,gamma,sl2,sm2,sd2 = x
    reind_v,dat_v,grad_matS,s,S,dt,lane_ID_v,val_v,rescale=\
    in_dic['reind_v'],in_dic['dat_v'],in_dic['grad_matS'],in_dic['s'],in_dic['S'],in_dic['dt'],in_dic['lane_ID_v'],in_dic['val_v'],in_dic['rescale']
    return grad_obj_total(mlam,gamma,sl2,sm2,reind_v,\
                          dat_v,s,S,grad_matS,dt,rescale,sd2,nproc=10)
def hessian_and_grad_tot(x,in_dic):
    H = np.zeros((4,4)); G=np.zeros((4,1))
    mlam,gamma,sl2,sm2 = x
    reind_v,dat_v,grad_matS,s,S,dt,lane_ID_v,val_v,rescale,sd2 = in_dic['reind_v'],in_dic['dat_v'],in_dic['grad_matS'],in_dic['s'],in_dic['S']\
                                                               ,in_dic['dt'],in_dic['lane_ID_v'],in_dic['val_v'],in_dic['rescale'],in_dic['sd2']
    for r,d in zip(reind_v,dat_v):
        Ht,Gt = grad_obj_1lane(r,d,mlam,gamma,sl2,sm2,S,s,dt,grad_matS,rescale,sd2,False,hessian_1cc)
        H+=Ht; G+=Gt
    return H,G
#-------------------PREDICTIONS OVER CC/LANE AND TOTAL-----------------------------------------
#-------------------PREDICTIONS-------------------------- 
def inverse(A): 
    """Inverse a 2x2 matrix"""
    assert A.shape==(2,2)
    return np.array([[A[1,1],-A[0,1]],[-A[1,0],A[0,0]]])\
            /(A[0,0]*A[1,1]-A[0,1]*A[1,0])
def print_mat(name,a,A):
    print(name,a[0,0],a[1,0],A[0,0],A[0,1],A[1,1])
def predictions_1cc(W,mlam,gamma,sl2,sm2,dt,s,S,rescale,sd2):
    """Return optiman length and growth (z) and std as erro """
    z = []; err_z=[]
    #### Initialize parameters for recurrence
    F, A, a = parameters(gamma,dt,mlam,sl2)
    ##### P(z_0|x_0^m)
    b,B = posteriori_matrices(W[0,0],s,S,sm2)
    #print_mat('n_prior',s,S)
    #print('n_measure',W[0,0],sm2,0,0)
    #print_mat('n_posterior',b,B)
    z.append(np.array(b)); err_z.append(np.sqrt(np.array([B[0,0],B[1,1]])))
    for j in range(1,W.shape[1]):
       ###### P(z_{t+dt}|D_t) = N(m,Q))
        m,Q = new_mean_cov(b,B,F,A,a)
        ##### P(z_{t+dt}|D_{t+dt}) = N(b',B')
        b,B = posteriori_matrices(W[0,j],m,Q,sm2)
        ##### Optimal predicitons 
        #InvB = inverse(B)
        #print_mat('prior',m,Q)
        #print('measure',W[0,j],sm2,0,0)
        #print_mat('posterior',b,B)
    	z.append(np.array(b)); err_z.append(np.sqrt(np.array([B[0,0],B[1,1]])))
    # Find next cell intial conditions
    m,Q = new_mean_cov(b,B,F,A,a)
    # Find next cell initial conditions
    s,S= cell_division_likelihood_and_grad(m,Q,None,sd2,rescale,grad=False)
    return z, err_z, s, S
def predictions_1lane(reind_,dat_,mlam,gamma,sl2,sm2,S,s,dt,lane_ID,val,rescale,sd2):
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
        predictions_1cc(W=dat[i][1][0],mlam=mlam,gamma=gamma,sl2=sl2,sm2=sm2,dt=dt,s=s,S=S,rescale=rescale,sd2=sd2)
        z = tmp[0]; err_z = tmp[1]
        ret.append([lane_ID+'_'+str(val[i,1]),z,err_z])
        #print(np.mean(np.hstack(z).T[:,0]-dat[i][1][0]))
    # give the inital condition to the right cell lane
        if np.isnan(reind[i,0]) == False:
            dat[int(reind[i,0])][0] = tmp[2:] # s,S,grad_S
    #If the second cell exists do the same
        if np.sum(np.isnan(dat[i][1][1]))==0:
            tmp =\
            predictions_1cc(W=dat[i][1][1],mlam=mlam,gamma=gamma,sl2=sl2,sm2=sm2,dt=dt,s=s,S=S,rescale=rescale,sd2=sd2)
            z = tmp[0]; err_z = tmp[1]
            ret.append([lane_ID+'_'+str(val[i,2]),z,err_z])
            #print(np.mean(np.hstack(z).T[:,0]-dat[i][1][1]))
            if np.isnan(reind[i,1]) == False:
                dat[int(reind[i,1])][0] = tmp[2:]
        #set_trace()
    #Return obj and gradobj
    return ret
#def prediction_total(mlam,gamma,sl2,sm2,reind_v,dat_v,s,S,dt,lane_ID_v,val_v,rescale,sd2,nproc=10):
#    """Apply in parallel on all lane ID and return a np.array"""
#    p = Pool(nproc)
#    fun = lambda x:\
#        predictions_1lane(x[0],x[1],mlam,gamma,sl2,sm2,S,s,dt,x[2],x[3],rescale,sd2)
#    tmp = p.map(fun,zip(reind_v,dat_v,lane_ID_v,val_v))
#    # Return a nice behave np array with cell_ID, z[0],z[1],err_z[0],err_z[1]
#    foo = []
#    for lan in tmp:
#        for cid in lan:
#            foo.append(np.hstack((np.hstack([cid[0]]*len(cid[1]))[:,None],np.hstack(cid[1]).T,np.vstack(cid[2]))))
#    return np.vstack(foo)
def prediction_total(mlam,gamma,sl2,sm2,reind_v,dat_v,s,S,dt,lane_ID_v,val_v,rescale,sd2,nproc=10):
    """Apply in parallel on all lane ID and return a np.array"""
    tmp = []
    for k in range(len(reind_v)):
        tmp.append(predictions_1lane(reind_=reind_v[k],dat_=dat_v[k],mlam=mlam,gamma=gamma,sl2=sl2,sm2=sm2,dt=dt,s=s,S=S,rescale=rescale,sd2=sd2,val=val_v[k],lane_ID=lane_ID_v[k]))
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
    prediction_total(mlam,gamma,sl2,sm2,reind_v,dat_v,s,S,dt,lane_ID_v,val_v,rescale,sd2)[:,1:2]
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
                            in_dic['val_v'],in_dic['rescale'],in_dic['sd2'])
################################################################################################
############################## DATA TREATEMENT #################################################
################################################################################################

def build_intial_mat(df,leng):
    """ Build the intiala matrix s,S and intial gradient grad_S """
    # Computation done in minutes (that's why *60)
    def sleres(t,y):
        """Return slope (lambda), intercept (x0) and residuals (form sm2)"""
        t = t - t.iloc[0]
        r = linregress(t,y)
        assert len(t)>2, "not enough points"
        return r.slope,r.intercept, y-(r.intercept+t*r.slope)
    G = df.groupby('cell').apply(lambda x: sleres(x.time_sec/60.,x['{}'.format(leng)]))
    G = np.vstack(G)
    lam = G[:,0]; x0 = G[:,1]; res = G[:,2]
    res = np.array([k for h in res for k in h])
    sm2 = np.var(res)
    ####### They do not depend on the parameters
    mat = np.zeros((2,2)); vec = np.zeros((2,1))
    #print("Wrong initial cond")
    #mat = np.ones((2,2)); vec = np.ones((2,1))
    grad_matS = {'m_mlam':vec,'m_gamma':vec,'m_sl2':vec,'m_sm2':vec,\
                 'Q_mlam':mat,'Q_gamma':mat,'Q_sl2':mat,'Q_sm2':mat,\
                 'Q_sd2':mat,'m_sd2':vec}
    for k in ['m','g','s','e']:
        grad_matS['m_{}'.format(k)]=vec
        grad_matS['Q_{}'.format(k)]=mat
        for i in ['m','g','s','e']:
            grad_matS['m_{}{}'.format(k,i)]=vec
            grad_matS['Q_{}{}'.format(k,i)]=mat
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
##
#def asym_dist_1lane(reind_,dat_,dt,rescale):
#    """Find the asymmetric distribution in log space for one lane"""
#    from copy import deepcopy
#    reind = deepcopy(reind_); dat = deepcopy(dat_)
#    distx0 = []; distlam = []; distk0 = []           # total objective and gradient
#    def pred_moth(i,j):
#        """Predict division length and growth rate. Do same for inital one"""
#        # Linear fit one cell cycle to estimate length mother and lenght daughter
#        W=dat[i][1][j].reshape(-1)
#        t = np.arange(0,dt*len(W),dt)
#        tmp = linregress(t,W)
#        tmp1 = linregress(t[:4],W[:4])
#        tmp2 = linregress(t[:4],W[-4:])
#        if np.isnan(reind[i,j]) == False:
#            # predict cell lenght at division and el_rat
#            foo = np.append(t,t[-1]+dt/2)*tmp.slope+tmp.intercept
#            dat[int(reind[i,j])][0] = {'ml':foo[-1],'mlam':tmp2.slope}
#        return tmp.intercept, tmp1.slope #x0 and lambda
#    ## APPLY
#    for i in range(len(dat)):
#        # If cell doesn't have mother just predict length of daugther and save them
#        if type(dat[i][0])!=dict:
#            pred_moth(i,0);
#            if np.sum(np.isnan(dat[i][1][1]))==0:
#                pred_moth(i,1)
#        # If it does has a mother predict its length and save the log  difference betwee half of mother cell and daugther one
#        else:
#            x0,lam = pred_moth(i,0)
#            distx0.append(dat[i][0]['ml']-rescale*np.log(2)-x0)
#            distlam.append(dat[i][0]['mlam']-lam)
#            distk0.append([x0,lam])
#            if np.sum(np.isnan(dat[i][1][1]))==0:
#                x0,lam = pred_moth(i,1)
#                distx0.append(dat[i][0]['ml']-rescale*np.log(2)-x0)
#                distlam.append(dat[i][0]['mlam']-lam)
#                distk0.append([x0,lam])
#    return distx0, distlam, distk0
#def asym_dist_(reind_v,dat_v,dt,rescale):
#    """Return distribution of difference between predictd half size and actual cell division (distx0); differene in growth rates between mother and daugheter (distlam); and initial condition (x,lam) distk0 """
#    distx0 = []; distlam = []; distk0 =[] 
#    for i,j in enumerate(dat_v):
#        dx0 , dlam, dk0 = asym_dist_1lane(reind_v[i],dat_v[i],dt,rescale)
#        distx0.append(dx0); distlam.append(dlam); distk0.append(dk0)
#    flat = lambda dist: np.array([j for k in dist for j in k]) 
#    return flat(distx0),flat(distlam), flat(distk0)
##
def asym_dist(_,dat_v,__,rescale):
    """Predict the amount of asymmetric division by considering the 2 daughter must sum up to the total"""
    from scipy.stats import linregress
    lg = lambda x: linregress(range(len(x.reshape(-1))),x.reshape(-1)).intercept
    flat = lambda x: np.array([g for m in x for g in m ])
    sxd2 =[]; sgd2 = []
    for j in dat_v:
        for k in j:
            if type(k[1][1])==float and (np.isnan(k[1][1])): 
                continue
            else:
                # Predict the length and gfp at begin
                x1 = k[1][0]
                x2 = k[1][1]
                x1,x2 = [lg(x) for x in [x1,x2]]
                perfx = np.log((np.exp(x1)+np.exp(x2)))-rescale*np.log(2) # perfect division
                sxd2.append((perfx-x1,perfx-x2))
    return flat(sxd2),_,_

def build_data_strucutre(df,leng,rescale,info=False):
    """Return for every lane the data with respective daughteres and initial conditions"""
    #Sometimes cells with 1 data point are present and we don't want them
    df = df.groupby('cell').filter(lambda x: x.values.shape[0]>1) #
    #dt = np.diff(np.sort(df['time_sec'].unique()))[1]/60
    dt = np.hstack(df.groupby('cell')['time_sec'].apply(lambda x: np.diff(x)/60).values)
    assert np.sum(dt!=dt[0])==0
    dt = dt[0]
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
    sd2, _, _ = asym_dist(reind_v,dat_v,dt,rescale)
    #asym division equal to 0.1 cv(0.1*rescale*np.log(2))**2
    return df,{'n_point':n_point,'dt':dt,'s':s,'S':S,'grad_matS':grad_matS,\
            'reind_v':reind_v,'dat_v':dat_v, 'val_v':val_v,\
               'lane_ID_v':lane_ID_v,'rescale':rescale,'sm2':sm,\
               'vec_dat_v':vec_dat_v,'sd2':np.var(sd2)}
#
def merge_df_pred(df,pred_mat):
    """Merge the output from predict with the initial dataframe"""
    # From mat to dataframe
    dft = pd.DataFrame(pred_mat,columns=\
                       ('cell_','pred_log_length','pred_growth_rate',\
                        'err_log_length','err_growth_rate'))
    # Give numerical values
    dft[['pred_log_length','pred_growth_rate','err_log_length','err_growth_rate']] = \
            dft[['pred_log_length','pred_growth_rate','err_log_length','err_growth_rate']].apply(pd.to_numeric)
    #Create subindex for merging
    dft['sub_ind'] = dft.groupby('cell_')['pred_log_length'].transform(lambda x:\
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
def denoised_dataset(df,step,nump=12):
    """Try to obtain a dataset without noise by sampling every <<step>> """
    # At least nump cell per cell
    df = df.groupby('cell').filter(lambda x: True if len(x['time_sec'])>=nump else False)
    df = df.reset_index()
    ret = []
    sor = np.sort(df['time_sec'].unique())
    for tau in range(step):
        tmp = sor[tau:][::step]
        dfsh = pd.concat([df.loc[df['time_sec']==tmp[k]] for k in range(tmp.shape[0])])
        dfsh = dfsh.sort_index().reset_index()
        ret.append(dfsh)
    return ret
################################################################################################
############################### ADDITIONAL FUNCTIONS FOR ANALYSIS  #############################
################################################################################################
def give_good_structure(df):
    """Create cell id lane id """
    df['date'] = df['date'].apply(lambda x:'c'+str(x))
    df['cell'] = df['date']+df['pos'].apply(lambda x: str(x))+df['gl'].apply(lambda x: str(x))+df['id'].apply(lambda x: str(x))
    df['lane_ID'] = df['date']+df['pos'].apply(lambda x: str(x))+df['gl'].apply(lambda x: str(x))
    return df
def genalogy(df,gen_deg='parent_cell'):
    """find one geanolgy upper"""
    if 'parent_cell' not in df.columns:
        df['parent_cell'] = df['lane_ID']+df['parent_id'].apply(lambda x:str(x))
    for k in df.cell.unique():
        daug = df.loc[df['cell']==k] #daug
        gmum = df.loc[df['cell']==daug['{}'.format(gen_deg)].iloc[0]]['parent_cell'] #grand grand mum
        try:
            df.loc[df['cell']==k,'g_'+'{}'.format(gen_deg)] = gmum.iloc[0]
        except IndexError:
            df.loc[df['cell']==k,'g_'+'{}'.format(gen_deg)] = 'nan'
    return df
def corr_par(df,elrat,par_deg='parent_cell'):
    """find correlation parameters, return corrcoef and number of points  """
    ret=[]
    for k in df.cell.unique():
        try:
            mum = elrat.loc[k]
            daug = elrat.loc[df.loc[df['cell']==k]['{}'.format(par_deg)].iloc[0]]
            ret.append([mum,daug])
        except KeyError:
            continue
    tmp = np.array(ret)
    return np.corrcoef(tmp[:,:1].reshape(-1),tmp[:,1:].reshape(-1))[0,1], tmp.shape[0]
def long_corr_in_data(df,leng,ngen):
    """Find corr over generations in dataframes"""
    pre=['']
    if 'g_'*(ngen-1)+'parent_cell' not in df.columns:
        for k in range(ngen-1):
            df = genalogy(df,'{}parent_cell'.format(pre[-1]))
            pre.append(pre[-1]+'g_')
    else:
        for k in range(ngen-1):
            pre.append(pre[-1]+'g_')
    elrat = df.groupby('cell').apply(lambda x:linregress(x['time_sec'],np.log(x['{}'.format(leng)])).slope)
    pre = [k+'parent_cell' for k in pre]
    return np.vstack([corr_par(df,elrat,par_deg=k) for k in pre]),elrat.std()/elrat.mean()
#def smilar_frame(W,cvd=0):
#    """From  OU create a dataframe with same shape as biological data (divison at twice the size).. Consider also asymmetric division """
#    explen = []; lane_ID=[]; parent_ID=[]; id_n=[-1]; time_sec=[]; df=[]
#    W = deepcopy(W); 
#    for i in range(W.shape[0]):
#        tmp = [W[i,0]]
#        fix = W[i,0]
#        ts=[0]
#        for k in range(1,W.shape[1]-1):
#            #if tmp[-1] < tmp[0]+np.log(2):
#            #Do the choice stocastically otherwise we accumulate always some lengths
#            #print([W[i,k]< tmp[0]+np.log(2),W[i,k+1]< tmp[0]+np.log(2)])
#            div = np.random.normal(np.log(2),np.log(2)*cvd)
#            if np.random.choice([W[i,k]< fix+div,W[i,k+1]< fix+div],1,p=[0.5,0.5])[0]:
#                tmp.append(W[i,k])
#                ts.append(ts[-1]+180)
#            else:
#                lane_ID = ['lane_num_{}'.format(i)]*int(len(tmp))
#                parent_ID = ['{}'.format(id_n[-1])]*int(len(tmp))
#                id_n = ['{}'.format(k)]*int(len(tmp))
#                explen= np.exp(tmp)
#                time_sec = ts
#                ts=[ts[-1]]         
#                tmp = [tmp[-1]-div]
#                W[i,:] = W[i,:]-div
#                #print(len(explen))
#                #print(len(lane_ID))
#                df.append(pd.DataFrame({'leng':explen,'lane_ID':lane_ID,'parent_id':parent_ID, 'id':id_n,'time_sec':time_sec}))
#    df = pd.concat(df,ignore_index=True)
#    df['cell'] = df['lane_ID']+'_'+df['id']
#    return df
def syntetic_corr_gen(mlam,gamma,sl2,sm2,dt,cpl=40, lenc = 25, ncel = 20):
    def sleres(y,dt=dt):
        """Return slope (lambda), intercept (x0) and residuals (form sm2)"""
        t = np.arange(0,dt*len(y),dt) 
        r = linregress(t,y)
        return r.slope
    W,X = integrated_ou(mlam,gamma,sl2,sm2=0,dt=dt,length=cpl*lenc,ncel=ncel)
    dfsy = smilar_frame(W,cvd=0.)
    dfsy['leng'] = dfsy['leng'].apply(lambda x: np.exp(np.random.normal(np.log(x),np.sqrt(sm2))))
    dfsy,in_dic_sy = build_data_strucutre(dfsy,'leng',1)
    dfsy['parent_cell'] = dfsy['lane_ID']+'_'+dfsy['parent_id']
    dfsy = genalogy(dfsy,'parent_cell')
    dfsy = genalogy(dfsy,'g_parent_cell')
    dfsy = genalogy(dfsy,'g_g_parent_cell')
    dfsy = genalogy(dfsy,'g_g_g_parent_cell')
    elratsy = dfsy.groupby('cell').apply(lambda x: sleres(x.log_resc_leng))
    corr_long = np.vstack([corr_par(dfsy,elratsy,par_deg=k) for k in ['parent_cell','g_parent_cell','g_g_parent_cell','g_g_g_parent_cell','g_g_g_g_parent_cell'] ])
    return corr_long
def give_unique_dataset(df,step,nump=3):
    """Denoise the dataset and rebuild step independent out of it! """
    d3glu = denoised_dataset(df,step,nump)
    tmp = []
    k=0 
    for dtm in d3glu:
        dtm = dtm.drop(['cell','lane_ID'],axis=1)
        dtm['date']=dtm['date'].apply(lambda x:str(x))+'_{}_'.format(k)
        dtm['cell'] = dtm['date'].apply(lambda x:str(x))+dtm['pos'].apply(lambda x: str(x))+dtm['gl'].apply(lambda x: str(x))+dtm['id'].apply(lambda x: str(x))
        dtm['lane_ID'] = dtm['date'].apply(lambda x:str(x))+dtm['pos'].apply(lambda x: str(x))+dtm['gl'].apply(lambda x: str(x))
        tmp.append(dtm)
        k+=1
    return pd.concat(tmp,ignore_index=True)
###############################################################################################
############################### The stocastics models  #########################################
################################################################################################
def ornstein_uhlenbeck(mlam,gamma,sl2,shape=(1,1000),dtsim=1):
    "Create ou model of shape (#ncel,#length_cell) with sampling dtsim"
    mat = np.zeros(shape)
    sig = np.sqrt(sl2)
    dW = np.random.normal(loc=mat,scale=np.sqrt(dtsim))
    add = sig*dW*dtsim
    mat[:,0]=add[:,0]+mlam
    for k in range(1,shape[1]):
        mat[:,k]=mat[:,k-1]-gamma*(mat[:,k-1]-mlam)*dtsim+add[:,k]
    return mat

def adder(V0,W,DV,prin=False,dt_sim=1):
    """V0 initial volume DV added volume (equal to V0 in stationary) W OU process"""
    #assert W.shape[0]==1
    W = W.reshape(-1)
    x0=V0
    integ = []
    j=0
    gr=[]
    last=W[0]
    while True:
        W = np.insert(W,0,0)
        try:
            #Find first index which fulfill condition
            exp = x0*np.exp(np.cumsum(W*dt_sim))
            ind = np.where(exp-x0<DV)[0][-1]+1
            if W.shape[0]==1:break
        except IndexError:
            break
        #Randomly choose if take first smaller or first larger which statistfy cond
        ind += np.random.choice([0,1])
        integ.append(exp[:ind])
        te = W[:ind]; te[0]=last
        gr.append(te)
        try: 
            last = W[ind-1]
        except IndexError:
            break
        W= W[ind:]
        x0=integ[-1][-1]/2
    #return integ
    #to_print = np.array([k[-1] for k in integ])
    #if prin: print('added length:',np.mean(to_print),'+/-',np.std(to_print)/np.sqrt(len(to_print)))
    return integ[:-1],gr[:-1]
def same_adder_shape(f,X):
    """Give to f numpy array the same shape as X list of arrays"""
    assert f.ndim==1
    ind  = np.append(np.array([0]),np.cumsum([x.shape[0] for x in X]))
    Y = []
    for i in range(len(ind)-1):
        Y.append(f[ind[i]:ind[i+1]])
    return Y[:-1],X[:-1]
def gfp_dynamics_1cc_(F,V,Cm1,beta,dtsim):
    """Knowing the concentration before division Cm1, mRNA dynamics F and Volume dynamics V compute GFP dynamics. Condition is concentration stay constant at division"""
    G = np.zeros_like(F)
    G[0] = Cm1*V[0]
    for k in range(len(F)-1):
        G[k+1] = G[k]+(V[k]*F[k]-beta*G[k])*dtsim
    return G,G[-1]/V[-1]
def gfp_dynamics_1cc(F,V,Cm1,beta,dtsim):
    """Divide GFP by 2 as in my model"""
    G = np.zeros_like(F)
    G[0] = Cm1/2
    for k in range(len(F)-1):
        G[k+1] = G[k]+(V[k]*F[k]-beta*G[k])*dtsim
    return G,G[-1]

def gfp_dynamics(F,V,Cm1,beta,dtsim):
    """Apply dynamics on all cells"""
    G = []
    n=0
    for f,v in zip(F,V):
        if n==0: cm1 = Cm1
        g,cm1 = gfp_dynamics_1cc(f,v,cm1,beta,dtsim)
        G.append(g)
        n+=1
    return G
def the_pred_values(ml,gl,sl2,mq,gq,sq2,DV,beta):
    """Easy values to comptue theoretically"""
    print("Corr time in elongation rate = {} and in mRNA = {}".format(1/gl,1/gq))
    print("Mean volume = {} and mean GFP = {}".format(3/2*DV,3/2*DV*mq/ml))
    cvs = lambda x,y,z: np.sqrt(x/(2*y*z**2))
    print("Cv in elongation rate {} and mRNA {}".format(cvs(sl2,gl,ml),cvs(sq2,gq,mq)))
def\
similar_frame(ml,gl,sl2,sm2,mq,gq,sq2,DV,beta,shape=(1,1000),dtsim=1,gfp_sym=False,gerr=None,C0=None):
    """From  OU create a dataframe with same shape as biological data (divison at twice the size).. Consider also asymmetric division """
    if shape[0]>10: print('change naming of indexing cells since they may mix')
    explen = []; lane_ID=[]; parent_ID=[]; id_n=[-1]; time_sec=[]; df=[]
    W = ornstein_uhlenbeck(ml,gl,sl2,shape,dtsim) # create elongation rate dynamics
    if gfp_sym :F = ornstein_uhlenbeck(mq,gq,sq2,shape,dtsim) # create elongation rate dynamics
    V = np.zeros_like(W)
    V0=DV
    if C0==None:#if we want to use other division properites
        C0=3/2*DV*mq/ml
    V[:,0] = V0 
    for i in range(W.shape[0]):
        X,Ww = adder(V0,W[i,:],DV,False,dtsim)#Adder model for size
        if gfp_sym:
            f,X = same_adder_shape(F[i,:],X)
            Ww = Ww[:-1]
            G = gfp_dynamics(f,X,C0,beta,dtsim)
        else:
            X = X[:-2]; Ww=Ww[:-2]
            assert(sum([len(x)-len(w) for x,w in zip(X,Ww)])==0)
        for j,k in enumerate(X):
            #if j<5: continue # discard first cells for stationarity
            lane_ID = ['{}'.format(i)]*int(len(k))
            parent_ID = ['{}'.format(id_n[-1])]*int(len(k))
            id_n = ['{}'.format(float(j))]*int(len(k))
            explen= np.exp(np.random.normal(np.log(k),np.sqrt(sm2)))
            if gfp_sym:
                if gerr==None:
                    gfp = np.random.normal(G[j],np.sqrt(G[j]))
                else:
                    gfp = np.random.normal(G[j],np.sqrt(gerr))
                if len(gfp)!=len(explen):
                    continue
                df.append(pd.DataFrame({'leng':explen,'lane_ID':lane_ID,'parent_id':parent_ID,\
                                        'id':id_n,'gfp':gfp,'gfp_no_noise':G[j],'leng_no_noise':k,'growth_rate':Ww[j],'q_dyn':f[j]}))
            else:
                #print("ok")
#                print(len(explen)-len(Ww[j]))
                df.append(pd.DataFrame({'leng':explen,'lane_ID':lane_ID,'parent_id':parent_ID,\
                                        'id':id_n,'leng_no_noise':k,'growth_rate':Ww[j]}))
    df = pd.concat(df,ignore_index=True)
    df['time_sec'] = np.arange(0,df.shape[0]*dtsim,dtsim)*60
    df['cell'] = df['lane_ID']+df['id']
    return df
#def ornstein_uhlenbeck(mlam,gamma,sl2,length=30,ncel=10,dt=3.,dtsim=1):
#    sam = dt/dtsim
#    lengthsim = length*sam
#    assert (sam).is_integer(), "not sam integer"
#    assert (lengthsim).is_integer(), "no lan integer"
#    sam = int(sam); lengthsim=int(lengthsim)
#    mat = np.zeros((ncel,lengthsim))
#    sig = np.sqrt(sl2)
#    dW = np.random.normal(loc=mat,scale=np.sqrt(dtsim))
#    add = sig*dW*dtsim
#    mat[:,0]=add[:,0]+mlam
#    for k in range(1,lengthsim):
#        mat[:,k]=mat[:,k-1]-gamma*(mat[:,k-1]-mlam)*dtsim+add[:,k]
#    return mat[:,::sam]
#def integrated_ou(mlam,gamma,sl2,sm2,X0=1,sx0=0.1,length=30,ncel=10,dt=3.,dtsim=1):
#    X = ornstein_uhlenbeck(mlam,gamma,sl2,length,ncel,dt,dtsim)
#    X0 = np.random.normal(loc=np.ones((ncel,1)),scale=sx0)
#    return np.random.normal(loc=np.hstack([X0,np.cumsum(X,axis=1)*dt+X0]),scale=np.sqrt(sm2))[:,:-1], X
#def W_er(st): 
#    diffW= abs(W-z[0,:])
#    percW = sum(sum(diffW>st*err_z[0,:]))/diffW.shape[1]
#    return percW
#def X_er(st):
#    diffX= abs(X-z[1,:])
#    percX = sum(sum(diffX>st*err_z[1,:]))/diffX.shape[1]
#    return percX
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
    print(X_er(1),X_er(2))
