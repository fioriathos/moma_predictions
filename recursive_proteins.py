import numpy as np
from numba import jit,prange
from copy import deepcopy
import pandas as pd
import time
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.stats import linregress
import scipy.integrate as integrate
######
###### Inverse and |determinatn| of 4x4 symmetric matrix
######
def inverse_Q_00(Q00,Q01,Q02,Q03,Q11,Q12,Q13,Q22,Q23,Q33):
     return   (Q13**2*Q22 - 2*Q12*Q13*Q23 + Q12**2*Q33 + Q11*(Q23**2 - Q22*Q33))/\
       (Q00*Q13**2*Q22 + Q03**2*(-Q12**2 + Q11*Q22) - 2*Q00*Q12*Q13*Q23 - Q01**2*Q23**2 + Q00*Q11*Q23**2 + \
         2*Q03*(Q02*Q12*Q13 - Q01*Q13*Q22 - Q02*Q11*Q23 + Q01*Q12*Q23) + Q00*Q12**2*Q33 + Q01**2*Q22*Q33 - Q00*Q11*Q22*Q33 + \
         Q02**2*(-Q13**2 + Q11*Q33) + 2*Q01*Q02*(Q13*Q23 - Q12*Q33))
def inverse_Q_01(Q00,Q01,Q02,Q03,Q11,Q12,Q13,Q22,Q23,Q33):
     return   (-(Q03*Q13*Q22) + Q03*Q12*Q23 + Q02*Q13*Q23 - Q01*Q23**2 - Q02*Q12*Q33 + Q01*Q22*Q33)/\
       (Q00*Q13**2*Q22 + Q03**2*(-Q12**2 + Q11*Q22) - 2*Q00*Q12*Q13*Q23 - Q01**2*Q23**2 + Q00*Q11*Q23**2 + \
         2*Q03*(Q02*Q12*Q13 - Q01*Q13*Q22 - Q02*Q11*Q23 + Q01*Q12*Q23) + Q00*Q12**2*Q33 + Q01**2*Q22*Q33 - Q00*Q11*Q22*Q33 + \
         Q02**2*(-Q13**2 + Q11*Q33) + 2*Q01*Q02*(Q13*Q23 - Q12*Q33))
def inverse_Q_02(Q00,Q01,Q02,Q03,Q11,Q12,Q13,Q22,Q23,Q33):
     return   (-(Q03*Q12*Q13) + Q02*Q13**2 + Q03*Q11*Q23 - Q01*Q13*Q23 - Q02*Q11*Q33 + Q01*Q12*Q33)/\
       (-(Q00*Q13**2*Q22) + Q03**2*(Q12**2 - Q11*Q22) + 2*Q00*Q12*Q13*Q23 + Q01**2*Q23**2 - Q00*Q11*Q23**2 - \
         2*Q03*(Q02*Q12*Q13 - Q01*Q13*Q22 - Q02*Q11*Q23 + Q01*Q12*Q23) - Q00*Q12**2*Q33 - Q01**2*Q22*Q33 + Q00*Q11*Q22*Q33 + \
         Q02**2*(Q13**2 - Q11*Q33) + Q01*Q02*(-2*Q13*Q23 + 2*Q12*Q33))
def inverse_Q_03(Q00,Q01,Q02,Q03,Q11,Q12,Q13,Q22,Q23,Q33):
     return   (-(Q02*Q12*Q13) + Q01*Q13*Q22 + Q03*(Q12**2 - Q11*Q22) + Q02*Q11*Q23 - Q01*Q12*Q23)/\
       (-(Q00*Q13**2*Q22) + Q03**2*(Q12**2 - Q11*Q22) + 2*Q00*Q12*Q13*Q23 + Q01**2*Q23**2 - Q00*Q11*Q23**2 - \
         2*Q03*(Q02*Q12*Q13 - Q01*Q13*Q22 - Q02*Q11*Q23 + Q01*Q12*Q23) - Q00*Q12**2*Q33 - Q01**2*Q22*Q33 + Q00*Q11*Q22*Q33 + \
         Q02**2*(Q13**2 - Q11*Q33) + Q01*Q02*(-2*Q13*Q23 + 2*Q12*Q33))
def inverse_Q_11(Q00,Q01,Q02,Q03,Q11,Q12,Q13,Q22,Q23,Q33):
     return   (Q03**2*Q22 - 2*Q02*Q03*Q23 + Q02**2*Q33 + Q00*(Q23**2 - Q22*Q33))/\
       (Q00*Q13**2*Q22 + Q03**2*(-Q12**2 + Q11*Q22) - 2*Q00*Q12*Q13*Q23 - Q01**2*Q23**2 + Q00*Q11*Q23**2 + \
         2*Q03*(Q02*Q12*Q13 - Q01*Q13*Q22 - Q02*Q11*Q23 + Q01*Q12*Q23) + Q00*Q12**2*Q33 + Q01**2*Q22*Q33 - Q00*Q11*Q22*Q33 + \
         Q02**2*(-Q13**2 + Q11*Q33) + 2*Q01*Q02*(Q13*Q23 - Q12*Q33))
def inverse_Q_12(Q00,Q01,Q02,Q03,Q11,Q12,Q13,Q22,Q23,Q33):
     return   (Q03**2*Q12 + Q00*Q13*Q23 - Q03*(Q02*Q13 + Q01*Q23) + Q01*Q02*Q33 - Q00*Q12*Q33)/\
       (-(Q00*Q13**2*Q22) + Q03**2*(Q12**2 - Q11*Q22) + 2*Q00*Q12*Q13*Q23 + Q01**2*Q23**2 - Q00*Q11*Q23**2 - \
         2*Q03*(Q02*Q12*Q13 - Q01*Q13*Q22 - Q02*Q11*Q23 + Q01*Q12*Q23) - Q00*Q12**2*Q33 - Q01**2*Q22*Q33 + Q00*Q11*Q22*Q33 + \
         Q02**2*(Q13**2 - Q11*Q33) + Q01*Q02*(-2*Q13*Q23 + 2*Q12*Q33))
def inverse_Q_13(Q00,Q01,Q02,Q03,Q11,Q12,Q13,Q22,Q23,Q33):
     return   (Q02**2*Q13 + Q01*Q03*Q22 - Q00*Q13*Q22 + Q00*Q12*Q23 - Q02*(Q03*Q12 + Q01*Q23))/\
       (-(Q00*Q13**2*Q22) + Q03**2*(Q12**2 - Q11*Q22) + 2*Q00*Q12*Q13*Q23 + Q01**2*Q23**2 - Q00*Q11*Q23**2 - \
         2*Q03*(Q02*Q12*Q13 - Q01*Q13*Q22 - Q02*Q11*Q23 + Q01*Q12*Q23) - Q00*Q12**2*Q33 - Q01**2*Q22*Q33 + Q00*Q11*Q22*Q33 + \
         Q02**2*(Q13**2 - Q11*Q33) + Q01*Q02*(-2*Q13*Q23 + 2*Q12*Q33))
def inverse_Q_22(Q00,Q01,Q02,Q03,Q11,Q12,Q13,Q22,Q23,Q33):
     return   (Q03**2*Q11 - 2*Q01*Q03*Q13 + Q01**2*Q33 + Q00*(Q13**2 - Q11*Q33))/\
       (Q00*Q13**2*Q22 + Q03**2*(-Q12**2 + Q11*Q22) - 2*Q00*Q12*Q13*Q23 - Q01**2*Q23**2 + Q00*Q11*Q23**2 + \
         2*Q03*(Q02*Q12*Q13 - Q01*Q13*Q22 - Q02*Q11*Q23 + Q01*Q12*Q23) + Q00*Q12**2*Q33 + Q01**2*Q22*Q33 - Q00*Q11*Q22*Q33 + \
         Q02**2*(-Q13**2 + Q11*Q33) + 2*Q01*Q02*(Q13*Q23 - Q12*Q33))
def inverse_Q_23(Q00,Q01,Q02,Q03,Q11,Q12,Q13,Q22,Q23,Q33):
     return   (-(Q02*Q03*Q11) + Q01*Q03*Q12 + Q01*Q02*Q13 - Q00*Q12*Q13 - Q01**2*Q23 + Q00*Q11*Q23)/\
       (Q00*Q13**2*Q22 + Q03**2*(-Q12**2 + Q11*Q22) - 2*Q00*Q12*Q13*Q23 - Q01**2*Q23**2 + Q00*Q11*Q23**2 + \
         2*Q03*(Q02*Q12*Q13 - Q01*Q13*Q22 - Q02*Q11*Q23 + Q01*Q12*Q23) + Q00*Q12**2*Q33 + Q01**2*Q22*Q33 - Q00*Q11*Q22*Q33 + \
         Q02**2*(-Q13**2 + Q11*Q33) + 2*Q01*Q02*(Q13*Q23 - Q12*Q33))
def inverse_Q_33(Q00,Q01,Q02,Q03,Q11,Q12,Q13,Q22,Q23,Q33):
     return   (Q02**2*Q11 - 2*Q01*Q02*Q12 + Q01**2*Q22 + Q00*(Q12**2 - Q11*Q22))/\
       (Q00*Q13**2*Q22 + Q03**2*(-Q12**2 + Q11*Q22) - 2*Q00*Q12*Q13*Q23 - Q01**2*Q23**2 + Q00*Q11*Q23**2 + \
         2*Q03*(Q02*Q12*Q13 - Q01*Q13*Q22 - Q02*Q11*Q23 + Q01*Q12*Q23) + Q00*Q12**2*Q33 + Q01**2*Q22*Q33 - Q00*Q11*Q22*Q33 + \
         Q02**2*(-Q13**2 + Q11*Q33) + 2*Q01*Q02*(Q13*Q23 - Q12*Q33))
def inverseQ(Q00,Q01,Q02,Q03,Q11,Q12,Q13,Q22,Q23,Q33):
    """The inverse of a 4x4 sym matrix is again sym"""
    IQ00 = inverse_Q_00(Q00,Q01,Q02,Q03,Q11,Q12,Q13,Q22,Q23,Q33)
    IQ01 = inverse_Q_01(Q00,Q01,Q02,Q03,Q11,Q12,Q13,Q22,Q23,Q33)
    IQ02 = inverse_Q_02(Q00,Q01,Q02,Q03,Q11,Q12,Q13,Q22,Q23,Q33)
    IQ03 = inverse_Q_03(Q00,Q01,Q02,Q03,Q11,Q12,Q13,Q22,Q23,Q33)
    IQ11 = inverse_Q_11(Q00,Q01,Q02,Q03,Q11,Q12,Q13,Q22,Q23,Q33)
    IQ12 = inverse_Q_12(Q00,Q01,Q02,Q03,Q11,Q12,Q13,Q22,Q23,Q33)
    IQ13 = inverse_Q_13(Q00,Q01,Q02,Q03,Q11,Q12,Q13,Q22,Q23,Q33)
    IQ22 = inverse_Q_22(Q00,Q01,Q02,Q03,Q11,Q12,Q13,Q22,Q23,Q33)
    IQ23 = inverse_Q_23(Q00,Q01,Q02,Q03,Q11,Q12,Q13,Q22,Q23,Q33)
    IQ33 = inverse_Q_33(Q00,Q01,Q02,Q03,Q11,Q12,Q13,Q22,Q23,Q33)
    return IQ00,IQ01,IQ02,IQ03,IQ11,IQ12,IQ13,IQ22,IQ23,IQ33
def det(Q00,Q01,Q02,Q03,Q11,Q12,Q13,Q22,Q23,Q33):
     return  np.abs(Q03**2*Q12**2 - 2*Q02*Q03*Q12*Q13 + Q02**2*Q13**2 - Q03**2*Q11*Q22 + 2*Q01*Q03*Q13*Q22 - Q00*Q13**2*Q22 +\
       2*Q02*Q03*Q11*Q23 - 2*Q01*Q03*Q12*Q23 - 2*Q01*Q02*Q13*Q23 + 2*Q00*Q12*Q13*Q23 + Q01**2*Q23**2 - Q00*Q11*Q23**2 -\
       Q02**2*Q11*Q33 + 2*Q01*Q02*Q12*Q33 - Q00*Q12**2*Q33 - Q01**2*Q22*Q33 + Q00*Q11*Q22*Q33)
########################################################################################
#############################INFERENCE METHODS##########################################
########################################################################################
#------------------ UPDATE 1 POINT OBJECTIVE-----------------------------------
#########
######### THE NEW MEAN COVARIANCE P(Z_{t+dt}|Dt)~N(nm,Q) assuming we know P(Zt|Dt)~N(m,C)
#########
# We could also write them in matrix form so that a part stay fixed (not to be continuously computed) and then use matrix multiplication (as we did for lengths)
#mean
def meangt(m0,m1,m2,m3,C00,C01,C02,C03,C11,C12,C13,C22,C23,C33,ml,gl,sl2,mq,gq,sq2,dt,b):
    """Mean gt computed with approximations"""
    eglt = np.exp(-gl*dt)
    f = (1-eglt)/(gl*dt)
    le = f*m1 + (1 - f)*ml + ((-3 - eglt**2 + 4*eglt + 2*dt*gl)*sl2)/(2.*gl**3)
    dle = f
    F = (np.exp(m0)*(-np.exp(-(b*dt)) + np.exp(dt*(-gq + le)))*(m3 - mq))/(b - gq + le) + (np.exp(m0)*(-np.exp(-(b*dt)) + np.exp(dt*le))*mq)/(b + le)
    G = (np.exp(m0)*(-np.exp(-(b*dt)) + np.exp(dt*(-gq + le))))/(b - gq + le)
    d2Fx = F
    d2Fl = (2*dle**2*np.exp(m0)*(-np.exp(-(b*dt)) + np.exp(dt*(-gq + le)))*(m3 - mq))/(b - gq + le)**3 -\
       (2*dle**2*dt*np.exp(dt*(-gq + le) + m0)*(m3 - mq))/(b - gq + le)**2 +\
       (dle**2*dt**2*np.exp(dt*(-gq + le) + m0)*(m3 - mq))/(b - gq + le) +\
       (2*dle**2*np.exp(m0)*(-np.exp(-(b*dt)) + np.exp(dt*le))*mq)/(b + le)**3 - (2*dle**2*dt*np.exp(dt*le + m0)*mq)/(b + le)**2 +\
       (dle**2*dt**2*np.exp(dt*le + m0)*mq)/(b + le)
    d2Fxl = -((dle*np.exp(m0)*(-np.exp(-(b*dt)) + np.exp(dt*(-gq + le)))*(m3 - mq))/(b - gq + le)**2) +\
       (dle*dt*np.exp(dt*(-gq + le) + m0)*(m3 - mq))/(b - gq + le) - (dle*np.exp(m0)*(-np.exp(-(b*dt)) + np.exp(dt*le))*mq)/(b + le)**2 +\
       (dle*dt*np.exp(dt*le + m0)*mq)/(b + le)
    dGl = -((dle*np.exp(m0)*(-np.exp(-(b*dt)) + np.exp(dt*(-gq + le))))/(b - gq + le)**2) + (dle*dt*np.exp(dt*(-gq + le) + m0))/(b - gq + le)
    dGx = (np.exp(m0)*(-np.exp(-(b*dt)) + np.exp(dt*(-gq + le))))/(b - gq + le)
    return m2*np.exp(-b*dt)+ F + d2Fx*C00/2+d2Fl*C11/2+d2Fxl*C01+dGl*C13+dGx*C03

#def meangt(m0,m1,m2,m3,C00,C01,C02,C03,C11,C12,C13,C22,C23,C33,ml,gl,sl2,mq,gq,sq2,dt,b):
#    return np.exp(m0)*m3*dt+(1-b*dt)*m2
def new_mean(m0,m1,m2,m3,C00,C01,C02,C03,C11,C12,C13,C22,C23,C33,ml,gl,sl2,mq,gq,sq2,dt,b):
    """Start from P(z_t|D_t)= N(m;C) and find P(z_{t+dt}|D_t), the mean here"""
    #Seems ok
    nm0 = m0+ml*dt+(m1-ml)*(1-np.exp(-gl*dt))/gl
    nm1 = ml+(m1-ml)*np.exp(-gl*dt)
    nm3 = mq + (m3-mq)*np.exp(-gq*dt)
    nm2 = meangt(m0,m1,m2,m3,C00,C01,C02,C03,C11,C12,C13,C22,C23,C33,ml,gl,sl2,mq,gq,sq2,dt,b)
    return nm0,nm1,nm2,nm3
#Covariance
def cov23(m0,m1,m2,m3,C00,C01,C02,C03,C11,C12,C13,C22,C23,C33,ml,gl,sl2,mq,gq,sq2,dt,b):
    eglt = np.exp(-gl*dt)
    f = (1-eglt)/(gl*dt)
    le = f*m1 + (1 - f)*ml + ((-3 - eglt**2 + 4*eglt + 2*dt*gl)*sl2)/(2.*gl**3)
    dle = f; t=dt
    cvz0 = (np.exp(m0 - b*t - gq*t)*(-((-1 + np.exp((b - gq + le)*t))/(b - gq + le)) + (-1 + np.exp((b + gq + le)*t))/(b + gq + le))*sq2)/(2.*gq)
    dcv_l0 =  (np.exp(m0 - b*t - gq*t)*sq2*((dle*(-1 + np.exp(t*(b - gq + le))))/(b - gq + le)**2 -\
              (dle*np.exp(t*(b - gq + le))*t)/(b - gq + le) - (dle*(-1 + np.exp(t*(b + gq + le))))/(b + gq + le)**2 +\
              (dle*np.exp(t*(b + gq + le))*t)/(b + gq + le)))/(2.*gq)
    ddcv_l0 = (np.exp(m0 - b*t - gq*t)*sq2*((-2*dle**2*(-1 + np.exp(t*(b - gq + le))))/(b - gq + le)**3 + \
           (2*dle**2*np.exp(t*(b - gq + le))*t)/(b - gq + le)**2 - \
           (dle**2*np.exp(t*(b - gq + le))*t**2)/(b - gq + le) + \
           (2*dle**2*(-1 + np.exp(t*(b + gq + le))))/(b + gq + le)**3 - \
           (2*dle**2*np.exp(t*(b + gq + le))*t)/(b + gq + le)**2 + \
           (dle**2*np.exp(t*(b + gq + le))*t**2)/(b + gq + le)))/(2.*gq)
    return cvz0 + C00/2*cvz0+ddcv_l0*C11/2+C01*dcv_l0+C23*np.exp(-(b+gq)*t)

def cov12(m0,m1,m2,m3,C00,C01,C02,C03,C11,C12,C13,C22,C23,C33,ml,gl,sl2,mq,gq,sq2,dt,b):
    """X=fun(0), Y=fun(gq)"""
    eglt = np.exp(-gl*dt)
    f = (1-eglt)/(gl*dt)
    le = f*m1 + (1 - f)*ml + ((-3 - eglt**2 + 4*eglt + 2*dt*gl)*sl2)/(2.*gl**3)
    dle = f; t=dt
    fun = lambda gq: (2*np.exp(-(b*t) - gl*t))/(b - gq + le) - (2*np.exp(-(b*t) - gl*t + (b - gq + le)*t))/(b - gq + le) -\
          np.exp(-(b*t) - gl*t)/(b - gl - gq + le) + np.exp(-(b*t) - gl*t + (b - gl - gq + le)*t)/(b - gl - gq + le) -\
          np.exp(-(b*t) - gl*t)/(b + gl - gq + le) + np.exp(-(b*t) - gl*t + (b + gl - gq + le)*t)/(b + gl - gq + le)
    dfun_l0 =lambda gq: (-2*dle*np.exp(-(b*t) - gl*t))/(b - gq + le)**2 + (2*dle*np.exp(-(b*t) - gl*t + (b - gq + le)*t))/(b - gq + le)**2 + \
       (dle*np.exp(-(b*t) - gl*t))/(b - gl - gq + le)**2 - \
       (dle*np.exp(-(b*t) - gl*t + (b - gl - gq + le)*t))/(b - gl - gq + le)**2 + \
       (dle*np.exp(-(b*t) - gl*t))/(b + gl - gq + le)**2 - \
       (dle*np.exp(-(b*t) - gl*t + (b + gl - gq + le)*t))/(b + gl - gq + le)**2 - \
       (2*dle*np.exp(-(b*t) - gl*t + (b - gq + le)*t)*t)/(b - gq + le) + \
       (dle*np.exp(-(b*t) - gl*t + (b - gl - gq + le)*t)*t)/(b - gl - gq + le) + \
       (dle*np.exp(-(b*t) - gl*t + (b + gl - gq + le)*t)*t)/(b + gl - gq + le)
    ddfun_l0 =lambda gq: (4*dle**2*np.exp(-(b*t) - gl*t))/(b - gq + le)**3 - (4*dle**2*np.exp(-(b*t) - gl*t + (b - gq + le)*t))/(b - gq + le)**3 - \
       (2*dle**2*np.exp(-(b*t) - gl*t))/(b - gl - gq + le)**3 + \
       (2*dle**2*np.exp(-(b*t) - gl*t + (b - gl - gq + le)*t))/(b - gl - gq + le)**3 - \
       (2*dle**2*np.exp(-(b*t) - gl*t))/(b + gl - gq + le)**3 + \
       (2*dle**2*np.exp(-(b*t) - gl*t + (b + gl - gq + le)*t))/(b + gl - gq + le)**3 + \
       (4*dle**2*np.exp(-(b*t) - gl*t + (b - gq + le)*t)*t)/(b - gq + le)**2 - \
       (2*dle**2*np.exp(-(b*t) - gl*t + (b - gl - gq + le)*t)*t)/(b - gl - gq + le)**2 - \
       (2*dle**2*np.exp(-(b*t) - gl*t + (b + gl - gq + le)*t)*t)/(b + gl - gq + le)**2 - \
       (2*dle**2*np.exp(-(b*t) - gl*t + (b - gq + le)*t)*t**2)/(b - gq + le) + \
       (dle**2*np.exp(-(b*t) - gl*t + (b - gl - gq + le)*t)*t**2)/(b - gl - gq + le) + \
       (dle**2*np.exp(-(b*t) - gl*t + (b + gl - gq + le)*t)*t**2)/(b + gl - gq + le)
    A = mq*fun(0)+(m3-mq)*fun(gq); dA = mq*dfun_l0(0)+(m3-mq)*dfun_l0(gq);
    ddA = mq*ddfun_l0(0)+(m3-mq)*ddfun_l0(gq)
    return sl2/(2*gl**2)*np.exp(m0)*(A*(1+C00/2)+dA*C01+ddA*C11/2\
                                    +C03*fun(gq)+C13*dfun_l0(gq))+C12*np.exp(-(b+gl)*dt)
def cv20(m0,m1,m2,m3,C00,C01,C02,C03,C11,C12,C13,C22,C23,C33,ml,gl,sl2,mq,gq,sq2,dt,b):
    eglt = np.exp(-gl*dt)
    f = (1-eglt)/(gl*dt)
    le = f*m1 + (1 - f)*ml + ((-3 - eglt**2 + 4*eglt + 2*dt*gl)*sl2)/(2.*gl**3)
    dle = f; t=dt
    fun = lambda gq:         -(((np.exp(gl*t) - np.exp((b - gq + le)*t))/(-b + gl + gq - le) + (2*(-1 + np.exp((b - gq + le)*t)))/(b - gq + le) - \
           (2*(-1 + np.exp((b - gq + le)*t)))/(np.exp(gl*t)*(b - gq + le)) - \
           (2*(-1 + np.exp((b - gl - gq + le)*t)))/(b - gl - gq + le) + \
           (-1 + np.exp((b - gl - gq + le)*t))/(np.exp(gl*t)*(b - gl - gq + le)) - \
           (2*(-1 + np.exp((b - gq + le)*t))*gl*t)/(b - gq + le))/np.exp(b*t))
    dfun = lambda gq:         -(((dle*(np.exp(gl*t) - np.exp((b - gq + le)*t)))/(-b + gl + gq - le)**2 - \
           (2*dle*(-1 + np.exp((b - gq + le)*t)))/(b - gq + le)**2 + \
           (2*dle*(-1 + np.exp((b - gq + le)*t)))/(np.exp(gl*t)*(b - gq + le)**2) + \
           (2*dle*(-1 + np.exp((b - gl - gq + le)*t)))/(b - gl - gq + le)**2 - \
           (dle*(-1 + np.exp((b - gl - gq + le)*t)))/(np.exp(gl*t)*(b - gl - gq + le)**2) - \
           (dle*np.exp((b - gq + le)*t)*t)/(-b + gl + gq - le) + (2*dle*(-1 + np.exp((b - gq + le)*t))*gl*t)/(b - gq + le)**2 + \
           (2*dle*np.exp((b - gq + le)*t)*t)/(b - gq + le) - (2*dle*np.exp(-(gl*t) + (b - gq + le)*t)*t)/(b - gq + le) - \
           (2*dle*np.exp((b - gl - gq + le)*t)*t)/(b - gl - gq + le) + \
           (dle*np.exp(-(gl*t) + (b - gl - gq + le)*t)*t)/(b - gl - gq + le) - \
           (2*dle*np.exp((b - gq + le)*t)*gl*t**2)/(b - gq + le))/np.exp(b*t))
    ddfun = lambda gq:         -(((2*dle**2*(np.exp(gl*t) - np.exp((b - gq + le)*t)))/(-b + gl + gq - le)**3 + \
           (4*dle**2*(-1 + np.exp((b - gq + le)*t)))/(b - gq + le)**3 - \
           (4*dle**2*(-1 + np.exp((b - gl - gq + le)*t)))/(b - gl - gq + le)**3 + \
           (2*dle**2*(-1 + np.exp((b - gl - gq + le)*t)))/(np.exp(gl*t)*(b - gl - gq + le)**3) - \
           (2*dle**2*np.exp((b - gq + le)*t)*t)/(-b + gl + gq - le)**2 - \
           (4*dle**2*(-1 + np.exp((b - gq + le)*t))*gl*t)/(b - gq + le)**3 - \
           (4*dle**2*np.exp((b - gq + le)*t)*t)/(b - gq + le)**2 + \
           (4*dle**2*np.exp((b - gl - gq + le)*t)*t)/(b - gl - gq + le)**2 - \
           (2*dle**2*np.exp(-(gl*t) + (b - gl - gq + le)*t)*t)/(b - gl - gq + le)**2 - \
           (dle**2*np.exp((b - gq + le)*t)*t**2)/(-b + gl + gq - le) + \
           (4*dle**2*np.exp((b - gq + le)*t)*gl*t**2)/(b - gq + le)**2 + (2*dle**2*np.exp((b - gq + le)*t)*t**2)/(b - gq + le) - \
           (2*dle**2*np.exp((b - gl - gq + le)*t)*t**2)/(b - gl - gq + le) + \
           (dle**2*np.exp(-(gl*t) + (b - gl - gq + le)*t)*t**2)/(b - gl - gq + le) - \
           (2*dle**2*np.exp((b - gq + le)*t)*gl*t**3)/(b - gq + le) - \
           (2*((2*dle**2*(-1 + np.exp((b - gq + le)*t)))/(b - gq + le)**3 - \
                (2*dle**2*np.exp((b - gq + le)*t)*t)/(b - gq + le)**2 + (dle**2*np.exp((b - gq + le)*t)*t**2)/(b - gq + le)))/\
            np.exp(gl*t))/np.exp(b*t))
    A = mq*fun(0)+(m3-mq)*fun(gq); dA = mq*dfun(0)+(m3-mq)*dfun(gq);
    ddA = mq*ddfun(0)+(m3-mq)*ddfun(gq)
    return sl2/(2*gl**3)*np.exp(m0)*(A*(1+C00/2)+dA*C01+ddA*C11/2\
           +C03*fun(gq)+C13*dfun(gq))+C02*np.exp(-b*dt)+C12*(1-np.exp(-gl*dt))/gl*np.exp(-b*dt)
def cov22(m0,m1,m2,m3,C00,C01,C02,C03,C11,C12,C13,C22,C23,C33,ml,gl,sl2,mq,gq,sq2,dt,b):
    eglt = np.exp(-gl*dt)
    f = (1-eglt)/(gl*dt)
    le = f*m1 + (1 - f)*ml + ((-3 - eglt**2 + 4*eglt + 2*dt*gl)*sl2)/(2.*gl**3)
    dle = f; t=dt
    F =   (4*np.exp((b - gq + le)*t)*gq*(b + le) + np.exp(2*(b + le)*t)*(b - gq + le)**2 - gq*(b + gq + le) - \
         np.exp(2*(b - gq + le)*t)*(b + le)*(b + gq + le))/(np.exp(2*b*t)*(b + le)*(b - gq + le)**2*(b + gq + le))
    dF =  -((dle*(4*np.exp((b - gq + le)*t)*gq*(b + le) + np.exp(2*(b + le)*t)*(b - gq + le)**2 - gq*(b + gq + le) - \
              np.exp(2*(b - gq + le)*t)*(b + le)*(b + gq + le)))/(np.exp(2*b*t)*(b + le)*(b - gq + le)**2*(b + gq + le)**2)) - \
       (2*dle*(4*np.exp((b - gq + le)*t)*gq*(b + le) + np.exp(2*(b + le)*t)*(b - gq + le)**2 - gq*(b + gq + le) - \
            np.exp(2*(b - gq + le)*t)*(b + le)*(b + gq + le)))/(np.exp(2*b*t)*(b + le)*(b - gq + le)**3*(b + gq + le)) - \
       (dle*(4*np.exp((b - gq + le)*t)*gq*(b + le) + np.exp(2*(b + le)*t)*(b - gq + le)**2 - gq*(b + gq + le) - \
            np.exp(2*(b - gq + le)*t)*(b + le)*(b + gq + le)))/(np.exp(2*b*t)*(b + le)**2*(b - gq + le)**2*(b + gq + le)) + \
       (-(dle*gq) + 4*dle*np.exp((b - gq + le)*t)*gq - dle*np.exp(2*(b - gq + le)*t)*(b + le) + \
          2*dle*np.exp(2*(b + le)*t)*(b - gq + le) - dle*np.exp(2*(b - gq + le)*t)*(b + gq + le) + \
          4*dle*np.exp((b - gq + le)*t)*gq*(b + le)*t + 2*dle*np.exp(2*(b + le)*t)*(b - gq + le)**2*t - \
          2*dle*np.exp(2*(b - gq + le)*t)*(b + le)*(b + gq + le)*t)/(np.exp(2*b*t)*(b + le)*(b - gq + le)**2*(b + gq + le))
    ddF =  (((6*dle**2)/(np.exp(2*b*t)*(b + le)*(b - gq + le)**4) + (4*dle**2)/(np.exp(2*b*t)*(b + le)**2*(b - gq + le)**3) + \
            (2*dle**2)/(np.exp(2*b*t)*(b + le)**3*(b - gq + le)**2))*\
          (4*np.exp((b - gq + le)*t)*gq*(b + le) + np.exp(2*(b + le)*t)*(b - gq + le)**2 - gq*(b + gq + le) - \
            np.exp(2*(b - gq + le)*t)*(b + le)*(b + gq + le)))/(b + gq + le) + \
       2*((-2*dle)/(np.exp(2*b*t)*(b + le)*(b - gq + le)**3) - dle/(np.exp(2*b*t)*(b + le)**2*(b - gq + le)**2))*\
        (-((dle*(4*np.exp((b - gq + le)*t)*gq*(b + le) + np.exp(2*(b + le)*t)*(b - gq + le)**2 - gq*(b + gq + le) - \
                 np.exp(2*(b - gq + le)*t)*(b + le)*(b + gq + le)))/(b + gq + le)**2) + \
          (-(dle*gq) + 4*dle*np.exp((b - gq + le)*t)*gq - dle*np.exp(2*(b - gq + le)*t)*(b + le) + \
             2*dle*np.exp(2*(b + le)*t)*(b - gq + le) - dle*np.exp(2*(b - gq + le)*t)*(b + gq + le) + \
             4*dle*np.exp((b - gq + le)*t)*gq*(b + le)*t + 2*dle*np.exp(2*(b + le)*t)*(b - gq + le)**2*t - \
             2*dle*np.exp(2*(b - gq + le)*t)*(b + le)*(b + gq + le)*t)/(b + gq + le)) + \
       ((2*dle**2*(4*np.exp((b - gq + le)*t)*gq*(b + le) + np.exp(2*(b + le)*t)*(b - gq + le)**2 - gq*(b + gq + le) - \
               np.exp(2*(b - gq + le)*t)*(b + le)*(b + gq + le)))/(b + gq + le)**3 - \
          (2*dle*(-(dle*gq) + 4*dle*np.exp((b - gq + le)*t)*gq - dle*np.exp(2*(b - gq + le)*t)*(b + le) + \
               2*dle*np.exp(2*(b + le)*t)*(b - gq + le) - dle*np.exp(2*(b - gq + le)*t)*(b + gq + le) + \
               4*dle*np.exp((b - gq + le)*t)*gq*(b + le)*t + 2*dle*np.exp(2*(b + le)*t)*(b - gq + le)**2*t - \
               2*dle*np.exp(2*(b - gq + le)*t)*(b + le)*(b + gq + le)*t))/(b + gq + le)**2 + \
          (2*dle**2*np.exp(2*(b + le)*t) - 2*dle**2*np.exp(2*(b - gq + le)*t) + 8*dle**2*np.exp((b - gq + le)*t)*gq*t + \
             8*dle**2*np.exp(2*(b + le)*t)*(b - gq + le)*t - \
             4*dle*np.exp(2*(b - gq + le)*t)*(dle*(b + le) + dle*(b + gq + le))*t + \
             4*dle**2*np.exp((b - gq + le)*t)*gq*(b + le)*t**2 + 4*dle**2*np.exp(2*(b + le)*t)*(b - gq + le)**2*t**2 - \
             4*dle**2*np.exp(2*(b - gq + le)*t)*(b + le)*(b + gq + le)*t**2)/(b + gq + le))/\
        (np.exp(2*b*t)*(b + le)*(b - gq + le)**2)
    return sq2/(2*gq)*np.exp(2*m0)*(F*(1+C00/2)+dF*C01+ddF*C11/2)+C22*np.exp(-2*b*dt)

#def cov22(m0,m1,m2,m3,C00,C01,C02,C03,C11,C12,C13,C22,C23,C33,ml,gl,sl2,mq,gq,sq2,dt,b):
#    return np.exp(C11)*C33*dt**2+(1-b*dt)**2*C22
def new_cov(m0,m1,m2,m3,C00,C01,C02,C03,C11,C12,C13,C22,C23,C33,ml,gl,sl2,mq,gq,sq2,dt,b):
    """Start from P(z_t|D_t)= N(m;C) and find P(z_{t+dt}|D_t), the covariance here"""
    eglt = np.exp(-gl*dt)
    egqt = np.exp(-gq*dt)
    vx = sl2/(2*gl**3)*(2*gl*dt-3+4*eglt-eglt**2)
    cxl = sl2/(2*gl**2)*(1-eglt)**2
    cxq = 0; clq = 0
    vl = sl2/(2*gl)*(1-eglt**2)
    vq = sq2/(2*gq)*(1-egqt**2)
    nC00 = (C11*(-1+eglt)**2+gl*(-2*C01*(-1+eglt)+gl*(C00+vx)))/gl**2
    nC01 = cxl+(eglt*(C11-C11*eglt+C01*gl))/gl
    nC03 = cxq+(egqt*(C13-C13*eglt+C03*gl))/gl
    nC11 = C11*eglt**2 + vl
    nC13 = clq + C13*eglt*egqt
    nC33 = vq + C33*egqt**2
    nC02 = cv20(m0,m1,m2,m3,C00,C01,C02,C03,C11,C12,C13,C22,C23,C33,ml,gl,sl2,mq,gq,sq2,dt,b)
    nC12 = cov12(m0,m1,m2,m3,C00,C01,C02,C03,C11,C12,C13,C22,C23,C33,ml,gl,sl2,mq,gq,sq2,dt,b)
    nC22 = cov22(m0,m1,m2,m3,C00,C01,C02,C03,C11,C12,C13,C22,C23,C33,ml,gl,sl2,mq,gq,sq2,dt,b)
    nC23 = cov23(m0,m1,m2,m3,C00,C01,C02,C03,C11,C12,C13,C22,C23,C33,ml,gl,sl2,mq,gq,sq2,dt,b)
    return nC00,nC01,nC02,nC03,nC11,nC12,nC13,nC22,nC23,nC33
def new_mean_cov(m0,m1,m2,m3,C00,C01,C02,C03,C11,C12,C13,C22,C23,C33,ml,gl,sl2,mq,gq,sq2,dt,b):
    nm = new_mean(m0,m1,m2,m3,C00,C01,C02,C03,C11,C12,C13,C22,C23,C33,ml,gl,sl2,mq,gq,sq2,dt,b)
    Q = new_cov(m0,m1,m2,m3,C00,C01,C02,C03,C11,C12,C13,C22,C23,C33,ml,gl,sl2,mq,gq,sq2,dt,b)
    return nm, Q
############
############    THE UPDATE MEAN AND COVARIANCE OF  P(z_{t+dt}|D_{t+dt})~N(m,C) assuming we know P(z_{t+dt}|D_{t})~N(nm,Q)
############
def post_cov(IQ00,IQ01,IQ02,IQ03,IQ11,IQ12,IQ13,IQ22,IQ23,IQ33,sx2,sg2):
    """Posterior covariance, I stands for Inverse"""
    return inverseQ(IQ00+1/sx2,IQ01,IQ02,IQ03,IQ11,IQ12,IQ13,IQ22+1/sg2,IQ23,IQ33)
def post_mean(xm,gm,m0,m1,m2,m3,IQ00,IQ01,IQ02,IQ03,IQ11,IQ12,IQ13,IQ22,IQ23,IQ33,sx2,sg2):
    """Posterior mean. Input IQ is the inverse of the Q matrix. Input m is the mean of P(z_{t+dt}|D_{t})~N(nm,Q)"""
    pm0 = (gm*(IQ03*IQ12*IQ13 - IQ02*IQ13**2 - IQ03*IQ11*IQ23 + IQ01*IQ13*IQ23 + IQ02*IQ11*IQ33 - \
            IQ01*IQ12*IQ33)*sx2 + IQ00*IQ13**2*m0*sx2 + IQ01**2*IQ33*m0*sx2 - IQ00*IQ11*IQ33*m0*sx2 + \
            IQ02*IQ13**2*m2*sx2 - IQ01*IQ13*IQ23*m2*sx2 - IQ02*IQ11*IQ33*m2*sx2 + IQ01*IQ12*IQ33*m2*sx2 - \
            IQ02**2*IQ13**2*m0*sg2*sx2 + IQ00*IQ13**2*IQ22*m0*sg2*sx2 + 2*IQ01*IQ02*IQ13*IQ23*m0*sg2*sx2 - \
            2*IQ00*IQ12*IQ13*IQ23*m0*sg2*sx2 - IQ01**2*IQ23**2*m0*sg2*sx2 + IQ00*IQ11*IQ23**2*m0*sg2*sx2 + \
            IQ02**2*IQ11*IQ33*m0*sg2*sx2 - 2*IQ01*IQ02*IQ12*IQ33*m0*sg2*sx2 + IQ00*IQ12**2*IQ33*m0*sg2*sx2 + \
            IQ01**2*IQ22*IQ33*m0*sg2*sx2 - IQ00*IQ11*IQ22*IQ33*m0*sg2*sx2 + \
            IQ03**2*m0*(IQ11 - IQ12**2*sg2 + IQ11*IQ22*sg2)*sx2 + \
            IQ03*(-2*IQ01*m0*(IQ13 + IQ13*IQ22*sg2 - IQ12*IQ23*sg2) - (IQ12*IQ13 - IQ11*IQ23)*(m2 - 2*IQ02*m0*sg2))*sx2 + \
            IQ13**2*xm - IQ11*IQ33*xm + IQ13**2*IQ22*sg2*xm - 2*IQ12*IQ13*IQ23*sg2*xm + IQ11*IQ23**2*sg2*xm + \
            IQ12**2*IQ33*sg2*xm - IQ11*IQ22*IQ33*sg2*xm)/\
          (IQ12**2*IQ33*sg2 + IQ01**2*IQ33*sx2 - IQ03**2*IQ12**2*sg2*sx2 + 2*IQ01*IQ03*IQ12*IQ23*sg2*sx2 - \
            IQ01**2*IQ23**2*sg2*sx2 - 2*IQ01*IQ02*IQ12*IQ33*sg2*sx2 + IQ00*IQ12**2*IQ33*sg2*sx2 + IQ01**2*IQ22*IQ33*sg2*sx2 - \
            2*IQ13*(IQ01*(IQ03 + IQ03*IQ22*sg2 - IQ02*IQ23*sg2)*sx2 + IQ12*sg2*(IQ23 - IQ02*IQ03*sx2 + IQ00*IQ23*sx2)) + \
            IQ13**2*(1 + IQ00*sx2 - IQ02**2*sg2*sx2 + IQ22*(sg2 + IQ00*sg2*sx2)) + \
            IQ11*(-2*IQ02*IQ03*IQ23*sg2*sx2 + IQ03**2*(1 + IQ22*sg2)*sx2 + IQ23**2*(sg2 + IQ00*sg2*sx2) - \
               IQ33*(1 + IQ00*sx2 - IQ02**2*sg2*sx2 + IQ22*(sg2 + IQ00*sg2*sx2))))
    pm1 = (-(IQ01*IQ33*m0) + IQ13**2*m1 - IQ11*IQ33*m1 + IQ13*IQ23*m2 - IQ12*IQ33*m2 - IQ02*IQ13*IQ23*m0*sg2 + \
            IQ01*IQ23**2*m0*sg2 + IQ02*IQ12*IQ33*m0*sg2 - IQ01*IQ22*IQ33*m0*sg2 + IQ13**2*IQ22*m1*sg2 - \
            2*IQ12*IQ13*IQ23*m1*sg2 + IQ11*IQ23**2*m1*sg2 + IQ12**2*IQ33*m1*sg2 - IQ11*IQ22*IQ33*m1*sg2 + \
            IQ00*IQ13**2*m1*sx2 + IQ01**2*IQ33*m1*sx2 - IQ00*IQ11*IQ33*m1*sx2 + IQ00*IQ13*IQ23*m2*sx2 + \
            IQ01*IQ02*IQ33*m2*sx2 - IQ00*IQ12*IQ33*m2*sx2 - IQ02**2*IQ13**2*m1*sg2*sx2 + IQ00*IQ13**2*IQ22*m1*sg2*sx2 + \
            2*IQ01*IQ02*IQ13*IQ23*m1*sg2*sx2 - 2*IQ00*IQ12*IQ13*IQ23*m1*sg2*sx2 - IQ01**2*IQ23**2*m1*sg2*sx2 + \
            IQ00*IQ11*IQ23**2*m1*sg2*sx2 + IQ02**2*IQ11*IQ33*m1*sg2*sx2 - 2*IQ01*IQ02*IQ12*IQ33*m1*sg2*sx2 + \
            IQ00*IQ12**2*IQ33*m1*sg2*sx2 + IQ01**2*IQ22*IQ33*m1*sg2*sx2 - IQ00*IQ11*IQ22*IQ33*m1*sg2*sx2 + \
            IQ03**2*(IQ12*(m2 - IQ12*m1*sg2) + IQ11*(m1 + IQ22*m1*sg2))*sx2 + \
            gm*(IQ01*(IQ03*IQ23 - IQ02*IQ33)*sx2 - IQ13*(IQ23 - IQ02*IQ03*sx2 + IQ00*IQ23*sx2) + \
               IQ12*(IQ33 - IQ03**2*sx2 + IQ00*IQ33*sx2)) + IQ01*IQ33*xm + IQ02*IQ13*IQ23*sg2*xm - IQ01*IQ23**2*sg2*xm - \
            IQ02*IQ12*IQ33*sg2*xm + IQ01*IQ22*IQ33*sg2*xm + \
            IQ03*(-(IQ23*((IQ01*m2 + 2*IQ02*IQ11*m1*sg2)*sx2 + IQ12*sg2*(m0 - 2*IQ01*m1*sx2 - xm))) + \
               IQ13*(m0 + IQ22*m0*sg2 - IQ02*m2*sx2 + 2*IQ02*IQ12*m1*sg2*sx2 - 2*IQ01*m1*(1 + IQ22*sg2)*sx2 - xm - \
                  IQ22*sg2*xm)))/\
          (IQ12**2*IQ33*sg2 + IQ01**2*IQ33*sx2 - IQ03**2*IQ12**2*sg2*sx2 + 2*IQ01*IQ03*IQ12*IQ23*sg2*sx2 - \
            IQ01**2*IQ23**2*sg2*sx2 - 2*IQ01*IQ02*IQ12*IQ33*sg2*sx2 + IQ00*IQ12**2*IQ33*sg2*sx2 + IQ01**2*IQ22*IQ33*sg2*sx2 - \
            2*IQ13*(IQ01*(IQ03 + IQ03*IQ22*sg2 - IQ02*IQ23*sg2)*sx2 + IQ12*sg2*(IQ23 - IQ02*IQ03*sx2 + IQ00*IQ23*sx2)) + \
            IQ13**2*(1 + IQ00*sx2 - IQ02**2*sg2*sx2 + IQ22*(sg2 + IQ00*sg2*sx2)) + \
            IQ11*(-2*IQ02*IQ03*IQ23*sg2*sx2 + IQ03**2*(1 + IQ22*sg2)*sx2 + IQ23**2*(sg2 + IQ00*sg2*sx2) - \
               IQ33*(1 + IQ00*sx2 - IQ02**2*sg2*sx2 + IQ22*(sg2 + IQ00*sg2*sx2))))
    pm2 = (gm*(-2*IQ01*IQ03*IQ13*sx2 + IQ01**2*IQ33*sx2 + IQ13**2*(1 + IQ00*sx2) - \
               IQ11*(IQ33 - IQ03**2*sx2 + IQ00*IQ33*sx2)) + \
            sg2*(-(IQ01*IQ13*IQ23*m0) + IQ01*IQ12*IQ33*m0 + IQ13**2*IQ22*m2 - 2*IQ12*IQ13*IQ23*m2 + IQ11*IQ23**2*m2 + \
               IQ12**2*IQ33*m2 - IQ11*IQ22*IQ33*m2 + IQ00*IQ13**2*IQ22*m2*sx2 + IQ03**2*(-IQ12**2 + IQ11*IQ22)*m2*sx2 - \
               2*IQ00*IQ12*IQ13*IQ23*m2*sx2 - IQ01**2*IQ23**2*m2*sx2 + IQ00*IQ11*IQ23**2*m2*sx2 + IQ00*IQ12**2*IQ33*m2*sx2 + \
               IQ01**2*IQ22*IQ33*m2*sx2 - IQ00*IQ11*IQ22*IQ33*m2*sx2 + IQ02**2*(-IQ13**2 + IQ11*IQ33)*m2*sx2 + \
               IQ01*IQ13*IQ23*xm - IQ01*IQ12*IQ33*xm + \
               IQ02*(2*IQ01*IQ13*IQ23*m2*sx2 - 2*IQ01*IQ12*IQ33*m2*sx2 + IQ13**2*(m0 - xm) + IQ11*IQ33*(-m0 + xm)) + \
               IQ03*(-2*IQ01*IQ13*IQ22*m2*sx2 + 2*IQ01*IQ12*IQ23*m2*sx2 + IQ11*IQ23*(m0 - 2*IQ02*m2*sx2 - xm) + \
                  IQ12*IQ13*(-m0 + 2*IQ02*m2*sx2 + xm))))/\
          (IQ12**2*IQ33*sg2 + IQ01**2*IQ33*sx2 - IQ03**2*IQ12**2*sg2*sx2 + 2*IQ01*IQ03*IQ12*IQ23*sg2*sx2 - \
            IQ01**2*IQ23**2*sg2*sx2 - 2*IQ01*IQ02*IQ12*IQ33*sg2*sx2 + IQ00*IQ12**2*IQ33*sg2*sx2 + IQ01**2*IQ22*IQ33*sg2*sx2 - \
            2*IQ13*(IQ01*(IQ03 + IQ03*IQ22*sg2 - IQ02*IQ23*sg2)*sx2 + IQ12*sg2*(IQ23 - IQ02*IQ03*sx2 + IQ00*IQ23*sx2)) + \
            IQ13**2*(1 + IQ00*sx2 - IQ02**2*sg2*sx2 + IQ22*(sg2 + IQ00*sg2*sx2)) + \
            IQ11*(-2*IQ02*IQ03*IQ23*sg2*sx2 + IQ03**2*(1 + IQ22*sg2)*sx2 + IQ23**2*(sg2 + IQ00*sg2*sx2) - \
               IQ33*(1 + IQ00*sx2 - IQ02**2*sg2*sx2 + IQ22*(sg2 + IQ00*sg2*sx2))))
    pm3 = (IQ01*IQ13*m0 + IQ12*IQ13*m2 - IQ11*IQ23*m2 + IQ13**2*m3 - IQ11*IQ33*m3 - IQ02*IQ12*IQ13*m0*sg2 + \
            IQ01*IQ13*IQ22*m0*sg2 + IQ02*IQ11*IQ23*m0*sg2 - IQ01*IQ12*IQ23*m0*sg2 + IQ13**2*IQ22*m3*sg2 - \
            2*IQ12*IQ13*IQ23*m3*sg2 + IQ11*IQ23**2*m3*sg2 + IQ12**2*IQ33*m3*sg2 - IQ11*IQ22*IQ33*m3*sg2 - \
            IQ01*IQ02*IQ13*m2*sx2 + IQ00*IQ12*IQ13*m2*sx2 + IQ01**2*IQ23*m2*sx2 - IQ00*IQ11*IQ23*m2*sx2 + \
            IQ00*IQ13**2*m3*sx2 + IQ01**2*IQ33*m3*sx2 - IQ00*IQ11*IQ33*m3*sx2 - IQ02**2*IQ13**2*m3*sg2*sx2 + \
            IQ00*IQ13**2*IQ22*m3*sg2*sx2 + 2*IQ01*IQ02*IQ13*IQ23*m3*sg2*sx2 - 2*IQ00*IQ12*IQ13*IQ23*m3*sg2*sx2 - \
            IQ01**2*IQ23**2*m3*sg2*sx2 + IQ00*IQ11*IQ23**2*m3*sg2*sx2 + IQ02**2*IQ11*IQ33*m3*sg2*sx2 - \
            2*IQ01*IQ02*IQ12*IQ33*m3*sg2*sx2 + IQ00*IQ12**2*IQ33*m3*sg2*sx2 + IQ01**2*IQ22*IQ33*m3*sg2*sx2 - \
            IQ00*IQ11*IQ22*IQ33*m3*sg2*sx2 + IQ03**2*m3*(IQ11 - IQ12**2*sg2 + IQ11*IQ22*sg2)*sx2 + \
            gm*(IQ01*(IQ02*IQ13 - IQ01*IQ23)*sx2 - IQ12*(IQ13 - IQ01*IQ03*sx2 + IQ00*IQ13*sx2) + \
               IQ11*(IQ23 - IQ02*IQ03*sx2 + IQ00*IQ23*sx2)) - IQ01*IQ13*xm + IQ02*IQ12*IQ13*sg2*xm - IQ01*IQ13*IQ22*sg2*xm - \
            IQ02*IQ11*IQ23*sg2*xm + IQ01*IQ12*IQ23*sg2*xm + \
            IQ03*(-2*IQ01*IQ13*m3*(1 + IQ22*sg2)*sx2 + IQ12*(-(IQ01*m2) + 2*IQ02*IQ13*m3*sg2 + 2*IQ01*IQ23*m3*sg2)*sx2 + \
               IQ12**2*sg2*(m0 - xm) + IQ11*(-(m0*(1 + IQ22*sg2)) + IQ02*(m2 - 2*IQ23*m3*sg2)*sx2 + xm + IQ22*sg2*xm)))/\
          (IQ12**2*IQ33*sg2 + IQ01**2*IQ33*sx2 - IQ03**2*IQ12**2*sg2*sx2 + 2*IQ01*IQ03*IQ12*IQ23*sg2*sx2 - \
            IQ01**2*IQ23**2*sg2*sx2 - 2*IQ01*IQ02*IQ12*IQ33*sg2*sx2 + IQ00*IQ12**2*IQ33*sg2*sx2 + IQ01**2*IQ22*IQ33*sg2*sx2 - \
            2*IQ13*(IQ01*(IQ03 + IQ03*IQ22*sg2 - IQ02*IQ23*sg2)*sx2 + IQ12*sg2*(IQ23 - IQ02*IQ03*sx2 + IQ00*IQ23*sx2)) + \
            IQ13**2*(1 + IQ00*sx2 - IQ02**2*sg2*sx2 + IQ22*(sg2 + IQ00*sg2*sx2)) + \
            IQ11*(-2*IQ02*IQ03*IQ23*sg2*sx2 + IQ03**2*(1 + IQ22*sg2)*sx2 + IQ23**2*(sg2 + IQ00*sg2*sx2) - \
               IQ33*(1 + IQ00*sx2 - IQ02**2*sg2*sx2 + IQ22*(sg2 + IQ00*sg2*sx2))))
    return pm0,pm1,pm2,pm3
def posteriori_matrices(xm,gm,m0,m1,m2,m3,IQ00,IQ01,IQ02,IQ03,IQ11,IQ12,IQ13,IQ22,IQ23,IQ33,sx2,sg2):
    """return mean and cov of P(z_{t+dt}|D_{t+dt})"""
    m = post_mean(xm,gm,m0,m1,m2,m3,IQ00,IQ01,IQ02,IQ03,IQ11,IQ12,IQ13,IQ22,IQ23,IQ33,sx2,sg2)
    C = post_cov(IQ00,IQ01,IQ02,IQ03,IQ11,IQ12,IQ13,IQ22,IQ23,IQ33,sx2,sg2)
    return m, C
######
###### THE LOG LIKELIHOOD P(x_{t+dt}g_{t+dt}|D_t)
######
# We know that p(z_{t+dt}^m|z_{t+dt})p(z_{t+dt}|D_t) ~N(post_mean,post_cov) and the integral of it gives a term sqrt((2pi^)4det(post_cov)) but it is not normalized. The right normalization is (sqrt(2*pi*sg2*2pi*sx2*det|Q|))*prefactor where prefacto are all the terms not depending on (z-nm) when we multiplied the two functions.
def prefactor(xm,gm,m0,m1,m2,m3,IQ00,IQ01,IQ02,IQ03,IQ11,IQ12,IQ13,IQ22,IQ23,IQ33,sx2,sg2):
        return (-(IQ00*IQ13**2*m0**2 + IQ01**2*IQ33*m0**2 - IQ00*IQ11*IQ33*m0**2 + 2*IQ02*IQ13**2*m0*m2 - \
           2*IQ01*IQ13*IQ23*m0*m2 - 2*IQ02*IQ11*IQ33*m0*m2 + 2*IQ01*IQ12*IQ33*m0*m2 + IQ13**2*IQ22*m2**2 - \
           2*IQ12*IQ13*IQ23*m2**2 + IQ11*IQ23**2*m2**2 + IQ12**2*IQ33*m2**2 - IQ11*IQ22*IQ33*m2**2 - \
           IQ02**2*IQ13**2*m0**2*sg2 + IQ00*IQ13**2*IQ22*m0**2*sg2 + 2*IQ01*IQ02*IQ13*IQ23*m0**2*sg2 - \
           2*IQ00*IQ12*IQ13*IQ23*m0**2*sg2 - IQ01**2*IQ23**2*m0**2*sg2 + IQ00*IQ11*IQ23**2*m0**2*sg2 + \
           IQ02**2*IQ11*IQ33*m0**2*sg2 - 2*IQ01*IQ02*IQ12*IQ33*m0**2*sg2 + IQ00*IQ12**2*IQ33*m0**2*sg2 + \
           IQ01**2*IQ22*IQ33*m0**2*sg2 - IQ00*IQ11*IQ22*IQ33*m0**2*sg2 - IQ02**2*IQ13**2*m2**2*sx2 + \
           IQ00*IQ13**2*IQ22*m2**2*sx2 + 2*IQ01*IQ02*IQ13*IQ23*m2**2*sx2 - 2*IQ00*IQ12*IQ13*IQ23*m2**2*sx2 - \
           IQ01**2*IQ23**2*m2**2*sx2 + IQ00*IQ11*IQ23**2*m2**2*sx2 + IQ02**2*IQ11*IQ33*m2**2*sx2 - \
           2*IQ01*IQ02*IQ12*IQ33*m2**2*sx2 + IQ00*IQ12**2*IQ33*m2**2*sx2 + IQ01**2*IQ22*IQ33*m2**2*sx2 - \
           IQ00*IQ11*IQ22*IQ33*m2**2*sx2 + gm**2*(IQ12**2*IQ33 - IQ03**2*IQ12**2*sx2 + 2*IQ01*IQ03*IQ12*IQ23*sx2 - \
              IQ01**2*IQ23**2*sx2 - 2*IQ01*IQ02*IQ12*IQ33*sx2 + IQ00*IQ12**2*IQ33*sx2 + IQ01**2*IQ22*IQ33*sx2 + \
              IQ13**2*(IQ22 - IQ02**2*sx2 + IQ00*IQ22*sx2) - \
              2*IQ13*(IQ01*(IQ03*IQ22 - IQ02*IQ23)*sx2 + IQ12*(IQ23 - IQ02*IQ03*sx2 + IQ00*IQ23*sx2)) + \
              IQ11*(-2*IQ02*IQ03*IQ23*sx2 + IQ02**2*IQ33*sx2 + IQ23**2*(1 + IQ00*sx2) - \
                 IQ22*(IQ33 - IQ03**2*sx2 + IQ00*IQ33*sx2))) - 2*IQ00*IQ13**2*m0*xm - 2*IQ01**2*IQ33*m0*xm + \
           2*IQ00*IQ11*IQ33*m0*xm - 2*IQ02*IQ13**2*m2*xm + 2*IQ01*IQ13*IQ23*m2*xm + 2*IQ02*IQ11*IQ33*m2*xm - \
           2*IQ01*IQ12*IQ33*m2*xm + 2*IQ02**2*IQ13**2*m0*sg2*xm - 2*IQ00*IQ13**2*IQ22*m0*sg2*xm - \
           4*IQ01*IQ02*IQ13*IQ23*m0*sg2*xm + 4*IQ00*IQ12*IQ13*IQ23*m0*sg2*xm + 2*IQ01**2*IQ23**2*m0*sg2*xm - \
           2*IQ00*IQ11*IQ23**2*m0*sg2*xm - 2*IQ02**2*IQ11*IQ33*m0*sg2*xm + 4*IQ01*IQ02*IQ12*IQ33*m0*sg2*xm - \
           2*IQ00*IQ12**2*IQ33*m0*sg2*xm - 2*IQ01**2*IQ22*IQ33*m0*sg2*xm + 2*IQ00*IQ11*IQ22*IQ33*m0*sg2*xm + \
           IQ00*IQ13**2*xm**2 + IQ01**2*IQ33*xm**2 - IQ00*IQ11*IQ33*xm**2 - IQ02**2*IQ13**2*sg2*xm**2 + \
           IQ00*IQ13**2*IQ22*sg2*xm**2 + 2*IQ01*IQ02*IQ13*IQ23*sg2*xm**2 - 2*IQ00*IQ12*IQ13*IQ23*sg2*xm**2 - \
           IQ01**2*IQ23**2*sg2*xm**2 + IQ00*IQ11*IQ23**2*sg2*xm**2 + IQ02**2*IQ11*IQ33*sg2*xm**2 - \
           2*IQ01*IQ02*IQ12*IQ33*sg2*xm**2 + IQ00*IQ12**2*IQ33*sg2*xm**2 + IQ01**2*IQ22*IQ33*sg2*xm**2 - \
           IQ00*IQ11*IQ22*IQ33*sg2*xm**2 + 2*IQ03*(IQ12*IQ13 - IQ11*IQ23)*\
            (IQ02*m0**2*sg2 + IQ02*m2**2*sx2 + m2*xm + IQ02*sg2*xm**2 - m0*(m2 + 2*IQ02*sg2*xm)) + \
           2*gm*(IQ01*IQ13*IQ23*m0 - IQ01*IQ12*IQ33*m0 - IQ13**2*IQ22*m2 + 2*IQ12*IQ13*IQ23*m2 - IQ11*IQ23**2*m2 - \
              IQ12**2*IQ33*m2 + IQ11*IQ22*IQ33*m2 - IQ00*IQ13**2*IQ22*m2*sx2 + IQ03**2*(IQ12**2 - IQ11*IQ22)*m2*sx2 + \
              2*IQ00*IQ12*IQ13*IQ23*m2*sx2 + IQ01**2*IQ23**2*m2*sx2 - IQ00*IQ11*IQ23**2*m2*sx2 - IQ00*IQ12**2*IQ33*m2*sx2 - \
              IQ01**2*IQ22*IQ33*m2*sx2 + IQ00*IQ11*IQ22*IQ33*m2*sx2 + IQ02**2*(IQ13**2 - IQ11*IQ33)*m2*sx2 - \
              IQ01*IQ13*IQ23*xm + IQ01*IQ12*IQ33*xm + \
              IQ02*(-2*IQ01*IQ13*IQ23*m2*sx2 + 2*IQ01*IQ12*IQ33*m2*sx2 + IQ11*IQ33*(m0 - xm) + IQ13**2*(-m0 + xm)) + \
              IQ03*(2*IQ01*IQ13*IQ22*m2*sx2 - 2*IQ01*IQ12*IQ23*m2*sx2 + IQ12*IQ13*(m0 - 2*IQ02*m2*sx2 - xm) + \
                 IQ11*IQ23*(-m0 + 2*IQ02*m2*sx2 + xm))) + \
           IQ03**2*(-(IQ12**2*(m0**2*sg2 + m2**2*sx2 - 2*m0*sg2*xm + sg2*xm**2)) + \
              IQ11*(m0**2*(1 + IQ22*sg2) + IQ22*m2**2*sx2 + xm**2 + IQ22*sg2*xm**2 - 2*m0*(xm + IQ22*sg2*xm))) - \
           2*IQ01*IQ03*(-(IQ12*IQ23*(m0**2*sg2 + m2**2*sx2 - 2*m0*sg2*xm + sg2*xm**2)) + \
              IQ13*(m0**2*(1 + IQ22*sg2) + IQ22*m2**2*sx2 + xm**2 + IQ22*sg2*xm**2 - 2*m0*(xm + IQ22*sg2*xm))))/\
        (2.*(IQ12**2*IQ33*sg2 + IQ01**2*IQ33*sx2 - IQ03**2*IQ12**2*sg2*sx2 + 2*IQ01*IQ03*IQ12*IQ23*sg2*sx2 - \
            IQ01**2*IQ23**2*sg2*sx2 - 2*IQ01*IQ02*IQ12*IQ33*sg2*sx2 + IQ00*IQ12**2*IQ33*sg2*sx2 + IQ01**2*IQ22*IQ33*sg2*sx2 - \
            2*IQ13*(IQ01*(IQ03 + IQ03*IQ22*sg2 - IQ02*IQ23*sg2)*sx2 + IQ12*sg2*(IQ23 - IQ02*IQ03*sx2 + IQ00*IQ23*sx2)) + \
            IQ13**2*(1 + IQ00*sx2 - IQ02**2*sg2*sx2 + IQ22*(sg2 + IQ00*sg2*sx2)) + \
            IQ11*(-2*IQ02*IQ03*IQ23*sg2*sx2 + IQ03**2*(1 + IQ22*sg2)*sx2 + IQ23**2*(sg2 + IQ00*sg2*sx2) - \
               IQ33*(1 + IQ00*sx2 - IQ02**2*sg2*sx2 + IQ22*(sg2 + IQ00*sg2*sx2))))))
def log_likelihood(xm,gm,m0,m1,m2,m3,IQ00,IQ01,IQ02,IQ03,IQ11,IQ12,IQ13,IQ22,IQ23,IQ33,PC00,PC01,PC02,PC03,PC11,PC12,PC13,PC22,PC23,PC33,sx2,sg2):
    pr = prefactor(xm,gm,m0,m1,m2,m3,IQ00,IQ01,IQ02,IQ03,IQ11,IQ12,IQ13,IQ22,IQ23,IQ33,sx2,sg2)
    return pr + 1/2*(det(IQ00,IQ01,IQ02,IQ03,IQ11,IQ12,IQ13,IQ22,IQ23,IQ33)+det(PC00,PC01,PC02,PC03,PC11,PC12,PC13,PC22,PC23,PC33)-np.log(sg2)-np.log(sx2))-np.pi
#------------------ LOG LIKELIHOOD AT CELL DIVISION-----------------------------------
def cell_division_likelihood_and_grad_(m0,m1,m2,m3,IQ00,IQ01,IQ02,IQ03,IQ11,IQ12,IQ13,IQ22,IQ23,IQ33,sxd2,sgd2):
    """The lengths and gfp get divided in 2 with sdx2 and sdg2 as variances"""
    IS00 =(IQ00 - 4*IQ02**2*sgd2 + 4*IQ00*IQ22*sgd2)/(1 + IQ00*sxd2 - 4*IQ02**2*sgd2*sxd2 + 4*IQ22*(sgd2 + IQ00*sgd2*sxd2))
    IS01 =(IQ01 - 4*IQ02*IQ12*sgd2 + 4*IQ01*IQ22*sgd2)/(1 + IQ00*sxd2 - 4*IQ02**2*sgd2*sxd2 + 4*IQ22*(sgd2 + IQ00*sgd2*sxd2))
    IS02 = (2*IQ02)/(1 + IQ00*sxd2 - 4*IQ02**2*sgd2*sxd2 + 4*IQ22*(sgd2 + IQ00*sgd2*sxd2))
    IS03 = (IQ03 + 4*IQ03*IQ22*sgd2 - 4*IQ02*IQ23*sgd2)/(1 + IQ00*sxd2 - 4*IQ02**2*sgd2*sxd2 + 4*IQ22*(sgd2 + IQ00*sgd2*sxd2))
    IS11 = -((-8*IQ01*IQ02*IQ12*sgd2*sxd2 + IQ01**2*(1 + 4*IQ22*sgd2)*sxd2 + 4*IQ12**2*(sgd2 + IQ00*sgd2*sxd2) -\
           IQ11*(1 + IQ00*sxd2 - 4*IQ02**2*sgd2*sxd2 + 4*IQ22*(sgd2 + IQ00*sgd2*sxd2)))/\
         (1 + IQ00*sxd2 - 4*IQ02**2*sgd2*sxd2 + 4*IQ22*(sgd2 + IQ00*sgd2*sxd2)))
    IS12 = (2*(IQ12 - IQ01*IQ02*sxd2 + IQ00*IQ12*sxd2))/(1 + IQ00*sxd2 - 4*IQ02**2*sgd2*sxd2 + 4*IQ22*(sgd2 + IQ00*sgd2*sxd2))
    IS13 = -((IQ01*(IQ03 + 4*IQ03*IQ22*sgd2 - 4*IQ02*IQ23*sgd2)*sxd2 + 4*IQ12*sgd2*(IQ23 - IQ02*IQ03*sxd2 + IQ00*IQ23*sxd2) -\
           IQ13*(1 + IQ00*sxd2 - 4*IQ02**2*sgd2*sxd2 + 4*IQ22*(sgd2 + IQ00*sgd2*sxd2)))/\
           (1 + IQ00*sxd2 - 4*IQ02**2*sgd2*sxd2 + 4*IQ22*(sgd2 + IQ00*sgd2*sxd2)))
    IS22 =(4*(IQ22 - IQ02**2*sxd2 + IQ00*IQ22*sxd2))/(1 + IQ00*sxd2 - 4*IQ02**2*sgd2*sxd2 + 4*IQ22*(sgd2 + IQ00*sgd2*sxd2))
    IS23 = 0
    IS33 = (4*(IQ22 - IQ02**2*sxd2 + IQ00*IQ22*sxd2))/(1 + IQ00*sxd2 - 4*IQ02**2*sgd2*sxd2 + 4*IQ22*(sgd2 + IQ00*sgd2*sxd2))
    #S = inverseQ(IS00,IS01,IS02,IS03,IS11,IS12,IS13,IS22,IS23,IS33)
    s = (m0-np.log(2),m1,m2/2,m3)
    return s,(IS00,IS01,IS02,IS03,IS11,IS12,IS13,IS22,IS23,IS33)

def cell_division_likelihood_and_grad(m0,m1,m2,m3,IQ00,IQ01,IQ02,IQ03,IQ11,IQ12,IQ13,IQ22,IQ23,IQ33,sxd2,sgd2):
    s = (m0-np.log(2),m1,m2/2,m3)
    IS00,IS01,IS02,IS03,IS11,IS12,IS13,IS22,IS23,IS33 = inverseQ(IQ00,IQ01,IQ02,IQ03,IQ11,IQ12,IQ13,IQ22,IQ23,IQ33)
    IS00+=sxd2; IS22+=sgd2
    return s,inverseQ(IS00,IS01,IS02,IS03,IS11,IS12,IS13,IS22,IS23,IS33)
#------------------ OBJECTIVE OVER CELL CYCLE/ LANE AND TOTAL-----------------------------------
def obj_and_grad_1cc(W,ml,gl,sl2,mq,gq,sq2,sx2,sg2,sxd2,sgd2,s0,s1,s2,s3,IS00,IS01,IS02,IS03,IS11,IS12,IS13,IS22,IS23,IS33,dt,b):
    """ Compute the objective over 1 cell cycle"""
    assert W.shape[1]==2
    # p(z_0|D_0)
    nm,PC = posteriori_matrices(W[0,0],W[0,1],s0,s1,s2,s3,IS00,IS01,IS02,IS03,IS11,IS12,IS13,IS22,IS23,IS33,sx2,sg2)
    ##### p(D_0)
    ll = log_likelihood(W[0,0],W[0,1],s0,s1,s2,s3,IS00,IS01,IS02,IS03,IS11,IS12,IS13,IS22,IS23,IS33,PC[0],PC[1],PC[2],PC[3],PC[4],PC[5],PC[6],PC[7],PC[8],PC[9],sx2,sg2)
    for j in range(1,W.shape[0]):
        ###### P(z_{t+dt}|D_t)
        m,Q = new_mean_cov(nm[0],nm[1],nm[2],nm[3],PC[0],PC[1],PC[2],PC[3],PC[4],PC[5],PC[6],PC[7],PC[8],PC[9],ml,gl,sl2,mq,gq,sq2,dt,b)
        ##### P(z_{t+dt}|D_{t+dt}) = N(b',B')
        IQ00,IQ01,IQ02,IQ03,IQ11,IQ12,IQ13,IQ22,IQ23,IQ33 = inverseQ(Q[0],Q[1],Q[2],Q[3],Q[4],Q[5],Q[6],Q[7],Q[8],Q[9])
        nm,PC = posteriori_matrices(W[j,0],W[j,1],m[0],m[1],m[2],m[3],IQ00,IQ01,IQ02,IQ03,IQ11,IQ12,IQ13,IQ22,IQ23,IQ33 ,sx2,sg2)
        ##### Likelihood
        ll += log_likelihood(W[j,0],W[j,1],m[0],m[1],m[2],m[3],IQ00,IQ01,IQ02,IQ03,IQ11,IQ12,IQ13,IQ22,IQ23,IQ33,PC[0],PC[1],PC[2],PC[3],PC[4],PC[5],PC[6],PC[7],PC[8],PC[9],sx2,sg2)
    # Predict for daughter cell
    m,Q = new_mean_cov(nm[0],nm[1],nm[2],nm[3],PC[0],PC[1],PC[2],PC[3],PC[4],PC[5],PC[6],PC[7],PC[8],PC[9],ml,gl,sl2,mq,gq,sq2,dt,b)
    IQ00,IQ01,IQ02,IQ03,IQ11,IQ12,IQ13,IQ22,IQ23,IQ33 = inverseQ(Q[0],Q[1],Q[2],Q[3],Q[4],Q[5],Q[6],Q[7],Q[8],Q[9])
    # Find next cell initial conditions
    s, IS = cell_division_likelihood_and_grad(m[0],m[1],m[2],m[3],IQ00,IQ01,IQ02,IQ03,IQ11,IQ12,IQ13,IQ22,IQ23,IQ33,sxd2,sgd2)
    # keep the same structure as in recursive_length
    return -ll, None, s, IS
def wrap_obj_and_grad_1cc(W,par_prot,par_len,s,IS,dt):
    """wrapper of obj and grad 1cc"""
    ml,gl,sl2,sx2,sxd2= par_len
    mq,gq,sq2,sg2,sgd2,b = par_prot
    s0,s1,s2,s3 = s
    IS00,IS01,IS02,IS03,IS11,IS12,IS13,IS22,IS23,IS33 = IS
    return  obj_and_grad_1cc(W,ml,gl,sl2,mq,gq,sq2,sx2,sg2,sxd2,sgd2,s0,s1,s2,s3,IS00,IS01,IS02,IS03,IS11,IS12,IS13,IS22,IS23,IS33,dt,b)
def grad_obj_1lane(reind_,dat_,par_prot,par_len,IS,s,dt,nparr=False):
    """Compute the ll and gradient for 1 lane"""
    reind = deepcopy(reind_); dat = deepcopy(dat_)
    obj = 0          # total objective and gradient
    for i in range(len(dat)):
        # Every time a cell do not have a "mother" (happen for first cells) intialize to initial conditions 
        if type(dat[i][0])!=tuple:
            dat[i][0] = [s,IS]
        else:
            s,S = dat[i][0] # Unpack initial conditions
        # Find ll over 1 cell cycle for 1 daughter
        tmp = wrap_obj_and_grad_1cc(dat[i][1][0],par_prot,par_len,s=s,IS=IS,dt=dt) # calculate obj,gobj,p0 for one daugter
        obj += tmp[0];  #update obj, gobj
    # give the inital condition to the right cell lane
        if np.isnan(reind[i,0]) == False:
            dat[int(reind[i,0])][0] = tmp[2:] # s,S,grad_S
    #If the second cell exists do the same
        if np.sum(np.isnan(dat[i][1][1]))==0:
            tmp = wrap_obj_and_grad_1cc(dat[i][1][1],par_prot,par_len,s=s,IS=IS,dt=dt) # calculate obj,gobj,p0 for one daugter
            obj += tmp[0];
            if np.isnan(reind[i,1]) == False:
                dat[int(reind[i,1])][0] = tmp[2:]
    #Return obj and gradobj
    if nparr:
        return np.append(np.array([obj]),None)
    else:
        return obj, None
def grad_obj_total(par_prot,par_len,reind_v,dat_v,s,IS,dt,nproc=10):
    """Apply in parallel on all lane ID"""
    p = Pool(nproc)
    fun = lambda x:\
        grad_obj_1lane(x[0],x[1],par_prot,par_len,IS,s,dt,True)
    ret = p.map(fun,zip(reind_v,dat_v))
    ret = np.sum(np.vstack(ret)[:,0],axis=0)#there is no gradient
    return ret,None
def grad_obj_wrap(par_prot,par_len,in_dic):
    reind_v,dat_v,s,IS,dt,lane_ID_v,val_v=\
    in_dic['reind_v'],in_dic['dat_v'],in_dic['s'],in_dic['IS'],in_dic['dt'],in_dic['lane_ID_v'],in_dic['val_v']
    return grad_obj_total(par_prot,par_len,reind_v,\
                          dat_v,s,IS,dt,nproc=10)
#-------------------PREDICTIONS-------------------------- 
def print_mat(name,a,A):
        print(name,a[0],a[1],a[2],a[3],A[0],A[4],A[7],A[9])
def predictions_1cc(W,ml,gl,sl2,mq,gq,sq2,sx2,sg2,sxd2,sgd2,s0,s1,s2,s3,IS00,IS01,IS02,IS03,IS11,IS12,IS13,IS22,IS23,IS33,dt,b):
    """Return optimal vector z and std over 1 cc"""
    from IPython.core.debugger import set_trace
    assert W.shape[1]==2
    z = []; err_z=[]
    # p(z_0|D_0)
    nm,PC = posteriori_matrices(W[0,0],W[0,1],s0,s1,s2,s3,IS00,IS01,IS02,IS03,IS11,IS12,IS13,IS22,IS23,IS33,sx2,sg2)
    z.append(nm); err_z.append(np.sqrt(np.array([PC[0],PC[4],PC[7],PC[9]])))
    ##### p(D_0)
    for j in range(1,W.shape[0]):
        #set_trace()
        ###### P(z_{t+dt}|D_t)
        m,Q = new_mean_cov(nm[0],nm[1],nm[2],nm[3],PC[0],PC[1],PC[2],PC[3],PC[4],PC[5],PC[6],PC[7],PC[8],PC[9],ml,gl,sl2,mq,gq,sq2,dt,b)
        ##### P(z_{t+dt}|D_{t+dt}) = N(b',B')
        IQ00,IQ01,IQ02,IQ03,IQ11,IQ12,IQ13,IQ22,IQ23,IQ33 = inverseQ(Q[0],Q[1],Q[2],Q[3],Q[4],Q[5],Q[6],Q[7],Q[8],Q[9])
        #print(IQ00,IQ01,IQ02,IQ03,IQ11,IQ12,IQ13,IQ22,IQ23,IQ33 )
        nm,PC = posteriori_matrices(W[j,0],W[j,1],m[0],m[1],m[2],m[3],IQ00,IQ01,IQ02,IQ03,IQ11,IQ12,IQ13,IQ22,IQ23,IQ33 ,sx2,sg2)
        print_mat('prior',m,Q)
        print('measure',W[j,0],W[j,1],sx2,sg2)
        print_mat('posterior',nm,PC)
        z.append(nm); err_z.append(np.sqrt(np.array([PC[0],PC[4],PC[7],PC[9]])))
        ##### Likelihood
    # Predict for daughter cell
    m,Q = new_mean_cov(nm[0],nm[1],nm[2],nm[3],PC[0],PC[1],PC[2],PC[3],PC[4],PC[5],PC[6],PC[7],PC[8],PC[9],ml,gl,sl2,mq,gq,sq2,dt,b)
    IQ00,IQ01,IQ02,IQ03,IQ11,IQ12,IQ13,IQ22,IQ23,IQ33 = inverseQ(Q[0],Q[1],Q[2],Q[3],Q[4],Q[5],Q[6],Q[7],Q[8],Q[9])
    # Find next cell initial conditions (9% asym div)
    s, IS = cell_division_likelihood_and_grad(m[0],m[1],m[2],m[3],IQ00,IQ01,IQ02,IQ03,IQ11,IQ12,IQ13,IQ22,IQ23,IQ33,sxd2,sgd2)
    return z, err_z,s,IS
def wrap_prediction_1cc(W,par_prot,par_len,s,IS,dt):
    """wrapper of obj and grad 1cc"""
    ml,gl,sl2,sx2,sxd2,= par_len
    mq,gq,sq2,sg2,sgd2,b = par_prot
    s0,s1,s2,s3 = s
    IS00,IS01,IS02,IS03,IS11,IS12,IS13,IS22,IS23,IS33 = IS
    return  predictions_1cc(W,ml,gl,sl2,mq,gq,sq2,sx2,sg2,sxd2,sgd2,s0,s1,s2,s3,IS00,IS01,IS02,IS03,IS11,IS12,IS13,IS22,IS23,IS33,dt,b)
def predictions_1lane(reind_,dat_,par_prot,par_len,IS,s,dt,lane_ID,val):
    """Return best_predictiona and error for every cell in form
    [laneID+id,[z,z_err]]"""
    reind = deepcopy(reind_); dat = deepcopy(dat_)
    ret = [] # Return
    count=0
    for i in range(len(dat)):
        # Every time a cell do not have a "mother" (happen for first cells) intialize to initial conditions
        if type(dat[i][0])!=tuple:
            dat[i][0] = [s,IS]
        else:
            s,IS= dat[i][0] # Unpack initial conditions
        # Find prediction over 1 cell cycle for 1 daughter
        tmp = wrap_prediction_1cc(dat[i][1][0],par_prot,par_len,dt=dt,s=s,IS=IS)
        z = tmp[0]; err_z = tmp[1]
        ret.append([lane_ID+'_'+str(val[i,1]),z,err_z])
        #print(np.mean(np.hstack(z).T[:,0]-dat[i][1][0]))
    # give the inital condition to the right cell lane
        if np.isnan(reind[i,0]) == False:
            dat[int(reind[i,0])][0] = tmp[2:] # s,S,grad_S
    #If the second cell exists do the same
        if np.sum(np.isnan(dat[i][1][1]))==0:
            tmp =\
            wrap_prediction_1cc(dat[i][1][1],par_prot,par_len,dt=dt,s=s,IS=IS)
            z = tmp[0]; err_z = tmp[1]
            ret.append([lane_ID+'_'+str(val[i,2]),z,err_z])
            #print(np.mean(np.hstack(z).T[:,0]-dat[i][1][1]))
            if np.isnan(reind[i,1]) == False:
                dat[int(reind[i,1])][0] = tmp[2:]
        #set_trace()
    #Return obj and gradobj
    return ret

def prediction_total(par_prot,par_len,reind_v,dat_v,s,IS,dt,lane_ID_v,val_v,nproc=10):
    """Apply in parallel on all lane ID and return a np.array"""
    tmp = []
    for k in range(len(reind_v)):
        tmp.append(predictions_1lane(reind_v[k],dat_v[k],par_prot,par_len,IS,s,dt,lane_ID_v[k],val_v[k]))
    # Return a nice behave np array with cell_ID, z[0],z[1],err_z[0],err_z[1]
    foo = []
    for lan in tmp:
        for cid in lan:
            foo.append(np.hstack((np.hstack([cid[0]]*len(cid[1]))[:,None],np.vstack(cid[1]),np.vstack(cid[2]))))
    return np.vstack(foo)

def predict(par_prot,par_len, in_dic):
    """It is just a  = xwrapper to keep nice data structure where min_dic is the dict
    returned by minimize and in dic the one returned by build_data_strucure"""
    return prediction_total(par_prot, par_len,\
                            in_dic['reind_v'],in_dic['dat_v'],in_dic['s'],\
                           in_dic['IS'],in_dic['dt'],in_dic['lane_ID_v'],\
                            in_dic['val_v'])
################################################################################################
############################## DATA TREATEMENT #################################################
################################################################################################
def build_intial_mat(df,leng,prot,b):
    """ Build the intiala matrix s,S and intial gradient grad_S """
    # Computation done in minutes (that's why *60)
    def sleres(t,y):
        """Return slope (lambda), intercept (x0) and residuals (form sm2)"""
        t = t - t.iloc[0]
        r = linregress(t,y)
        assert len(t)>2, "not enough points"
        return r.slope,r.intercept, y-(r.intercept+t*r.slope)
    X = df.groupby('cell').apply(lambda x: sleres(x.time_sec/60.,x['{}'.format(leng)]))
    G = df.groupby('cell').apply(lambda x:\
                                 sleres(x.time_sec[:4]/60.,x['{}'.format(prot)][:4]))
    X = np.vstack(X); G = np.vstack(G)
    lam = X[:,0]; x0 = X[:,1]; res = X[:,2]
    q= G[:,0]; g0 = G[:,1]; res_g = G[:,2]
    res = np.array([k for h in res for k in h])
    res_g = np.array([k for h in res_g for k in h])
    sx2 = np.var(res)
    sg2 = np.var(res_g)
    q = (q+b*g0)/(np.exp(x0.astype(float)))
    ####### They do not depend on the parameters
    mat = np.zeros((4,4)); vec = np.zeros((4,1))
    s = (np.nanmean(x0),np.nanmean(lam),np.nanmean(g0),np.nanmean(q))
    IS =\
    (1./np.nanvar(x0),0,0,0,1/np.nanvar(lam),0,0,1/np.nanvar(g0),0,1/np.nanvar(q))
    return s, IS, sx2, sg2
def build_mat(mother,daugther,dfl,leng,prot):
    """Find the correct structure. It will return [mum_ind,daughter1_ind,daughter2_ind] and [mum_ind,[Len(dau1),Len(daug2)]]"""
    tmp = []; tmp1=[]
    for i,j in zip(mother,daugther):
        which = lambda x:\
            dfl.loc[dfl['id']==j[x]][['{}'.format(leng),'{}'.format(prot)]].values
        if j.shape[0]==1:
            tmp.append([i,j[0],np.nan])
            tmp1.append([i,[which(0),np.nan]])
        else:
            tmp1.append([i,[which(0),which(1)]])
            tmp.append([i,j[0],j[1]])
    return np.vstack(tmp),tmp1
def who_goes_where(val):
    """ Connects the geanology e.g. if val [[19,[20,21]],[20,[22,nan]],[21,[23,nan]] ] it will return [[1,2],..] saying that division of cell 20 (daughter of 19) is at index 1 """
    tmp1 = [np.where(val[:,:1]==k)[0] for k in val [:,1:2]]
    tmp2 = [np.where(val[:,:1]==k)[0] for k in val [:,2:3]]
    fun = lambda x: np.nan if x.size==0 else x
    tmp1 = np.vstack(list(map(fun,tmp1)))
    tmp2 = np.vstack(list(map(fun,tmp2)))
    return np.hstack([tmp1,tmp2])
    return flat(distx0),flat(distlam), flat(distk0)
def asym_div(dat_v,rescale=1):
    """Predict the amount of asymmetric division by considering the 2 daughter must sum up to the total"""
    from scipy.stats import linregress
    lg = lambda x: linregress(range(len(x)),x.reshape(-1)).intercept
    flat = lambda x: np.array([g for m in x for g in m ])
    sxd2 =[]; sgd2 = []
    for j in dat_v:
        for k in j:
            if type(k[1][1])==float and (np.isnan(k[1][1])): 
                continue
            else:
                # Predict the length and gfp at begin

                x1 = k[1][0][:,0]
                g1 = k[1][0][:4,1]
                x2 = k[1][1][:,0]
                g2 = k[1][1][:4,1]
                x1,x2,g1,g2 = [lg(x) for x in [x1,x2,g1,g2]]
                perfx = (x1+x2)-rescale*np.log(2) # perfect division
                perfg = (g1+g2)/2 # perfect division
                sxd2.append((perfx-x1,perfx-x2))
                sgd2.append((perfg-g1,perfg-g2))
    return np.var(flat(sxd2)),np.var(flat(sgd2))
#
def build_data_strucutre(df,leng,prot,b):
    """Return for every lane the data with respective daughteres and initial\
    conditions, b=beta"""
    #Sometimes cells with 1 data point are present and we don't want them
    df = df.groupby('cell').filter(lambda x: x.values.shape[0]>1) #
    dt = np.diff(np.sort(df['time_sec'].unique()))[1]/60
    #assert dt%3==0 and dt != 0., "look if dt make sense"
    #rescae
    df['log_resc_'+leng] = np.log(df['{}'.format(leng)])
    #print("The variable to use is: log_resc_{}".format(leng))
    s,IS,sx2,sg2= build_intial_mat(df,'log_resc_'+leng,prot,b)
	#n_point = df.shape[0]
    n_point = df.shape[0]
    dat_v = []; reind_v = [];
    val_v = []; lane_ID_v = []
    for lid in df['lane_ID'].unique():
        dfl = df.loc[df['lane_ID']==lid] # select one lande id
        daugther = [dfl.loc[dfl['parent_id']==k].id.unique() for k in dfl.parent_id.unique()] # find its daughters
        mothers = dfl.parent_id.unique() # respective mothers
        # Correct data strucutre
        val, dat = build_mat(mothers,daugther,dfl,'log_resc_'+leng,prot)
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
    vec_dat_v = np.vstack(vec_dat_v)
    sdx2, sdg2  = asym_div(dat_v)
    #asym division equal to 0.1 cv(0.1*rescale*np.log(2))**2
    return df,{'n_point':n_point,'dt':dt,'s':s,'IS':IS,\
            'reind_v':reind_v,'dat_v':dat_v, 'val_v':val_v,\
               'lane_ID_v':lane_ID_v,'sx2':sx2,\
               'vec_dat_v':vec_dat_v,'sdg2':sdg2,'sdx2':sdx2,'sg2':sg2}
#
def merge_df_pred(df,pred_mat):
    """Merge the output from predict with the initial dataframe"""
    # From mat to dataframe
    dft = pd.DataFrame(pred_mat,columns=\
                       ('cell_','pred_log_length','pred_growth_rate','pred_proteins','pred_q',\
                        'err_log_length','err_growth_rate','err_prot','err_q'))
    # Give numerical values
    dft[['pred_log_length','pred_growth_rate','pred_proteins','pred_q','err_log_length','err_growth_rate','err_prot','err_q']] = \
    dft[['pred_log_length','pred_growth_rate','pred_proteins','pred_q','err_log_length','err_growth_rate','err_prot','err_q']].apply\
            (lambda x:pd.to_numeric(x,errors='coerce'))
    #Create subindex for merging
    dft['sub_ind'] = dft.groupby('cell_')['pred_log_length'].transform(lambda x:\
                                                               np.arange(len(x)))
    # Create same indexing in df
    df['cell_'] = df['lane_ID']+df['id'].apply(lambda x: '_'+str(int(float(x)))+'.0')
    df['sub_ind'] = df.groupby('cell_')['time_sec'].transform(lambda x:\
                                                              np.arange(len(x)))
    #Concat, reindex, delete column used for mergin 
    dff = \
    pd.concat([df.set_index(['cell_','sub_ind']),dft.set_index(['cell_','sub_ind'])],axis=1)
    dff = dff.reset_index()
    return dff.drop(['cell_','sub_ind'],axis=1),dft
def find_best_lengths(files,pwd='/scicore/home/nimwegen/fiori/MoMA_predictions/predictions/',lengths=['length_um','length_erik','length_moma','length_raw','length_pixel']):
    """Find best length measurment """
    tmp=[]
    for k in lengths:
        df = pd.read_csv(pwd+files+'_'+k+'.csv')
        err = np.mean(np.abs(df['err_growth_rate']/df['pred_growth_rate']))
        tmp.append([files,k,err])
    tmp = np.vstack(tmp)
    return tmp[tmp[:,2]==min(tmp[:,2])]
def denoised_dataset(df,step,nump=3):
    """Try to obtain a dataset without noise by sampling every <<step>> """
    # At least nump cell per cell
    df = df.groupby('cell').filter(lambda x: True if len(x['time_sec'])>nump*step else False)
    df = df.reset_index()
    ret = []
    for tau in range(step):
        tmp = (np.sort(df['time_sec'].unique()[tau:])[::step])
        dfsh = pd.concat([df.loc[df['time_sec']==tmp[k]] for k in range(tmp.shape[0])])
        dfsh = dfsh.sort_index().reset_index()
        ret.append(dfsh)
    return ret
################################################################################################
############################### ADDITIONAL FUNCTIONS FOR ANALYSIS  #############################
################################################################################################
def connect_and_filt_df(files,dt,pwd='/scicore/home/nimwegen/rocasu25/Documents/Projects/biozentrum/MoM_constitExpr/20190612_forAthos/'):
    """Connect all pandas in files.txt togetehr"""
    def sleres(y,dt=dt):
        """Return slope (lambda), intercept (x0) and residuals (form sm2)"""
        t = np.arange(0,len(y)*dt,dt)
        r = linregress(t,y)
        return r.rvalue
    tmp = []
    for j in files:
        dtm = pd.read_csv(pwd+j+'/'+j+'.csv')
        dtm['date'] = j
        dtm['cell'] = dtm['date']+dtm['pos'].apply(lambda x: str(x))+dtm['gl'].apply(lambda x: str(x))+dtm['id'].apply(lambda x: str(x))
        dtm['lane_ID'] = dtm['date']+dtm['pos'].apply(lambda x: str(x))+dtm['gl'].apply(lambda x: str(x))
        tmp.append(dtm)
    df = pd.concat(tmp)
    assert np.sum(df['discard_top'])==0
    assert np.sum(np.sum(df['end_type']!='div'))==0
    df = df.groupby('cell').filter(lambda x: len(x['time_sec'])>2)
    df = df.groupby('cell').filter(lambda x:\
                                   sleres(np.log(x.length_um))>0.98)
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
    """find correlation parameters """
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
def long_corr_in_data(df,dt):
    """Find corr over generations in dataframes"""
    def sleres(y,dt=dt):
        """Return slope (lambda), intercept (x0) and residuals (form sm2)"""
        t = np.arange(0,dt*len(y),dt) 
        r = linregress(t,y)
        return r.slope
    df = genalogy(df,'parent_cell')
    df = genalogy(df,'g_parent_cell')
    df = genalogy(df,'g_g_parent_cell')
    df = genalogy(df,'g_g_g_parent_cell') 
    elrat = df.groupby('cell').apply(lambda x: sleres(np.log(x.length_um)))
    return np.vstack([corr_par(df,elrat,par_deg=k) for k in ['parent_cell','g_parent_cell','g_g_parent_cell','g_g_g_parent_cell','g_g_g_g_parent_cell'] ])    
def smilar_frame(W,cvd=0):
    """From  OU create a dataframe with same shape as biological data (divison at twice the size).. Consider also asymmetric division """
    explen = []; lane_ID=[]; parent_ID=[]; id_n=[-1]; time_sec=[]; df=[]
    W = deepcopy(W); 
    for i in range(W.shape[0]):
        tmp = [W[i,0]]
        fix = W[i,0]
        ts=[0]
        for k in range(1,W.shape[1]-1):
            #if tmp[-1] < tmp[0]+np.log(2):
            #Do the choice stocastically otherwise we accumulate always some lengths
            #print([W[i,k]< tmp[0]+np.log(2),W[i,k+1]< tmp[0]+np.log(2)])
            div = np.random.normal(np.log(2),np.log(2)*cvd)
            if np.random.choice([W[i,k]< fix+div,W[i,k+1]< fix+div],1,p=[0.5,0.5])[0]:
                tmp.append(W[i,k])
                ts.append(ts[-1]+180)
            else:
                lane_ID = ['lane_num_{}'.format(i)]*int(len(tmp))
                parent_ID = ['{}'.format(id_n[-1])]*int(len(tmp))
                id_n = ['{}'.format(k)]*int(len(tmp))
                explen= np.exp(tmp)
                time_sec = ts
                ts=[ts[-1]]         
                tmp = [tmp[-1]-div]
                W[i,:] = W[i,:]-div
                #print(len(explen))
                #print(len(lane_ID))
                df.append(pd.DataFrame({'leng':explen,'lane_ID':lane_ID,'parent_id':parent_ID, 'id':id_n,'time_sec':time_sec}))
    df = pd.concat(df,ignore_index=True)
    df['cell'] = df['lane_ID']+'_'+df['id']
    return df
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
################################################################################################
############################### The stocastics models  #########################################
################################################################################################
def ornstein_uhlenbeck(mlam,gamma,sl2,length_min=80,ncel=10,dt=.5):
    """Generate OU process with dtsim (cannot be too large otherwise non stable
    solutions). Then give back results at every dt"""
    np.random.seed()
    length = length_min/dt
    assert (length).is_integer(), "not length integer"
    length = int(length)
    mat = np.zeros((ncel,length))
    sig = np.sqrt(sl2)
    dW = np.random.normal(loc=mat,scale=np.sqrt(dt))
    add = sig*dW*dt
    mat[:,0]=add[:,0]+mlam
    for k in range(1,length):
        mat[:,k]=mat[:,k-1]-gamma*(mat[:,k-1]-mlam)*dt+add[:,k]
    return mat
def log_length_synth(L,sx2,dt,X0=0):
    """L is the OU for log-length production whereas sx2 the measurment error\
    on lenght. Return log_loength with noise X, without noise Xnn and underlyling OU process"""
    Xnn = X0 + np.cumsum(L,axis=1)*dt 
    return np.random.normal(loc=Xnn,scale=np.sqrt(sx2)),Xnn, L
def protein_prod(Q,Xnn,G0,beta,sg2,dt):
    """Q is the OU of protein prod, Xnn the not measurment noise log_loength,
    sg2 error on protein measurment and beta the bleaching """
    G = np.zeros_like(Xnn)
    G[:,0] = G0
    for k in range(1,Xnn.shape[1]):
        G[:,k] = G[:,k-1]+np.exp(Xnn[:,k-1])*Q[:,k-1]*dt-beta*G[:,k-1]*dt
    return np.random.normal(loc=G,scale=np.sqrt(sg2)),G
def sampling_from_path(mat,dtsim,dt):
    """dtsim is the time interval used to generate the data whereas dt is the\
    one where we have the data (like the microscope one)"""
    sam = dt/dtsim
    assert (sam).is_integer(), "not sam integer"
    sam = int(sam)
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
    W,X = integrated_ou(mlam=1.,gamma=0.02,sig=.03,sigm2=0.08,length=30000,ncel=1,dt=3.)
    m0 = np.array([[np.mean(W[:,0])],[np.mean(X[:,0])]])
    M0 = np.array([[np.var(W[:,0]),0],[0,np.var(X[:,0])]])
    t1 = time.time()
    #print(obj_and_grad_1cc(W,mlam=1.,gamma=0.02,sl2=.03**2,sm2=0.08,dt=3.,m0=m0,M0=M0,rescale=rescale))
    print(time.time()-t1)
    #z, err_z,_,_ = predictions_1cc(W,mlam=1.,gamma=0.02,sig_l_s=.03**4,sig_m_s=0.08,dt=3.,m0=m0,M0=M0)
    #print(W_er(1),W_er(2))
    print(X_er(1),X_er(2))
