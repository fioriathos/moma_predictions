import numpy as np
import pandas as pd


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
    X = np.array(X)
    Y = np.array(Y)
    ind = np.array([j.shape[0]-h.shape[0] for j,h in zip(Y,X)])==0
    return Y[ind],X[ind]
def gfp_dynamics_1cc(F,V,Cm1,beta,dtsim):
    """Knowing the concentration before division Cm1, mRNA dynamics F and Volume dynamics V compute GFP dynamics. Condition is concentration stay constant at division"""
    G = np.zeros_like(F)
    G[0] = Cm1*V[0]
    for k in range(len(F)-1):
        G[k+1] = G[k]+(V[k]*F[k]-beta*G[k])*dtsim
    return G,G[-1]/V[-1]
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
similar_frame(ml,gl,sl2,sm2,mq,gq,sq2,DV,beta,shape=(1,1000),dtsim=1,gfp_sym=True):
    """From  OU create a dataframe with same shape as biological data (divison at twice the size).. Consider also asymmetric division """
    if shape[0]>10: print('change naming of indexing cells since they may mix')
    explen = []; lane_ID=[]; parent_ID=[]; id_n=[-1]; time_sec=[]; df=[]
    W = ornstein_uhlenbeck(ml,gl,sl2,shape,dtsim) # create elongation rate dynamics
    if gfp_sym :F = ornstein_uhlenbeck(mq,gq,sq2,shape,dtsim) # create elongation rate dynamics
    V = np.zeros_like(W)
    V0=DV
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
                gfp = np.random.normal(G[j],np.sqrt(G[j]))
                df.append(pd.DataFrame({'leng':explen,'lane_ID':lane_ID,'parent_id':parent_ID,\
                                        'id':id_n,'gfp':gfp,'leng_no_noise':k,'growth_rate':Ww[j],'q_dyn':f[j]}))
            else:
                #print("ok")
#                print(len(explen)-len(Ww[j]))
                df.append(pd.DataFrame({'leng':explen,'lane_ID':lane_ID,'parent_id':parent_ID,\
                                        'id':id_n,'leng_no_noise':k,'growth_rate':Ww[j]}))
    df = pd.concat(df,ignore_index=True)
    df['time_sec'] = np.arange(0,df.shape[0]*dtsim,dtsim)*60
    df['cell'] = df['lane_ID']+df['id']
    return df
#


