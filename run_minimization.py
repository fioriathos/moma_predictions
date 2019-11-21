import minimize_lengths as mn
import recursive_lengths as rl
import pandas as pd
import numpy as np
from numba import jit,prange
@jit(parallel=True)
def compute_different_tau(in_dic):
    ret = []
    taus =  [tau for tau in range(3,180,18)]
    for i in prange(len(taus)):
        tau = taus[i]
        mod = mn.minimize_lengths(free={'sm2':sms,'sl2':0.27},fixed={'gamma':1./tau,'m_lam':s[1,0]})
        tmp = mod.minimize(in_dic = in_dic)
        print(tmp[-1])
    return 
if __name__=="__main__":
    df = pd.read_csv('/scicore/home/nimwegen/rocasu25/Documents/Projects/biozentrum/MoM_constitExpr/20190612_forAthos/myframes_chr_rpmB_glucose_20190515/myframes_chr_rpmB_glucose_20190515.csv')
    df['res_log_length_um'] = np.log(df['length_um'])*1e02
    in_dic = rl.build_data_strucutre(df,'res_log_length_um')
    s,S,_,sms = rl.build_intial_mat(df,'res_log_length_um')
    compute_different_tau(in_dic)

