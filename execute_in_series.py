import sys
import numpy as np
import pandas as pd
import recursive_lengths as rl
import minimize_lengths as mn
from scipy.stats import linregress
def load(filen):
    dfo = pd.read_csv('/scicore/home/nimwegen/rocasu25/Documents/Projects/biozentrum/MoM_constitExpr/20190612_forAthos/'+filen+'/'+filen+'.csv')
    dfo = dfo.groupby('cell').filter(lambda x: x.values.shape[0]>1)
    return dfo
def find_param(filen,leng,resc,rsq,const=False):
    dfo = pd.read_csv('/scicore/home/nimwegen/rocasu25/Documents/Projects/biozentrum/MoM_constitExpr/20190612_forAthos/'+filen+'/'+filen+'.csv')
    dfo = dfo.groupby('cell').filter(lambda x: x.values.shape[0]>1)
    #filter out bad cells
    dfo = dfo.groupby('cell').filter(lambda x:\
                                     linregress(x.time_sec,np.log(x['{}'.format(leng)]))[2]>rsq)
    df,in_dic = rl.build_data_strucutre(dfo,leng,resc)
    # Set sl2 such taht cv2=0.25
    mod = \
    mn.minimize_lengths(free={'gamma':0.0125,'sl2':0.012*2*in_dic['s'][1,0]**2*0.25,'sm2':in_dic['sm2'],'m_lam':in_dic['s'][1,0]},fixed={})
    mod.boundary[1] = (1e-13,1.)
    if const:
        mod.boundary[1] = (1e-13,1./40)
    return mod.minimize(in_dic = in_dic)
def predict(filen,min_dic,leng,resc):
    dfo =\
    pd.read_csv('/scicore/home/nimwegen/rocasu25/Documents/Projects/biozentrum/MoM_constitExpr/20190612_forAthos/'+filen+'/'+filen+'.csv')
    dfo = dfo.groupby('cell').filter(lambda x: x.values.shape[0]>1)
    df,in_dic = rl.build_data_strucutre(dfo,leng,resc)
    pred_mat = rl.predict(min_dic,in_dic)
    return rl.merge_df_pred(df,pred_mat)
if __name__=='__main__':
    import pickle
    file_name = sys.argv[1]
    length_type = sys.argv[2]
    min_dic = find_param(file_name,length_type,resc=100,rsq=0.99)
    min_dic['fname'] = file_name
    min_dic['length_type'] = length_type
    #with\
    #open('/scicore/home/nimwegen/fiori/MoMA_predictions/predictions/inferences.pkl')\
    #        as f:
    #    pickle.dump(min_dic,f)
    print(file_name,length_type,min_dic)
    df = predict(file_name,min_dic,leng=length_type,resc=100)
    df.to_csv('/scicore/home/nimwegen/fiori/MoMA_predictions/predictions/'+file_name+'_'+length_type+'.csv')
