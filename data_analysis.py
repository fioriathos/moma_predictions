import numpy as np
import pandas as pd
import recursive_lengths as rl
import seaborn as sns
from scipy.stats import linregress
from scipy import stats
import matplotlib.pyplot as plt
##### Some functions useful for data analysis
def load_and_filer(pwd,rval=0.95):
    """load csv filter div and 0.95 r value and at least 2 clls"""
    df = pd.read_csv(pwd)
    df = rl.give_good_structure(df)
    df = df.loc[(df['end_type']=='DIVISION')|(df['end_type']=='DIV')|(df['end_type']=='div')]
    if 'length_box' in df.columns: #guillaume data
        df['time_sec'] = df['frame']*60*3
        df['length_box_um'] = df['length_box']*0.065
    else:
        df['length_box_um'] = (df['vertical_bottom'] - df['vertical_top'])*0.065
    df = df.groupby('cell').filter(lambda x: True if len(x['length_box_um'])>2 else False)
    df =df.groupby('cell').filter(lambda x: linregress(x['time_sec'],np.log(x['length_box_um'])).rvalue>rval)
    #df = rl.give_unique_dataset(df,6,18)
    df =df[['length_box_um','time_sec','parent_id','id','gl','date','pos','cell','lane_ID','end_type']]
    return df
def predict(df,x):
    """Predict length and growht rate using inferred parameters"""
    ml,g,sl2,sm2,sd2= x
    _,in_dic = rl.build_data_strucutre(df,'length_box_um',1)
    pred_mat = rl.prediction_total(ml,g,sl2,sm2,in_dic['reind_v'],in_dic['dat_v'],in_dic['s'],in_dic['S'],in_dic['dt'],in_dic['lane_ID_v'],in_dic['val_v'],1,sd2,nproc=10)
    df = rl.merge_df_pred(df,pred_mat)
    return df
def panda_files_descriptions(df):
    """Some simple statistics of the panda file"""
    division_time = (df.groupby('cell')['time_sec'].last()-df.groupby('cell')['time_sec'].first()).mean()/60
    number_cells = len(df.cell.unique())
    corr = rl.long_corr_in_data(df,'length_box_um',2)
    inilen = df.groupby('cell')['length_box_um'].first().mean()
    divlen = df.groupby('cell')['length_box_um'].last().mean()
    elrat = df.groupby('cell').apply(lambda x:\
                                     linregress(x['time_sec'],np.log(x['length_box_um'])).slope).mean()*60
    ac_ti =\
    (df.groupby('cell')['time_sec'].nth(1)-df.groupby('cell')['time_sec'].nth(0)).mean()/60
    return {'ac_ti':ac_ti,'length_before_division':divlen,'length_after_division':inilen,\
            'cv_el_rat':corr[1],'corr_1_gen':corr[0][0,0],'corr_2_gen':corr[0][1,0],'division_time':division_time,'number_cells':number_cells,'elrat':elrat}
def usefulquantities(dffin):
    """Some interesting variable to define on dffin (the already predicted
    paths) """
    dffin['log_length_box'] = np.log(dffin['length_box_um'])
    dffin['time_min']=dffin['time_sec']/60
    dffin['pred_length_box_um'] = np.exp(dffin['pred_log_length'])
    dffin['unique_id'] = dffin['cell']+dffin['time_sec'].apply(lambda x:str(x))
    dffin['cv_gr']= dffin.groupby('cell')['pred_growth_rate'].transform(lambda x:\
                                                                  np.std(x)/np.mean(x))
    dffin['std_gr']= dffin.groupby('cell')['pred_growth_rate'].transform(lambda x: np.std(x))
    dffin['mean_gr'] = dffin.groupby('cell')['pred_growth_rate'].transform(lambda x: np.mean(x))
    dffin['mean_len'] = dffin.groupby('cell')['pred_length_box_um'].transform(lambda x: np.mean(x))
    dffin['norm_pred_growth_rate'] = (dffin['pred_growth_rate']-dffin.groupby('cell')['pred_growth_rate'].transform(lambda\
    x: np.mean(x)))/dffin.groupby('cell')['pred_growth_rate'].transform(lambda x: np.mean(x))
    dffin = rl.genalogy(dffin,'parent_cell') #Create genealogy
    dffin = rl.genalogy(dffin,'g_parent_cell')
    dffin = rl.genalogy(dffin,'g_g_parent_cell')
    dffin = dffin.set_index('unique_id')
    qq= dffin.groupby('cell').apply(lambda x: (x['pred_length_box_um']-x['pred_length_box_um'].iloc[0])/(x['pred_length_box_um'].iloc[-1]-x['pred_length_box_um'].iloc[0])).rename('add_len')
    jj= dffin.groupby('cell').apply(lambda x: (x['time_sec']-x['time_sec'].iloc[0])/(x['time_sec'].iloc[-1]-x['time_sec'].iloc[0])).rename('cell_cycle')
    return pd.concat([dffin, qq.reset_index().set_index('unique_id')['add_len'], jj.reset_index().set_index('unique_id')['cell_cycle']], axis=1, join='inner')

def binnedcorr(df,variable,bins,var_y='pred_growth_rate'):
    """Bin the variable and compute mean+std+number of points for growth rate """
    le = np.linspace(df['{}'.format(variable)].min(),df['{}'.format(variable)].max(),bins+1)
    bins = pd.IntervalIndex.from_tuples([(i,j) for i,j in zip(le[:-1],le[1:])])
    df['bin_var'] = pd.cut(df['{}'.format(variable)], bins)
    gro = df.groupby('bin_var').apply(lambda x: x['{}'.format(var_y)])
    #return gro.mean(),gro.std(),gro.size()
    gro = gro.reset_index()
    gro['bin_var'] =  gro['bin_var'].astype('str')
    gro['bin_var'] = gro['bin_var'].apply(lambda x:'('+';'.join(['{:.2f}'.format(float(i)) for  i in x.replace('(','').replace('[','').replace(']','').replace(')','').split(",")])+')')
    return  gro[['bin_var','{}'.format(var_y)]].rename(columns={'bin_var':'bin_{}'.format(variable)})
def at_birth(df,variable,npoint):
    """Variable at birth and elongation rate mean over npoint"""
    return df.groupby('cell')[['{}'.format('{}'.format(variable)),'pred_growth_rate']].apply(lambda x: x.head(npoint).mean()).rename(columns={'pred_length_box_um':'{}_at_birth'.format(variable)})
def connect_cells(dfte,vari):
    """Connect cells between genealogies and return dataframe with super_cell id and variable"""
    # Create the variabel cell for mother, grand mother and grand grand mother
    if 'g_parent_cell' not in dfte.columns:
        dfte = rl.genalogy(dfte,'parent_cell') #Create genealogy
    if 'g_g_parent_cell' not in dfte.columns:
        dfte = rl.genalogy(dfte,'g_parent_cell')
    if 'g_g_g_parent_cell' not in dfte.columns:
        dfte = rl.genalogy(dfte,'g_g_parent_cell')
    #give unique index to all cells
    dfte['uid'] = dfte['cell']+dfte['time_sec'].apply(lambda x: str(x))
    vac=[];sc=[];uid = []
    # Create a vecotor for the variable of interest of cell,mother,grand mother and grand grand mother and an unique identifier of it
    for c,idx in enumerate(dfte['cell'].unique()):
        dau = dfte.loc[dfte['cell']==idx]
        pc = dau['parent_cell'].iloc[0]
        mum = dfte.loc[dfte['cell']==pc]
        gpc = dau['g_parent_cell'].iloc[0]
        gmum = dfte.loc[dfte['cell']==gpc]
        ggpc = dau['g_g_parent_cell'].iloc[0]
        ggmum = dfte.loc[dfte['cell']==ggpc]
        gggpc = dau['g_g_g_parent_cell'].iloc[0]
        gggmum = dfte.loc[dfte['cell']==gggpc]
        fte = lambda x: x[['{}'.format(vari),'uid']].values
        tmp = np.vstack([fte(gggmum),fte(ggmum),fte(gmum),fte(mum),fte(dau)])
        vac.append(tmp[:,0])
        uid.append(tmp[:,1])
        sc.append(['super_cell_{}'.format(c)]*len(tmp))
    return pd.DataFrame({'super_cell':np.hstack(sc),'uid':np.hstack(uid),'{}'.format(vari):np.hstack(vac)})
def correlation(dffin,Dt,vari):
    """Correlation function of <vari,vari_dt> """
    if Dt==0:return np.array([1,0])
    rigth = dffin.groupby('super_cell').apply(lambda x: x[['{}'.format(vari),'uid']].values[Dt:,:]).values
    left = dffin.groupby('super_cell').apply(lambda x: x[['{}'.format(vari),'uid']].values[:-Dt,:]).values
    tmp = pd.DataFrame(data=np.hstack((np.vstack(left),np.vstack(rigth))))
    #find unique index
    tmp = tmp.drop_duplicates()
    per_r = ((tmp.loc[:,0]*tmp.loc[:,2]).mean()-tmp.loc[:,0].mean()*tmp.loc[:,2].mean())/(tmp.loc[:,0].std()*tmp.loc[:,2].std())
    err_r = np.sqrt((1-per_r**2)/(tmp.shape[0]-2))
    return np.array([per_r,err_r])
def autocorrelation(df,maxt,step,vari,acquisiton_time,division_time):
    """Find autocorrelation even between genalogy frmo t=0 to t=maxt with step steps"""
    maxt = int(maxt/acquisiton_time)
    step = int(step/acquisiton_time)
    df = connect_cells(df,vari)
    return np.vstack([correlation(df,Dt,vari) for Dt in\
                      np.arange(0,maxt,step)]),\
        np.arange(0,maxt,step)*acquisiton_time/division_time
def muther_dau_gdau(df,var_m,var_d,var_gd):
    """Find the variables between mum daug gdaug"""
    if 'g_parent_cell' not in df.columns:
        df = rl.genalogy(df,'parent_cell') #Create genealogy
    tmp={'gr_mu_{}'.format(var_m):[],'mu_{}'.format(var_d):[],'daugther_{}'.format(var_gd):[]}
    for k in df.cell.unique():
        dau = df.loc[df['cell']==k]
        dau_var = dau['{}'.format(var_gd)].iloc[0]
        nid = dau.parent_cell.iloc[0]
        mu = df.loc[df['cell']==nid]
        try:#if mother exists
            mu_var = mu['{}'.format(var_d)].iloc[0]
            nid = mu.g_parent_cell.iloc[0]
            mu_var = mu['{}'.format(var_d)].iloc[0]
            gmu = df.loc[df['cell']==nid]
        except IndexError:
            continue
        try:# if grand mother exists
            tmp['gr_mu_{}'.format(var_m)].append(gmu['{}'.format(var_m)].iloc[0])
            tmp['mu_{}'.format(var_d)].append(mu_var)
            tmp['daugther_{}'.format(var_gd)].append(dau_var)
        except IndexError:
            tmp['gr_mu_{}'.format(var_m)].append(np.nan)
            tmp['mu_{}'.format(var_d)].append(mu_var)
            tmp['daugther_{}'.format(var_gd)].append(dau_var)
    return pd.DataFrame(tmp)
def qq_plot(obs,var,fname):
    """qq plot with normal dist"""
    plt.figure()
    z = (obs-np.mean(obs))/np.std(obs)
    stats.probplot(z, dist="norm", plot=plt)
    plt.plot(np.arange(-3,3),np.arange(-3,3))
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    plt.title("Normal Q-Q plot {} in {}".format(var,fname))
    plt.savefig("qq_{}".format(var))
def\
plot_corr(df,var_x,var_y,one_per_cell,kind='reg',addsave='',fname='',ret=False):
    """Select 1 number per cell and plot the result"""
    if one_per_cell:
        ff =\
        df.dropna().groupby('cell')[['{}'.format(var_x),'{}'.format(var_y)]].first()
    else: ff=df
    g = sns.jointplot(x="{}".format(var_x), y="{}".format(var_y),\
                      data=ff,kind=kind)
    g.plot_joint(plt.scatter, c="w", s=0.1, marker="o")
    if kind=='reg':
        g.annotate(stats.pearsonr)
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Corr between {} and {} {} for\n {}".format(var_x,var_y,addsave,fname))
    g.savefig("Corr_between_{}_and_{}_{}.png".format(var_x,var_y,addsave),bbox_inches='tight')
    if ret: return ff
def plot_autocorrelation(df,pa_fi_de):
    divt = pa_fi_de['division_time']
    act = pa_fi_de['ac_ti']
    tmp_dat =\
    autocorrelation(df,int(3*divt),step=act*2,vari='pred_growth_rate',acquisiton_time=act,division_time=divt)
    plt.figure()
    plt.plot(tmp_dat[1],tmp_dat[0][:,0])
    plt.xlabel('time (doubling)')
    plt.title("Autocorrelation for growth rate in {}".format(fname))
    plt.savefig("autocorrelation.png")
    tmp_dat = np.hstack([tmp_dat[0],tmp_dat[1][:,None]])
    np.save('autocorrelation_{}'.format(fname),tmp_dat)

if __name__=="__main__":
    import sys
    import json
    import pickle
    file_name = sys.argv[1]
    inf_par = sys.argv[2] #pikle
    fname = file_name.replace('.csv','')
    if sys.argv[3] == 'N':
        df_ = load_and_filer(file_name)
        #Inferred param
        with open(inf_par,'rb') as f:
            param = pickle.load(f)
        df = predict(df_,param['param'])
        df = usefulquantities(df)
        #Description
        print(param)
        df.to_csv(file_name.replace('.csv','')+'_pred.csv')
    else:
        print("Prediction file loaded")
        df = pd.read_csv(file_name.replace('.csv','')+'_pred.csv')
        df['rvalue'] = df.groupby('cell')['length_box_um'].transform(lambda x: linregress(range(len(x)),x).rvalue)
    pa_fi_de  =panda_files_descriptions(df)
#    with open('file_description.txt', 'w') as file:
#             file.write(json.dumps(pa_fi_de)) # use `json.loads` to do the
#    # Is the growth rate normal distributed?
    qq_plot(df['pred_growth_rate'],'growth_rate',fname=fname)
#    # Is the over/under the mean normal distributed?
    qq_plot(df['norm_pred_growth_rate'],'norm_growth_rate',fname=fname)
#    #Autocorrelation function
    plot_autocorrelation(df,pa_fi_de)
    # Who correlates with whom?
#    plot_corr(df,'mean_len','mean_gr',True,fname=fname)
#    plot_corr(df,'mean_len','cv_gr',True,fname=fname)
#    plot_corr(df,'mean_gr','cv_gr',True,fname=fname)
#    plot_corr(df,'pred_length_box_um','norm_pred_growth_rate',True,'reg','at_birth',fname=fname)
#    plot_corr(df,'cell_cycle','norm_pred_growth_rate',False,'scatter',fname=fname)
#    plot_corr(df,'add_len','norm_pred_growth_rate',False,fname=fname)
#    plot_corr(df,'pred_length_box_um','norm_pred_growth_rate',False,fname=fname)
#    #Between muther and daugther
#    tmp_dat = muther_dau_gdau(df,'cv_gr','cv_gr','cv_gr')
#    plot_corr(tmp_dat,'mu_cv_gr','daugther_cv_gr',False,fname=fname)
#    plot_corr(tmp_dat,'gr_mu_cv_gr','daugther_cv_gr',False,fname=fname)
#    tmp_dat = muther_dau_gdau(df,'mean_gr','mean_gr','mean_gr')
#    plot_corr(tmp_dat,'mu_mean_gr','daugther_mean_gr',False,fname=fname)
#    plot_corr(tmp_dat,'gr_mu_mean_gr','daugther_mean_gr',False,fname=fname)
#    plot_corr(df,'mean_gr','rvalue',True,fname=fname)
#    plot_corr(df,'mean_len','rvalue',True,fname=fname)
#    plot_corr(df,'mean_len','cv_gr',True,fname=fname)


