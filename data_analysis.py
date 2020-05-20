##### Some functions useful for data analysis
def load_and_filer(pwd):
    df = pd.read_csv(pwd)
    df = rl.give_good_structure(df)
    df = df.groupby('cell').filter(lambda x: len(x)>2)
    df = df.loc[df['end_type']=='DIVISION']
    df =df.groupby('cell').filter(lambda x: linregress(x['frame'],np.log(x['length_box'])).rvalue>0.95)
    df['time_sec'] = df['frame']*60*3
    df['length_box_um'] = df['length_box']*0.065
    #df = rl.give_unique_dataset(df,6,18)
    df =df[['length_box_um','time_sec','parent_id','id','gl','date','pos','cell','lane_ID','length_box','end_type']]
    return df
def predict(df,x):
    ml,g,sl2,sm2,sd2= x
    _,in_dic = rl.build_data_strucutre(df,'length_box_um',1)
    pred_mat = rl.prediction_total(ml,g,sl2,sm2,in_dic['reind_v'],in_dic['dat_v'],in_dic['s'],in_dic['S'],in_dic['dt'],in_dic['lane_ID_v'],in_dic['val_v'],1,sd2,nproc=10)
    df = rl.merge_df_pred(df,pred_mat)
    return df
def useful_quantities(dffin):
    dffin['log_length_box'] = np.log(dffin['length_box_um'])
    dffin['time_min']=dffin['time_sec']/60
    dffin['cell_cycle'] = dffin.groupby('cell').apply(lambda x: (x['time_sec']-x['time_sec'].iloc[0])/(x['time_sec'].iloc[-1]-x['time_sec'].iloc[0]) ).values
    dffin['pred_length_box_um'] = np.exp(dffin['pred_log_length'])
    dffin['added_length'] = dffin.groupby('cell').apply(lambda x: x['pred_length_box_um']-x['pred_length_box_um'].iloc[0] ).values
    return dffin
def binnedcorr(df,variable,bins):
    """Bin the variable and compute mean+std+number of points for growth rate """
    le = np.linspace(df['{}'.format(variable)].min(),df['{}'.format(variable)].max(),bins+1)
    bins = pd.IntervalIndex.from_tuples([(i,j) for i,j in zip(le[:-1],le[1:])])
    df['bin_var'] = pd.cut(df['{}'.format(variable)], bins)
    gro = df.groupby('bin_var').apply(lambda x: x['pred_growth_rate'])
    #return gro.mean(),gro.std(),gro.size()
    gro = gro.reset_index()
    gro['bin_var'] =  gro['bin_var'].astype('str')
    gro['bin_var'] = gro['bin_var'].apply(lambda x:'('+';'.join(['{:.2f}'.format(float(i)) for  i in x.replace('(','').replace('[','').replace(']','').replace(')','').split(",")])+')')
    return  gro[['bin_var','pred_growth_rate']].rename(columns={'bin_var':'bin_{}'.format(variable)}) 
def at_birth(df,variable,npoint):
    """Variable at birth and elongation rate mean over npoint"""
    return df.groupby('cell')[['{}'.format('{}'.format(variable)),'pred_growth_rate']].apply(lambda x: x.head(npoint).mean()).rename(columns={'pred_length_box_um':'{}_at_birth'.format(variable)})
def autocorrelation(df,maxt,step,vari='pred_growth_rate',acquisiton_time=3):
    """Find autocorrelation even between genalogy frmo t=0 to t=maxt with step steps"""
    def connect_cells(dfte,vari):
        """Connect cells between genealogies and return dataframe with super_cell id and variable"""
        # Create the variabel cell for mother, grand mother and grand grand mother
        if 'g_parent_cell' not in dfte.columns:
            dfte = rl.genalogy(dfte,'parent_cell') #Create genealogy
            dfte = rl.genalogy(dfte,'g_parent_cell')
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
            tmp = np.hstack([dau['{}'.format(vari)].values,mum['{}'.format(vari)].values,gmum['{}'.format(vari)].values,ggmum['{}'.format(vari)].values])
            tmp1 = np.hstack([ggmum['time_sec'].apply(lambda x:str(x)+'_'+str(ggpc)).values,gmum['time_sec'].apply(lambda x:str(x)+'_'+str(gpc)).values,mum['time_sec'].apply(lambda x:str(x)+'_'+str(pc)).values,dau['time_sec'].apply(lambda x:str(x)+'_'+str(idx)).values])
            vac.append(tmp)
            uid.append(tmp1)
            sc.append(['super_cell_{}'.format(c)]*len(tmp))
        return pd.DataFrame({'super_cell':np.hstack(sc),'uid':np.hstack(uid),'{}'.format(vari):np.hstack(vac)})
    def correlation(dffin,Dt,vari):
        """Correlation function of <vari,vari_dt> """
        if Dt==0:return 1,1e10
        rigth = dffin.groupby('super_cell').apply(lambda x: x[['{}'.format(vari),'uid']].values[Dt:,:]).values
        left = dffin.groupby('super_cell').apply(lambda x: x[['{}'.format(vari),'uid']].values[:-Dt,:]).values
        tmp = np.hstack((np.vstack(left),np.vstack(rigth)))
        #find unique index
        #return tmp
        _,ind = np.unique(tmp[:,1:2]+tmp[:,3:4],return_index=True) 
        # Unique index array
        tmp = tmp[ind,:]
        left = tmp[:,0:1]; rigth = tmp[:,2:3]
        sl = np.std(left);sr = np.std(rigth)
        ml = np.mean(left);mr = np.mean(rigth)
        ct = np.mean(left*rigth)
        return (ct-ml*mr)/(sl*sr),left.shape[0]
    df = connect_cells(df,vari)
    #for Dt in np.arange(0,maxt,step):
    #    print(Dt)
    #    print(correlation(df,Dt,vari))
    return np.array([correlation(df,Dt,vari) for Dt in np.arange(0,maxt,step)]), np.arange(0,maxt,step)*acquisiton_time

def plot_figures(df,medium):
    # Cell cycle
    tmp = binnedcorr(df,'cell_cycle',10)
    g = sns.catplot( x="bin_cell_cycle", y="pred_growth_rate",kind='box', data=tmp,height=10)
    g.set_xticklabels(rotation=45)
    g.savefig("{}_cell_cycle.png".format(medium))
    #Length
    tmp = binnedcorr(df,'pred_length_box_um',20)
    g = sns.catplot( x="bin_pred_length_box_um", y="pred_growth_rate",kind='box', data=tmp,height=10)
    g.set_xticklabels(rotation=45)
    g.savefig("{}_length_box_um.png".format(medium))
    dfilt = df.loc[(df['pred_length_box_um']>=df.groupby('cell')['pred_length_box_um'].first().mean())&(df['pred_length_box_um']<=df.groupby('cell')['pred_length_box_um'].last().mean())]
    tmp = binnedcorr(dfilt,'pred_length_box_um',10)
    g = sns.catplot( x="bin_pred_length_box_um", y="pred_growth_rate",kind='box', data=tmp,height=10)
    g.set_xticklabels(rotation=45) 
    g.savefig("{}_length_box_um_zoom.png".format(medium))
    g.set(ylim=(0, 0.015))
    #Added length
    tmp = binnedcorr(df,'added_length',20)
    g = sns.catplot( x="bin_added_length", y="pred_growth_rate",kind='box', data=tmp,height=10)
    g.set_xticklabels(rotation=45)
    g.savefig("{}_added_length.png".format(medium))
    dfilt = df.loc[(df['added_length']>=0)&(df['added_length']<=df['added_length'].mean()+2*df['added_length'].std())]
    tmp = binnedcorr(dfilt,'added_length',10)
    g = sns.catplot( x="bin_added_length", y="pred_growth_rate",kind='box', data=tmp,height=10)
    g.set_xticklabels(rotation=45)
    g.savefig("{}_added_length_zoom.png".format(medium))
    g.set(ylim=(0, 0.015))
    # LEngth at birth
    dfi = at_birth(df,'pred_length_box_um',5)
    tmp = binnedcorr(dfi,'pred_length_box_um_at_birth',20)
    g = sns.catplot( x="bin_pred_length_box_um_at_birth", y="pred_growth_rate",kind='box', data=tmp,height=10)
    g.set_xticklabels(rotation=45)
    g.savefig("{}_length_at_birth.png".format(medium))
    me=dfi['pred_length_box_um_at_birth'].mean()
    st=dfi['pred_length_box_um_at_birth'].std()
    dfilt = dfi.loc[(dfi['pred_length_box_um_at_birth']>=me-2*st)&(dfi['pred_length_box_um_at_birth']<=me+2*st)]
    tmp = binnedcorr(dfilt,'pred_length_box_um_at_birth',10)
    g = sns.catplot( x="bin_pred_length_box_um_at_birth", y="pred_growth_rate",kind='box', data=tmp,height=10)
    g.set_xticklabels(rotation=45)
    g.savefig("{}_length_at_birth_zoom.png".format(medium))




