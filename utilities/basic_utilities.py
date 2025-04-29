
import pandas as pd
import math
import numpy as np
import warnings
import pdb





def sort_vars(df, varlist):
    numeric_vars = []
    str_vars = []
    oth_vars = []
    
    for v in varlist:
        if df[v].dtype in (int, float, bool):
            numeric_vars.append(v)
        elif df[v].dtype == object:
            str_vars.append(v)
        else:
            oth_vars.append(v)
    return numeric_vars, str_vars, oth_vars



def precision_and_scale(x):
    max_digits = 14
    int_part = int(abs(x))
    magnitude = 1 if int_part == 0 else int(math.log10(int_part)) + 1
    if magnitude >= max_digits:
        return (magnitude, 0)
    frac_part = abs(x) - int_part
    multiplier = 10 ** (max_digits - magnitude)
    frac_digits = multiplier + int(multiplier * frac_part + 0.5)
    while frac_digits % 10 == 0:
        frac_digits /= 10
    scale = int(math.log10(frac_digits))
    return (magnitude, scale)



def binning_q(df, var_bins):
    df1 = df
    cutpoints = {}
    for k in var_bins.keys():
        
        if df1[k].dtype in (int,float): 
            int_len, d_len = precision_and_scale(df1[k].values[0])
            if int_len >= 2:
                
                df1['{}_bin'.format(k)], cutpoints[k] = pd.qcut(df1[k], q=var_bins[k], retbins=True, precision=0)    
            else:
                df1['{}_bin'.format(k)], cutpoints[k] = pd.qcut(df1[k], q=var_bins[k], retbins=True, precision=1+d_len) 
            
            df1['{}_bin'.format(k)] = df1['{}_bin'.format(k)].astype('str')
        else:
            warnings.warn("Skipping variable {}, not numeric...".format(k))
            
    return df1, cutpoints




def binning_c(df, var_bins):
    df1 = df
    cutpoints = {}
    for k in var_bins.keys():
        if df1[k].dtype in (int,float): 
            int_len, d_len = precision_and_scale(df1[k].values[0])
            if int_len >= 2:
                df1['{}_bin'.format(k)], cutpoints[k] = pd.cut(df1[k], bins=var_bins[k], retbins=True, precision=0)    
            else:
                df1['{}_bin'.format(k)], cutpoints[k] = pd.cut(df1[k], bins=var_bins[k], retbins=True, precision=1+d_len)   
            df1['{}_bin'.format(k)] = df1['{}_bin'.format(k)].astype('str')   
        else:
            warnings.warn("Skipping variable {}, not numeric...".format(k))
    return df1, cutpoints


def proc_means(df, myvars):
    df1 = df[myvars].describe(percentiles=[.01, .05, .1, .25, .5, .75, .9, .95, .99]).reset_index()
    df1 = df1.rename(columns={'index':'Statistics'})
    row = pd.DataFrame(columns = df1.columns.to_list())
    
    row.loc[0, 'Statistics'] = 'missing_count'
    for var in myvars:
        row.loc[0, var] = df1[var].isna().sum()
    
    df1 = pd.concat([df1,row], ignore_index=True)
    
    row.loc[0,'Statistics'] = 'missing_pct'
    for var in myvars:
        row.loc[0,var] = df1[var].isna().sum()/df1.loc[df1['Statistics']=='count', var][0]*100
    
    df1 = pd.concat([df1, row], ignore_index=True)
    
    df1['Statistics'] = df1['Statistics'].astype('str')
    return df1


def proc_freq(df, vars):
    df_list = list()
    df0 = df.copy(deep=True)
    df0[vars] = df0[vars].astype('category')
    for var in vars:
        df1 = df0.groupby(var)[var].count()
        df1 = pd.DataFrame(df1).rename(columns={var:'Frequency'}).reset_index()

        r_data = [['missing', df[var].isna().sum()]]
        row = pd.DataFrame(r_data, columns=[var,'Frequency'])
        df1 = df1.append(row, ignore_index=True)
        total = df.shape[0]
        df1['Percent'] = df1['Frequency']/total*100
        df1['Cum Frequency'] = df1['Frequency'].cumsum()
        df1['Cum Percent'] = df1['Percent'].cumsum()
        df_list.append(df1)
    return df_list


def catsum_val(df, vars, val, desc_val):
    df_list = list()
    for var in vars:
        df1 = df.loc[df[var].notna()==True, :].groupby(var)[val].sum()
        df1 = pd.DataFrame(df1).rename(columns={val:"{}".format(desc_val)}).reset_index()
        r_data = [['missing', df.loc[df[var].isna()==True, val].sum()]]
        row = pd.DataFrame(r_data, columns=[var,"{}".format(desc_val)])
        df1 = df1.append(row, ignore_index=True)
        total = df[val].sum()
        df1["{} Pct".format(desc_val)] = df1['{}'.format(desc_val)]/total*100
        df1["Cum {}".format(desc_val)] = df1["{}".format(dscc_val)].cumsum()
        df1["Cum {} Pct".format(desc_val)] = df1["{} Pct".format(desc_val)].cumsum()
        df_list.append(df1)
    return df_list
    


def cat_wt_avg(df, byvars, wt, val, desc_val):
    df_list = list()
    df['wt_sum'] = df[wt]*df[val]
    for var in byvars:
        df1 = df.loc[df[var].notna()==True, :].groupby(var)[wt].sum()
        df2 = df.loc[df[var].notna()==True, :].groupby(var)['wt_sum'].sum()
        df1 = pd.DataFrame(df1).reset_index()
        df2 = pd.DataFrame(df2).rename(columns={'wt_sum':'Wted sum of {}'.format(desc_val)}).reset_index()
        
        r_data1 = [['missing', df.loc[df[var].isna() == True, wt].sum()]]
        r_data2 = [['missing', df.loc[df[var].isna() == True, 'wt_sum'].sum()]]
        row1 = pd.DataFrame(r_data1, columns=[var, wt])
        df1 = df1.append(row1, ignore_index=True)
        row2 = pd.DataFrame(r_data2, columns=[var, 'Wted sum of {}'.format(desc_val)])
                
        df2 = df2.append(row2, ignore_index=True)
        df1 = df1.merge(df2, on=var)
        total = df[wt].sum()
        df1["{} Pct".format(wt)] = df1["{}".format(wt)]/total*100
        
        df1["Weighted {}".format(val)] = df1.apply(lambda x: 0 if x[wt]==0 else x["Wted sum of {}".format(desc_val)]/x[wt], axis=1)
        
        df1 = df1[[var, wt, "{} Pct".format(wt), "Weighted {}".format(val), "Wted sum of {}".format(desc_val)]]
        df1 = df1.sort_values(by=["Weighted {}".format(val)], ascending=False)
        
        df_list.append(df1)
    df = df.drop('wt_sum', axis=1)
    return df_list



def write_tbles(dflist, writer, sheetname, start_row, start_col):
    for df in dflist:
        df.to_excel(writer, index=False, sheet_name = sheetname, \
            startrow = start_row, startcol = start_col)
        start_row += (2 + df.shape[0])
    
    return start_row, start_col + 1 + df.shape[1]



def pivot(df, varlist, by_var):
    df_grp = df.groupby(by_var)[df.columns[0]].count().reset_index()
    df_grp = df_grp.rename(columns={df.columns[0]:'count'})

    for k in varlist.keys():
        
        pair = varlist[k]  
        
        if len(pair) == 2:
            agg = pair[0]
            weight = pair[1]
            df1 = df[[by_var,k]].copy(deep=True)
            if weight == '1':
                df1[weight] = 1
        else:
            agg = pair
            df1 = df[[by_var,k]].copy(deep=True)
            
        if agg == 'weighted_avg':
            df1['wt'] = df1[weight]*df1[k]
            grp1 = df1.groupby(by_var)['wt'].agg([np.sum]).reset_index()
            grp2 = df1.groupby(by_var)[weight].agg([np.sum]).reset_index()
            grp2 = grp2.rename(columns={'sum':k})
            grp = grp1.merge(grp2, on=by_var)
            grp['WA_{}'.format(k)] = grp['sum']/grp[k]
            grp = grp[[by_var, 'WA_{}'.format(k)]]
        elif agg == 'sum':
            grp = df1.groupby(by_var)[k].agg([np.sum]).reset_index()
            grp = grp.rename(columns={'sum':'sum_{}'.format(k)})
        elif agg == 'mean':
            grp = df1.groupby(by_var)[k].agg([np.mean]).reset_index()
            grp = grp.rename(columns={'mean':'mean_{}'.format(k)})
        elif agg == 'min':
            grp = df1.groupby(by_var)[k].agg([np.min]).reset_index()
            grp = grp.rename(columns={'min':'min_{}'.format(k)})
        elif agg == 'max':
            grp = df1.groupby(by_var)[k].agg([np.max]).reset_index()
            grp = grp.rename(columns={'max':'max_{}'.format(k)})
        elif agg == 'median':
            grp = df1.groupby(by_var)[k].agg([np.median]).reset_index()
            grp = grp.rename(columns={'median':'med_{}'.format(k)})
        elif agg == 'std':
            grp = df1.groupby(by_var)[k].agg([np.std]).reset_index()
            grp = grp.rename(columns={'std':'std_{}'.format(k)})
        elif agg == 'logodds':
            grp1 = df1.groupby(by_var)[k].agg([np.sum]).reset_index()
            grp2 = df_grp[[by_var,'count']]
            grp  = grp1.merge(grp2, on=by_var)
            grp['ones'] = grp['sum']
            grp['zeros'] = grp['count'] - grp['sum']
            grp['logodds_{}'.format(k)]=np.log(grp['ones']/grp['zeros'])
            grp = grp[[by_var,'logodds_{}'.format(k)]]
        else:
            warnings.warn("Skipping variable {}, invalid aggregation...".format(k))
            continue

        int_l, dec_l = precision_and_scale(min(df[k]))
        if agg in ['sum','mean','min','max','median','std']:
            grp = grp.round({grp.columns[1]:dec_l})
        df_grp = df_grp.merge(grp, on=by_var)
    
    return df_grp
            
            
        



