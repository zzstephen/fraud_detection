#authored by stephen Zhou

import pandas as pd
import math
import numpy as np
import warnings
import pdb
from scipy.stats import ttest_ind
import yaml
import csv
import json


def split_file(input_filepath:str, output_fd:str, chunk_size:int, output_prefix:str="chunk"):
    """
    Splits a large text file into smaller files of a specified size.

    Args:
        input_filepath (str): Path to the input text file.
        output_fd (str): Path to save the chunk files
        chunk_size (int): Maximum number of lines per chunk.
        output_prefix (str, optional): Prefix for output file names. Defaults to "chunk".
    """
    with open(input_filepath, 'r') as infile:
        chunk_number = 1
        lines = infile.readlines()
        
        for i in range(0, len(lines), chunk_size):
            chunk = lines[i:i + chunk_size]
            output_filepath = f"{output_fd}/{output_prefix}_{chunk_number}.txt"
            
            with open(output_filepath, 'w') as outfile:
                outfile.writelines(chunk)
            
            print(f"Chunk {chunk_number} written to {output_filepath}")
            chunk_number += 1



def csv_to_txt(csv_filepath, txt_filepath, delimiter=','):
    with open(csv_filepath, 'r', newline='') as csvfile, open(txt_filepath, 'w') as txtfile:
        csv_reader = csv.reader(csvfile, delimiter=delimiter)
        for row in csv_reader:
            txtfile.write(' '.join(row) + '\n')


def csv_to_json(csv_file_path, json_file_path):

    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        data = list(csv.DictReader(csvfile))
    
    with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=4)




def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None

def grp_ttest(varlist, df1, df2):

    res = pd.DataFrame(columns = varlist)
    
    for i in range(len(varlist)):
        
        temp = list(ttest_ind(df1[varlist[i]], df2[varlist[i]]))

        res.at['T-statistic',varlist[i]] = temp[0]

        res.at['P-value',varlist[i]] = temp[1]

    return res


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
                
                df1['{}_bin'.format(k)], cutpoints[k] = pd.qcut(df1[k], q=var_bins[k], retbins=True, precision=0, duplicates='drop')    
            else:
                df1['{}_bin'.format(k)], cutpoints[k] = pd.qcut(df1[k], q=var_bins[k], retbins=True, precision=1+d_len, duplicates='drop') 
            
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
        row.loc[0, var] = df[var].isna().sum()
        df1.loc[df1['Statistics'] == 'count', var] += row.loc[0, var]
    
    df1 = pd.concat([df1,row], ignore_index=True)

    
    row.loc[0,'Statistics'] = 'missing_pct'
    for var in myvars:
        row.loc[0,var] = df[var].isna().sum()/df1.loc[df1['Statistics']=='count', var][0]*100
    
    df1 = pd.concat([df1, row], ignore_index=True)
    
    df1['Statistics'] = df1['Statistics'].astype('str')
    return df1


def cross_tab(df, var1, var2):
    
    res_freq = pd.crosstab(df[var1],df[var2], dropna=False)

    res_col_pct = res_freq.astype('float').copy()
    res_row_pct = res_freq.astype('float').copy()

    for i in range(res_freq.shape[0]):
        res_col_pct.iloc[i,:] /= res_freq.sum(axis=1).values[i]
        

    for i in range(res_freq.shape[1]):
        res_row_pct.iloc[:,i] /= res_freq.sum(axis=0).values[i]

    return res_freq, res_row_pct, res_col_pct


def proc_freq(df, vars):
    df_list = list()
    df0 = df.copy(deep=True)
    df0[vars] = df0[vars].astype('category')
    for var in vars:
        df1 = df0.groupby(var, observed=False)[var].count()
        df1 = pd.DataFrame(df1).rename(columns={var:'Frequency'}).reset_index()

        r_data = [['missing', df[var].isna().sum()]]
        row = pd.DataFrame(r_data, columns=[var,'Frequency'])
        # df1 = df1.append(row, ignore_index=True)
        df1 = pd.concat([df1, row], ignore_index=True)

        total = df.shape[0]
        df1['Percent'] = df1['Frequency']/total*100
        df1['Cum Frequency'] = df1['Frequency'].cumsum()
        df1['Cum Percent'] = df1['Percent'].cumsum()
        df_list.append(df1)
    return df_list



def catsum_val(df, vars, val, desc_val):
    df_list = list()
    for var in vars:
        df1 = df.loc[df[var].notna()==True, :].groupby(var, observed=False)[val].sum()
        df1 = pd.DataFrame(df1).rename(columns={val:"{}".format(desc_val)}).reset_index()
        r_data = [['missing', df.loc[df[var].isna()==True, val].sum()]]
        row = pd.DataFrame(r_data, columns=[var,"{}".format(desc_val)])
        # df1 = df1.append(row, ignore_index=True)
        df1 = pd.concat([df1, row], ignore_index = True)
        
        total = df[val].sum()
        df1["{} Pct".format(desc_val)] = df1['{}'.format(desc_val)]/total*100
        df1["Cum {}".format(desc_val)] = df1["{}".format(desc_val)].cumsum()
        df1["Cum {} Pct".format(desc_val)] = df1["{} Pct".format(desc_val)].cumsum()
        df_list.append(df1)
    return df_list
    


def cat_wt_avg(df, byvars, wt, val, desc_val):
    df_list = pd.DataFrame()
    df['wt_sum'] = df[wt]*df[val]
    for var in byvars:
        df1 = df.loc[df[var].notna()==True, :].groupby(var, observed=False)[wt].sum()
        df2 = df.loc[df[var].notna()==True, :].groupby(var, observed=False)['wt_sum'].sum()
        df1 = pd.DataFrame(df1).reset_index()
        df2 = pd.DataFrame(df2).rename(columns={'wt_sum':'Wted sum of {}'.format(desc_val)}).reset_index()
        
        r_data1 = [['missing', df.loc[df[var].isna() == True, wt].sum()]]
        r_data2 = [['missing', df.loc[df[var].isna() == True, 'wt_sum'].sum()]]
        row1 = pd.DataFrame(r_data1, columns=[var, wt])
        # df1 = df1.append(row1, ignore_index=True)
        df1 = pd.concat([df1, row1], ignore_index=True)
        
        row2 = pd.DataFrame(r_data2, columns=[var, 'Wted sum of {}'.format(desc_val)])
                
        # df2 = df2.append(row2, ignore_index=True)
        df2 = pd.concat([df2, row2], ignore_index=True)
        
        df1 = df1.merge(df2, on=var)
        total = df[wt].sum()
        df1["{} Pct".format(wt)] = df1["{}".format(wt)]/total*100
        
        df1["Weighted {}".format(val)] = df1.apply(lambda x: 0 if x[wt]==0 else x["Wted sum of {}".format(desc_val)]/x[wt], axis=1)
        
        df1 = df1[[var, wt, "{} Pct".format(wt), "Weighted {}".format(val), "Wted sum of {}".format(desc_val)]]
        df1 = df1.sort_values(by=["Weighted {}".format(val)], ascending=False)
        
        df_list = pd.concat([df_list, df1])

        
    df = df.drop('wt_sum', axis=1)
    return df_list



def write_tbles(dflist, writer=None, sheetname='sheet1', start_row=2, start_col=2):
    if writer == None:
        writer = pd.ExcelWriter("output.xlsx") 

    end_start_col = start_col

    for df in dflist:
        df.to_excel(writer, index=False, sheet_name = sheetname,
            startrow = start_row, startcol = start_col)
        start_row += (2 + df.shape[0])
        end_start_col = max(end_start_col, start_col + 1 + df.shape[1])

    return start_row, end_start_col



def pivot(df, varlist, by_vars):
    
    df_grp = df.groupby(by_vars)[df.columns[0]].count().reset_index()
    df_grp = df_grp.rename(columns={df.columns[0]:'count'})

    
    for k in varlist.keys():
        
        pair = varlist[k]  
        
        if len(pair) == 2:
            agg = pair[0]
            weight = pair[1]
            assert weight in df.columns, f'{weight} column does not exist' 
            temp = by_vars + [k, weight]
            df1 = df[temp].copy(deep=True)
            df1[f'wt_{k}'] = df1[weight]*df1[k]
            grp1 = df1.groupby(by_vars)[weight].sum().reset_index()
            grp1 = grp1.rename(columns={weight:'sum_of_weights'})
  
        else:
            agg = pair
            temp = by_vars + [k]
            df1 = df[temp].copy(deep=True)
    
        if agg == 'weighted_avg':
            
            grp2 = df1.groupby(by_vars)[f'wt_{k}'].sum().reset_index()
            grp2 = grp2.rename(columns={f'wt_{k}':f'sum_of_weighted_{k}'})
            
            grp = grp1.merge(grp2, on=by_vars)
            
            grp['WA_{}'.format(k)] = grp[f'sum_of_weighted_{k}']/grp['sum_of_weights']
            temp = by_vars + ['WA_{}'.format(k)]
            grp = grp[temp]
            
        elif agg == 'sum':
            grp = df1.groupby(by_vars)[k].agg(['sum']).reset_index()
            grp = grp.rename(columns={'sum':'sum_{}'.format(k)})
        elif agg == 'mean':
            grp = df1.groupby(by_vars)[k].agg(['mean']).reset_index()
            grp = grp.rename(columns={'mean':'mean_{}'.format(k)})
        elif agg == 'min':
            grp = df1.groupby(by_vars)[k].agg(['min']).reset_index()
            grp = grp.rename(columns={'min':'min_{}'.format(k)})
        elif agg == 'max':
            grp = df1.groupby(by_vars)[k].agg(['max']).reset_index()
            grp = grp.rename(columns={'max':'max_{}'.format(k)})
        elif agg == 'median':
            grp = df1.groupby(by_vars)[k].agg(['median']).reset_index()
            grp = grp.rename(columns={'median':'med_{}'.format(k)})
        elif agg == 'std':
            grp = df1.groupby(by_vars)[k].agg(['std']).reset_index()
            grp = grp.rename(columns={'std':'std_{}'.format(k)})
            
        elif agg == 'logodds':

            if len(pair) == 2:
                grp2 = df1.groupby(by_vars)[f'wt_{k}'].sum().reset_index()
                grp2 = grp2.rename(columns={f'wt_{k}':f'wt_sum_{k}'})
                grp  = df_grp.merge(grp1, on=by_vars).merge(grp2, on=by_vars)
                grp['logodds_{}'.format(k)]=np.log(grp[f'wt_sum_{k}']/(grp['sum_of_weights']-grp[f'wt_sum_{k}']))
                temp = by_vars + ['logodds_{}'.format(k)]
                grp = grp[temp]

            else:
                grp2 = df1.groupby(by_vars)[k].agg(['sum']).reset_index()
                temp = by_vars + ['count']
                grp3 = df_grp[temp]
                grp  = grp2.merge(grp3, on=by_vars)
                grp['logodds_{}'.format(k)]=np.log(grp['sum']/(grp['count'] - grp['sum']))
                temp = by_vars + ['logodds_{}'.format(k)]
                grp = grp[temp]

            
        else:
            warnings.warn("Skipping variable {}, invalid aggregation...".format(k))
            continue

        int_l, dec_l = precision_and_scale(min(df[k]))
        if agg in ['sum','mean','min','max','median','std']:
            grp = grp.round({grp.columns[1]:dec_l})
        df_grp = df_grp.merge(grp, on=by_vars)
    
    return df_grp
            
            




