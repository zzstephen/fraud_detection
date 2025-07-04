#authored by Stephen Zhou

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib.pyplot import figure
import plotly.express as px
import pdb
from sklearn.feature_selection import *
import networkx as nx
import seaborn as sns
#imports for plotly interactive visualisation library
import plotly.graph_objs as go
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn import *
from sklearn.ensemble import *
from sklearn import svm
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.stattools import adfuller
from sklearn.svm import LinearSVC
from sklearn.metrics import normalized_mutual_info_score
from xgbfir import *
from xgboost import *



def getWeights(d, thres):
    w,k = [1.0], 1
    while True:
        w_ = -w[-1]/k*(d-k+1)
        if abs(w_)<thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1,1)


def f_fracDiff(series, d, thres=1e-5):
    w = getWeights(d, thres)
    width, df = len(w)-1, {}
    for name in series.columns:
        seriesF = series[[name]].fillna(method='ffill').dropna()
        df_ = pd.Series(np.zeros(seriesF.shape[0]))
        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1-width], seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1,name]):
                continue
            df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0,0]
        df[name] = df_.copy(deep=True)
        
    df = pd.concat(df, axis=1)
    return df


def f_plotMinFFD(df, var):
    out = pd.DataFrame(columns=['adfStat','p-Value','lags','nObs','95% conf','corr'])
    df1 = df[[var]].copy(deep=True)
    for dd in np.linspace(0,1,21):
        df2 = f_fracDiff(df1, dd, thres=0.01) if dd>0.01 else df1
        df2 = df2.loc[df2[var]>0]
        corr = np.corrcoef(df1.loc[df2.index,var], df2[var])[0,1] if dd>0.01 else 1
        df2 = adfuller(df2, maxlag=1, regression='c', autolag=None)
        out.loc[dd]=list(df2[:4])+[df2[4]['5%']]+[corr]
    # out.to_csv('test_MinFFD.csv')
    out[['adfStat','corr']].plot(secondary_y='adfStat')
    plt.axhline(out['95% conf'].mean(), linewidth = 2, color='r', linestyle = 'dotted')
    plt.xlabel('d')
    # plt.savefig('test_MinFFD.png')
    return out
    
    
def f_importances(coef, names, title):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.title(title)
    plt.show()
    

def get_coordinates(mst):
    """Returns the positions of nodes and edges in a format for Plotly to draw the network"""
    # get list of node positions
    
    pos = nx.fruchterman_reingold_layout(mst)
    
    Xnodes = [pos[n][0] for n in mst.nodes()]
    Ynodes = [pos[n][1] for n in mst.nodes()]

    Xedges = []
    Yedges = []
    for e in mst.edges():
        # x coordinates of the nodes defining the edge e
        Xedges.extend([pos[e[0]][0], pos[e[1]][0], None])
        Yedges.extend([pos[e[0]][1], pos[e[1]][1], None])

    return Xnodes, Ynodes, Xedges, Yedges



def assign_colour(correlation):
    if correlation > 0:
        return "#ffa09b"  # red
    else:
        return "#9eccb7"  # green
    
    
    

class sk_feature_selection:
    
    
#     def __init__(self):





    @staticmethod
    def f_normalized_mi_matrix(df:pd.DataFrame, varlist:list):
        
        n_cols = len(varlist)
        nmi_matrix = np.zeros((n_cols, n_cols))

        for i in range(n_cols):
            for j in range(n_cols):
                nmi_matrix[i, j] = normalized_mutual_info_score(df[varlist[i]], df[varlist[j]])
        

        matrix = pd.DataFrame(columns = ['feature'] + varlist)
        
        matrix['feature'] = varlist
        
        for i in range(n_cols):
            matrix.iloc[i,1:] = nmi_matrix[i]

        return matrix


        
    @staticmethod    
    def f_low_variation(df:pd.DataFrame, varlist:list, thres:float):
        
        var_std = pd.DataFrame(columns = ['feature','rescaled_std','recommendation','outliers'])

        for v in varlist:

            v_o_range = df[v].max() - df[v].min()
            
            v_min = df[v].quantile(0.01)
            v_max = df[v].quantile(0.99)
            v_range = v_max - v_min
    
            var = df[[v]].loc[(df[v]>=v_min) & (df[v]<=v_max)]
            var = var/(v_max-v_min)
            re_std = var.std().values[0]
            
            if v_o_range > 10*v_range:
                outlier = True
            else:
                outlier = False

            row_df = pd.DataFrame([{'feature':v, 'rescaled_std':re_std, 'recommendation':('keep' if re_std>thres else 'drop'), 'outliers':outlier}])

            if var_std.empty:
                var_std = row_df
            else:
                var_std = pd.concat([var_std, row_df], ignore_index=True)
            
            var_std = var_std.sort_values(by=['rescaled_std'], ascending=False).reset_index(drop=True)
        
        return var_std
    
    
    
    @staticmethod
    def f_mutual_info(df:pd.DataFrame, cont_vars:list, disc_vars:list, y_var:str, ytype:str = 'continuous', n:int = 3, random_state:int=123, corr_network:bool=True):
        
        # cont_vars = []
        # disc_vars = []
        
        # for v in x_vars:
        #     if df[v].dtype in (int, bool):
        #         disc_vars.append(v)
        #     elif df[v].dtype == float:
        #         cont_vars.append(v)

        cont_mi = pd.DataFrame(columns=['feature','mutual_info'])
        disc_mi = pd.DataFrame(columns=['feature','mutual_info'])
        
        if ytype == 'continuous':
            if len(cont_vars)>0:
                cont_mi['mutual_info'] = mutual_info_regression(df[cont_vars], df[y_var].values.ravel(), discrete_features=False, n_neighbors=n, random_state=random_state)
                cont_mi['feature'] = cont_vars
            if len(disc_vars)>0:
                disc_mi['mutual_info'] = mutual_info_regression(df[disc_vars], df[y_var].values.ravel(), discrete_features=True, n_neighbors=n, random_state=random_state)
                disc_mi['feature'] = disc_vars
    
        elif ytype == 'categorical':
            if len(cont_vars)>0:
                cont_mi['mutual_info'] = mutual_info_classif(df[cont_vars], df[y_var].values.ravel(), discrete_features=False, n_neighbors=n, random_state=random_state)
                cont_mi['feature'] = cont_vars
            if len(disc_vars)>0:
                disc_mi['mutual_info'] = mutual_info_classif(df[disc_vars], df[y_var].values.ravel(), discrete_features=True, n_neighbors=n, random_state=random_state)
                disc_mi['feature'] = disc_vars
        
        mi = pd.concat([cont_mi, disc_mi], ignore_index=True)
        mi = mi.sort_values(by=['mutual_info'], ascending=False).reset_index(drop=True)
        
        if corr_network == True:

            xy_vars = [y_var] + cont_vars + disc_vars
            
            
            correlation_matrix = df[xy_vars].corr()
            edges = correlation_matrix.stack().reset_index()
            
            edges.columns = ['node_1','node_2','correlation']
            edges = edges.loc[edges['node_1'] != edges['node_2']].copy()
            
#             threshold = 0.5

            Gx = nx.from_pandas_edgelist(edges, 'node_1', 'node_2', edge_attr=['correlation'])
    
            mst = nx.minimum_spanning_tree(Gx)

#             remove = []

#             for node_1, node_2 in Gx.edges():
#                 corr = Gx[node_1][node_2]['correlation']
#                 if abs(corr) < threshold:
#                     remove.append((node_1, node_2))
                    
#             Gx.remove_edges_from(remove)



            node_label = list(mst.nodes())
        
            # create tooltip string by concatenating statistics
            description = [f"<b>{node}</b>" +    
                           "<br>mutual info ranking # " + "{}".format(1 if node==y_var else (mi.loc[mi['feature']==node].index.values[0]+1)) +
                           "<br>correlation with " + "{}: {}".format(y_var, np.round(correlation_matrix[y_var].iloc[index],3))
                           for index, node in enumerate(node_label)]
            
            
            Xnodes, Ynodes, Xedges, Yedges = get_coordinates(mst)
            node_colour = [assign_colour(i) for i in correlation_matrix[y_var]]
            node_size = [abs(i)*100 for i in correlation_matrix[y_var]]
            
            
            tracer = go.Scatter(x=Xedges, y=Yedges,
                    mode='lines',
                    line= dict(color='#DCDCDC', width=1),
                    hoverinfo='none',
                    showlegend=False)

            tracer_marker = go.Scatter(x=Xnodes, y=Ynodes,
                                       mode='markers+text',
                                       textposition='top center',
                                       marker=dict(size=node_size,
                                                        line=dict(width=1),
                                                        color=node_colour),
                                       hoverinfo='text',
                                       hovertext=description,
                                       text=node_label,
                                       textfont=dict(size=7),
                                       showlegend=False)


            axis_style = dict(title='',
                              titlefont=dict(size=20),
                              showgrid=False,
                              zeroline=False,
                              showline=False,
                              ticks='',
                              showticklabels=False)


            layout = dict(title='Plotly - Correlation Network for target {}'.format(y_var),
                          width=800,
                          height=800,
                          autosize=False,
                          showlegend=False,
                          xaxis=axis_style,
                          yaxis=axis_style,
                          hovermode='closest',
                         plot_bgcolor = '#fff')


            fig = dict(data=[tracer, tracer_marker], layout=layout)

            iplot(fig)
            
        return mi
        
    


    @staticmethod
    def f_get_interaction_select(df:pd.DataFrame, x_vars:list, y_var:str, mtype:str='regression', order:int=2, weight:str=None, out_file:str = '', base_margin=None):

        assert mtype in ['regression','classification'], 'mtype must be classification or regression'

        df1 = df.copy()

        if weight == None:
            df1['sample_weight'] = 1
            weight = 'sample_weight'
        else:
            assert weight in df.columns, f'{weight} does not exist in the columns'

        if out_file == '':
            out_file = f'{mtype}_feature_interactions_order{order}.xlsx'

        if mtype == 'regression':

            clf = XGBRegressor(objective ='reg:linear',
                               colsample_by = 0.7, subsample= 0.7,
                               num_parallel_tree=100, num_boost_round=1, seed = 123, n_jobs=-1)

        else:

            clf = XGBClassifier(objective ='binary:logistic',
                               colsample_by = 0.7, subsample= 0.7,
                               num_parallel_tree=100, num_boost_round=1, seed = 123, n_jobs=-1)

        if base_margin != None:
            clf.fit(df1[x_vars], df1[y_var], base_margin = base_margin, sample_weight=df1[weight])
        else:
            clf.fit(df1[x_vars], df1[y_var], sample_weight=df1[weight])

        saveXgbFI(clf, feature_names=x_vars, OutputXlsxFile=out_file, MaxTrees=100, MaxInteractionDepth=order-1)


    @staticmethod
    def f_feature_select(df:pd.DataFrame, x_vars:list, y_var:str, mtype:str='regression', chart:str='off'):
        
        df1 = df.copy()
        
        for v in x_vars:
            if df1[v].dtype in (int, float):
                v_mean = df1[v].mean()
                v_std = df1[v].std()
                df1[v] = df1[v].apply(lambda x: (x - v_mean)/v_std)
            
        
        if mtype == 'regression':
            coeffs = pd.DataFrame(columns = ['feature','Ridge','random_forest'])
            coeffs.feature = x_vars
            
            clf1 = linear_model.Ridge(alpha=0.05)
            clf1.fit(df1[x_vars], df1[y_var])
            
            coeffs.Ridge = clf1.coef_
            
            clf2 = RandomForestRegressor(n_estimators=50, random_state=0)
            clf2.fit(df1[x_vars], df1[y_var])
            coeffs.random_forest = clf2.feature_importances_
            
            if chart == 'on':
                f_importances(coeffs.Ridge, x_vars, 'Ridge Regression Coefficients')
                f_importances(coeffs.random_forest, x_vars, 'Random Forest Gini importance')
            
            
        elif mtype == 'classification':
            coeffs = pd.DataFrame(columns=['feature','linear_SVC','random_forest'])
            clf1 = LinearSVC(C=0.05, penalty='l2', dual=False)
            clf1.fit(df1[x_vars], df1[y_var])

            coeffs.feature = x_vars
            
            coeffs.linear_SVC=clf1.coef_[0].tolist()
            
            
            clf2 = RandomForestClassifier(n_estimators=50, random_state=0)
            clf2.fit(df1[x_vars], df1[y_var])
            coeffs.random_forest = clf2.feature_importances_
            
            if chart == 'on':
                f_importances(coeffs.linear_SVC, x_vars, 'Linear SVC Coefficients')
                f_importances(coeffs.random_forest, x_vars, 'Random Forest Gini importance')
            
        return coeffs
    
    
 
    
    
    


