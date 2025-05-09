#authored by stephen Zhou

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib.pyplot import figure
import plotly.express as px
import pdb
import plotly.io as pio
pio.renderers.default = "iframe"
from scipy.interpolate import *
from plotly.subplots import make_subplots
from patsy import dmatrix, demo_data, ContrastMatrix, Poly
import plotly.graph_objects as go
from plotly.offline import iplot
from scipy.stats import ttest_ind_from_stats




def px_scatter_plot(df, x_var, y_var, by_var1 = None, by_var2 = None, color_var=None, x_trans = None, y_trans = None, width=400, height=300, show=True, title=None):

    #x_trans, y_trans are functions

    
    
    keep_var0 = [x_var, y_var, by_var1, by_var2, color_var]
    keep_var = [i for i in keep_var0 if i]
    df1 = df[keep_var].copy(deep=True)
    
    df1['x_var'] = df1.apply(lambda x: x_trans(x[x_var]) if (x_trans != None) else x[x_var], axis=1)
    df1['y_var'] = df1.apply(lambda x: y_trans(x[y_var]) if (y_trans != None) else x[y_var], axis=1)
    
    row_level = len(df1[by_var1].unique()) if by_var1 != None else 1
    col_level = len(df1[by_var2].unique()) if by_var2 != None else 1

    if title == None:
        title = f"{x_var} vs {y_var}"
    
    fig = px.scatter(df1, x = 'x_var', y = 'y_var', color = color_var, facet_row=by_var1, facet_col=by_var2, trendline='ols', 
                     trendline_color_override="red", title=title)

    fig.update_layout(
    xaxis_title=x_var,
    yaxis_title=y_var)

    if show:
        fig.show()
        
    return fig
    
def px_ecdf_plot(df, x_var, cat_var = None, by_var = None, width=600, height=450):

    row_level = len(df[by_var].unique()) if by_var != None else 1
    fig = px.ecdf(df, x=x_var, color=cat_var, markers=True, lines=False, facet_row= by_var, marginal="histogram", width=width, height=row_level*height)
    fig.show()

    
def px_bin_plot(df, x_bin_var, y_var, size_var, by_var = None, width=600, height=450):
    subfig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # create two independent figures with px.line each containing data from multiple columns
    fig = px.line(df, x=x_bin_var, y=y_var, color=by_var, markers=True)
    fig2 = px.line(df, x=x_bin_var, y=size_var, color=by_var, markers=True)

    fig2.update_traces(yaxis="y2")

    subfig.add_traces(fig.data + fig2.data)
    subfig.layout.xaxis.title=x_bin_var
    subfig.layout.yaxis.title=y_var
    subfig.layout.yaxis2.title=size_var

    subfig.for_each_trace(lambda t: t.update(line=dict(color=t.marker.color)))
    subfig.show()

    
def f_get_dummies(df, varlist, drop_first=True):

    return pd.get_dummies(df,prefix=varlist, columns=varlist, drop_first=drop_first, prefix_sep=':',dummy_na=True)


def f_get_1d_knots(df, varlist, knots_list, overlap=False):
    
    df1 = df.copy()
    
    if overlap == False:
        for v in varlist:
            knots = knots_list[v]
            df1["{}_{}".format(v,0)] = df1.apply(lambda x: min(knots[0],x[v]), axis=1)
            for k in range(1, len(knots), 1):
                df1["{}_{}".format(v,k)] = df1.apply(lambda x: min(max(0,x[v]-knots[k-1]),knots[k]-knots[k-1]), axis=1)
            df1["{}_{}".format(v,len(knots))] = df1.apply(lambda x: max(x[v]-knots[-1],0), axis=1)
    else:
        for v in varlist:
            knots = knots_list[v]
            df1["{}_0".format(v)] = df1[v]
            for k in range(len(knots)):
                df1["{}_{}".format(v,k+1)] = df1.apply(lambda x: max(x[v] - knots[k],0),axis=1)
    return df1
    
    

def px_scatter_3d_plot(df, x1_var, x2_var, y_var, width=800, height=600):

    x = df[x1_var].values
    y = df[x2_var].values
    z = df[y_var].values
    
    
    trace = go.Scatter3d(
       x = x, y = y, z = z, mode = 'markers', marker = dict(
          size = 10,
          color = z, # set color to an array/list of desired values
          colorscale = 'Viridis'
          )
       )
    layout = go.Layout(title = '3D scatter plot of {} by {}, {}'.format(y_var, x1_var, x2_var), width=width, height=height)
    fig = go.Figure(data = [trace], layout = layout)
    iplot(fig)
    return fig


