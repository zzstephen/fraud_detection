import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib.pyplot import figure
import plotly.express as px
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
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.cluster import AgglomerativeClustering
import heapq


def find_tree_leaves(tree, root, leaves):
    if tree.children_left[root] == -1 and tree.children_right[root] == -1:
        leaves.append(root)
    else:
        if tree.children_left[root] != -1:
            find_tree_leaves(tree, tree.children_left[root], leaves)
        if tree.children_right[root] != -1:
            find_tree_leaves(tree, tree.children_right[root], leaves)


class feature_engineer:

    @staticmethod
    def auto_binning(df:pd.DataFrame, x_var:str, y_var:str, n_bins:int, mtype:str='regression', class_weight:dict=None):

        assert mtype in ['regression','classification'], 'mtype must be regression or classification'

        bins = [min(df[x_var])]

        if mtype == 'regression':
            #class_weight={0: 1, 1: 5}
            clf = DecisionTreeRegressor(max_leaf_nodes=n_bins, random_state=123, class_weight=class_weight)
        else:
            clf = DecisionTreeClassifier(max_leaf_nodes=n_bins, random_state=123, class_weight=class_weight)

        clf.fit(df[[x_var]], df[y_var])
        leaves = []
        find_tree_leaves(clf.tree_, 0, leaves)
        splits = sorted(clf.tree_.threshold[list(set(range(len(clf.tree_.threshold))) - set(leaves))])
        bins += splits

        bins.append(max(df[x_var]))

        return bins