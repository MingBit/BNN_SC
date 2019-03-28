#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 10:33:38 2018

@author: angela
"""

import numpy as np
from matplotlib.pyplot import scatter
import pandas as pd
import operator
from scipy.stats import pearsonr
#from itertools import chain

path = '/home/angela/Phd_Pro/'

ex_matrix = pd.read_csv('/home/angela/MING_V9T/KK_Run36/UMI_KK_Filterby_Gene500_UMI5000.csv', sep=' ')
design = pd.read_csv('/home/angela/MING_V9T/KK_Run36/KK_Run36_Design.tsv', sep = ' ')
#print(ex_matrix.index, ex_matrix.columns)
Cont_cells = design[design['cell_type'] == 'Cont'].index
ex_Cont_matrix = ex_matrix.loc[:, Cont_cells]

Tox_expr = ex_Cont_matrix.loc['Tox']
Tox_expr = Tox_expr[Tox_expr > 0]
Tox_expr_Asc = pd.DataFrame(Tox_expr.sort_values(ascending = True))
Tox_expr_Dsc = pd.DataFrame(Tox_expr.sort_values(ascending = False))
Tox_expr_Pos = ex_matrix.loc[:, Tox_expr_Asc.index]

Tox_Pos_Corr = pd.DataFrame(np.random.normal(loc = 20, scale = 5, size = (434,50)), 
							columns = ['Posgene' + str(i) for i in range(50)],
							index = Tox_expr_Asc.index)
Tox_Neg_Corr = pd.DataFrame(np.random.normal(loc = 5, scale = 2, size = (434,50)), 
							columns = ['Neggene' + str(i) for i in range(50)],
							index = Tox_expr_Asc.index)


for ge in Tox_Pos_Corr.columns:
		Tox_Pos_Corr[ge] = np.fromiter(map(operator.add, Tox_Pos_Corr[[ge]].values, Tox_expr_Asc.values), 
			  dtype = np.int)
for ge in Tox_Neg_Corr.columns:
		Tox_Neg_Corr[ge] = np.fromiter(map(operator.add, Tox_Neg_Corr[[ge]].values, Tox_expr_Dsc.values), 
			  dtype = np.int)

scatter(Tox_Pos_Corr['Posgene2'].values, Tox_expr_Asc.values.T[0])

#np.corrcoef(Tox_expr_Asc.values.T, Tox_Pos_Corr['gene2'].values)
#np.cov(Tox_expr_Asc.values.T, Tox_Pos_Corr['gene2'].values)
r,p = pearsonr(Tox_expr_Asc.values.T[0], Tox_Pos_Corr['Posgene20'].values)
r

#merge dataframes
merge_df = pd.concat([Tox_expr_Pos, Tox_Pos_Corr.T, Tox_Neg_Corr.T])
#Filter by rowmeans
mean_df = merge_df.mean(axis = 1) 
filter_gene = mean_df[mean_df > 1].index
filter_df = merge_df.loc[filter_gene]

#KNN test
from sklearn.neighbors import kneighbors_graph

gene_knn = pd.DataFrame(kneighbors_graph(filter_df, 100, n_jobs=6).toarray(),
						index = filter_df.index,
						columns = filter_df.index)

#Score pos-like / neg-like genes
gene_knn.sum(axis = 1)

tmp = gene_knn.loc[:, Tox_Pos_Corr.columns]
tmp1 = tmp.T
tmp2 = tmp[tmp.sum(axis = 1) > 1]
writer = pd.ExcelWriter('/home/angela/MING_V9T/KK_Run36/gene_KNN_Neg.xlsx', engine = 'xlsxwriter')

tmp2.to_excel(writer, sheet_name = 'Sheet1')
writer.save()

