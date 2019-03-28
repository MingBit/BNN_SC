# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Functions import SBR_Func

path = '/home/angela/PhD_Pro/TODOLIST/BN_Test/SC_Published_Data/'
Expr_Data = pd.read_excel(path + 'Moignard_HSC_Data.xlsx', index_col = 0)

#create series from numpy
coutries = ['USA','Nigeria','France', 'Ghana']
my_data = [100,200, 30, 40]

tmp = pd.Series(my_data, coutries)
tmp['USA']

np_arr = np.array(my_data)
pd.Series(np_arr)

my_dict = {'a':10, 'b':2, 'c':90, 'd': 80}
pd.Series(my_dict)
		
#pandas with dataframe
tmp.head(5)
(tmp[['Cbfa2t3h', 'Cdh1']])

