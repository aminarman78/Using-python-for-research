# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 19:09:29 2020

@author: Amin
"""

# import pandas as pd

# x = pd.Series([6,3,8,6], index = ['z','d','q','a'])
# age = {"Tim":29 , 'Jim':28, 'Pam':18, 'Sam':25}
# y = pd.Series(age)
# data = {'name' : ['Tim', 'Jim', 'Pam', 'Sam'], 
#         'age': [29, 31, 27, 35],
#         'ZIP':['02115', '02130', '67700', '00100']}
# z = pd.DataFrame(data, columns = ['name', 'age', 'ZIP'])
# print(z['name'])
# print(z.name)
# print(x.index)
# print(sorted(x.index))
# x.reindex(sorted(x.index))
# x = pd.Series([6,3,8,6], index = ['z','d','q','a'])
# g = pd.Series([7,5,2,1], index = ['a','b','c','d'])
# =============================================================================
# 
# =============================================================================
### Loading and inspecting data
# import numpy as np
# import pandas as pd

# whisky = pd.read_csv('whiskies.txt')
# whisky['Regions'] = pd.read_csv('regions.txt')
# whisky.head()
# whisky.tail()

# whisky.iloc[5:10]
# whisky.iloc[5:10, 0:6]
# whisky.columns
# flavors = whisky.iloc[:, 2:14]
# ### Exploring correlations
# corr_flavors = pd.DataFrame.corr(flavors)
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10,10))
# plt.pcolor(corr_flavors)
# plt.colorbar()
# plt.savefig('corr_flavors.pdf')

# corr_whisky = pd.DataFrame.corr(flavors.transpose())
# plt.figure(figsize=(10,10))
# plt.pcolor(corr_whisky)
# plt.axis('tight')
# plt.colorbar()
# plt.savefig('corr_whisky.pdf')

# ### Clustering whiskies by flavor profiles
# from sklearn.cluster.bicluster import SpectralCoclustering

# model = SpectralCoclustering(n_clusters=6, random_state=0)
# model.fit(corr_whisky)
# np.sum(model.rows_, axis=1)
# np.sum(model.rows_, axis=0)
# model.row_labels_
# ### Comparing correlation matrices

# whisky['Group'] = pd.Series(model.row_labels_, index = whisky.index)
# whisky = whisky.iloc[np.argsort(model.row_labels_)]
# whiskiy = whisky.reset_index(drop=True)
# correlations = pd.DataFrame.corr(whisky.iloc[:, 2:14].transpose())

# plt.figure(figsize = (14,7))
# plt.subplot(121)
# plt.pcolor(corr_whisky)
# plt.title('Original')
# plt.axis('tight')
# plt.subplot(122)
# plt.pcolor(correlations)
# plt.title('Rearranged')
# plt.axis('tight')
# plt.savefig('Correlations.pdf')
# =============================================================================
# 
# =============================================================================

### Homework
##### Homework is written in Jupyter Notebook and the file named as HW4-1, refer to the directory
