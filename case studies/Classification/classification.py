# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 13:46:57 2020

@author: Amin
"""
# import numpy as np

# p1 = np.array([1, 1])
# p2 = np.array([4, 4])

# def distance(p1, p2):
#     """Finds the distance between points p1 and p2"""
#     return np.sqrt(np.sum(np.power(p2 - p1, 2)))

# dist = distance(p1, p2)
# # =============================================================================
# # 
# # =============================================================================
# import random
# def majority_vote(votes):
    
#     vote_counts = {}
#     for vote in votes:
#         if vote in vote_counts:
#             vote_counts[vote] += 1
#         else:
#             vote_counts[vote] = 1
#     winners = []
#     max_count = max(vote_counts.values())
#     for vote, count in vote_counts.items():
#         if count == max_count:
#             winners.append(vote)    
#     return random.choice(winners)

# # import scipy.stats as ss
# # def majority_vote_short(votes):
# #     mode, count = ss.mstats.mode(votes)
# #     return mode
    
    
    
    

    
    
# votes = [1,2,3,1,2,3,1,2,3,3,3,3,2,2,2]
# winner = majority_vote(votes)
# # =============================================================================
# # 
# # =============================================================================

# ##Finding nearest neighbors
# points = np.array([[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]])
# p = np.array([2.5, 2])

# import matplotlib.pyplot as plt
# plt.plot(points[:, 0], points[:, 1], 'ro')
# plt.plot(p[0], p[1], 'bo')
# plt.axis([0.5, 3.5, 0.5, 3.5])

# def find_nearest_neighbors(p, points, k):
#     """Find the k nearest neighbors of point p and return their indices """
#     distances = np.zeros(points.shape[0])
#     for i in range(len(distances)):
#         distances[i] = distance(p, points[i])
#     ind = np.argsort(distances)
#     return ind[:k]

# outcomes = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
# def knn_predict(p, points, outcomes, k=5):
#     ind = find_nearest_neighbors(p, points, k)
#     return majority_vote(outcomes[ind])

# knn_predict(np.array([2.5, 2.7]), points, outcomes, k=2)
# # =============================================================================
# # 
# # =============================================================================
# ##Generating synthetic data
# import scipy.stats as ss
# import matplotlib.pyplot as plt


# def generate_synth_data(n):
#     points = np.concatenate((ss.norm(0, 1).rvs((n, 2)), ss.norm(1, 1).rvs((n, 2))), axis = 0)
#     outcomes = np.concatenate((np.repeat(0, n), np.repeat(1, n)))
#     return (points, outcomes)

# n = 20
# (points, outcomes) = generate_synth_data(n)

# plt.figure()
# plt.plot(points[:n, 0], points[:n, 1], 'ro')
# plt.plot(points[n:, 0], points[n:, 1], 'bo')
# # plt.savefig('bivariate.pdf')

# # =============================================================================
# # 
# # =============================================================================
# ## making a prediction grid

# def make_prediction_grid(predictors, outcomes, limits, h, k):
#     """Classify reach point on the predictoin grid."""
#     (x_min, x_max, y_min, y_max) = limits
#     xs = np.arange(x_min, x_max, h)
#     ys = np.arange(y_min, y_max, h)
#     xx, yy = np.meshgrid(xs, ys)
    
#     prediction_grid = np.zeros(xx.shape, dtype=int)
#     for i,x in enumerate(xs):
#         for j,y in enumerate(ys):
#             p = np.array([x, y])
#             prediction_grid[j, i] = knn_predict(p, predictors, outcomes, k)
#     return (xx, yy, prediction_grid)
# ### beforehand prepared file on edx 
# def plot_prediction_grid (xx, yy, prediction_grid, filename):
#     """ Plot KNN predictions for every point on the grid."""
#     from matplotlib.colors import ListedColormap
#     background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
#     observation_colormap = ListedColormap (["red","blue","green"])
#     plt.figure(figsize =(10,10))
#     plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
#     plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
#     plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
#     plt.xticks(()); plt.yticks(())
#     plt.xlim (np.min(xx), np.max(xx))
#     plt.ylim (np.min(yy), np.max(yy))
#     plt.savefig(filename)
# ###
# (predictors, outcomes) = generate_synth_data(n=50)

# k=5 ; filename = 'knn_synth_5.pdf'; limits = (-3,4,-3,4); h = 0.1
# (xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits, h, k)
# plot_prediction_grid(xx, yy, prediction_grid, filename)    

# k=50 ; filename = 'knn_synth_50.pdf'; limits = (-3,4,-3,4); h = 0.1
# (xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits, h, k)    
# plot_prediction_grid(xx, yy, prediction_grid, filename) 

# # =============================================================================
# # 
# # =============================================================================
# ##Applying the knn method
# from sklearn import datasets
# iris = datasets.load_iris()

# predictors = iris.data[:, 0:2]
# outcomes = iris.target

# plt.plot(predictors[outcomes == 0][:, 0], predictors[outcomes == 0][:, 1], 'ro')
# plt.plot(predictors[outcomes == 1][:, 0], predictors[outcomes == 1][:, 1], 'go')
# plt.plot(predictors[outcomes == 2][:, 0], predictors[outcomes == 2][:, 1], 'bo')
# plt.savefig('iris.pdf')

# k=50 ; filename = 'iris_gird.pdf'; limits = (4,8,1.5,4.5); h = 0.1
# (xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits, h, k)    
# plot_prediction_grid(xx, yy, prediction_grid, filename) 
# ##
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=5)

# =============================================================================
# 
# =============================================================================
###HomeWork
# import numpy as np, random, scipy.stats as ss
# import pandas as pd
# def majority_vote_fast(votes):
#     mode, count = ss.mstats.mode(votes)
#     return mode

# def distance(p1, p2):
#     return np.sqrt(np.sum(np.power(p2 - p1, 2)))

# def find_nearest_neighbors(p, points, k=5):
#     distances = np.zeros(points.shape[0])
#     for i in range(len(distances)):
#         distances[i] = distance(p, points[i])
#     ind = np.argsort(distances)
#     return ind[:k]
# ##1
# def knn_predict(p, points, outcomes, k=5):
#     ind = find_nearest_neighbors(p, points, k)
#     return majority_vote_fast(outcomes[ind])[0]
# wines = pd.read_csv('wine.csv')
# wines.head()
# ##2
# wines = wines.rename(columns={"color": "is_red"})
# wines['is_red'] = wines['is_red'].replace('red', 1)
# wines['is_red'] = wines['is_red'].replace('white', 0)
# numeric_data = wines.copy()
# numeric_data
# sum(numeric_data['is_red'])
# ##3
# import sklearn.preprocessing

# scaled_data = sklearn.preprocessing.scale(numeric_data)
# numeric_data = pd.DataFrame(data=scaled_data, columns=numeric_data.columns)
# import sklearn.decomposition

# pca = sklearn.decomposition.PCA(n_components=2)
# principal_components =  pca.fit_transform(numeric_data)
# principal_components.shape
# ##4
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# from matplotlib.backends.backend_pdf import PdfPages
# observation_colormap = ListedColormap(['red', 'blue'])
# x = principal_components[:,0]
# y = principal_components[:,1]

# plt.title("Principal Components of Wine")
# plt.scatter(x, y, alpha = 0.2,c = numeric_data['high_quality'], cmap = observation_colormap, edgecolors = 'none')
# plt.xlim(-8, 8); plt.ylim(-8, 8)
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.show()
# ##5
# import numpy as np 
# np.random.seed(1) # do not change

# x = np.random.randint(0, 2, 1000)
# y = np.random.randint(0 ,2, 1000)

# def accuracy(predictions, outcomes):
#     same_values = np.sum(predictions == outcomes);
#     return 100 * same_values / len(outcomes)

# print(accuracy(x, y))
# ##6
# print(accuracy(0, wines["high_quality"]))
# ##7
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors = 5)
# knn.fit(numeric_data, wines['high_quality'])
# # Enter your code here!
# library_predictions = knn.predict(numeric_data)
# accuracy = accuracy(library_predictions, wines['high_quality'])
# print(accuracy)
# ##8
# n_rows = wines.shape[0]

# import random
# random.seed(123)
# selection = random.sample(range(n_rows), 10)
# print(selection)
