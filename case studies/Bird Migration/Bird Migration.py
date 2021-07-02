# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 03:20:03 2020

@author: Amin
"""
# import pandas as pd

# birddata = pd.read_csv('bird_tracking.csv')
# birddata.info()
# birddata.head()
### simple data visuallization

# import matplotlib.pyplot as plt
# import numpy as np

# bird_names = pd.unique(birddata.bird_name) 
# plt.figure(figsize=(7,7))
# for bird_name in bird_names:
#     ix = birddata.bird_name == bird_name
#     x, y = birddata.longitude[ix], birddata.latitude[ix]
#     plt.plot(x,y,'.', label=bird_name)
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.legend(loc="lower right")
# plt.savefig('3traj.pdf')
# =============================================================================
# 
# =============================================================================
### Examining flight speed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

birddata = pd.read_csv('bird_tracking.csv')
# ix = birddata.bird_name == 'Eric'
# speed = birddata.speed_2d[ix]
# plt.hist(speed[:10])
#debugging hist function 
# np.isnan(speed)
# np.isnan(speed).any()
# print(np.sum(np.isnan(speed)))

# ind = np.isnan(speed)
# plt.hist(speed[~ind])
# plt.savefig('hist.pdf')

### modifying histogram and normalizing

# plt.figure(figsize = (8, 4))
# speed = birddata.speed_2d[birddata.bird_name == 'Eric']
# ind = np.isnan(speed)
# plt.hist(speed[~ind], bins = np.linspace(0, 30, 20))
# plt.xlabel('2D speed (m/s)')
# plt.ylabel('Frequency');

# ### Histogram with Pandas
# birddata.speed_2d.plot(kind='hist', range=[0, 30])
# plt.xlabel('2D speed (m/s)')
# plt.ylabel('Frequency');
# plt.savefig('pd_hist.pdf')

# ## Using Datetime
# birddata.columns
# birddata.date_time
# import datetime
# time1 = datetime.datetime.today()
# time2 = datetime.datetime.today()
# del_time = time2 - time1
# print(del_time)

# date_str = birddata.date_time[0]
# print(type(date_str))
# timestamps = []
# for k in range(len(birddata)):
#     timestamps.append(datetime.datetime.strptime\
#                       (birddata.date_time.iloc[k][:-3], "%Y-%m-%d %H:%M:%S"))

# birddata["timestamp"] = pd.Series(timestamps, index=birddata.index)

# times = birddata.timestamp[birddata.bird_name == 'Eric']
# elapsed_time = [time - times[0] for time in times]
# print(elapsed_time[200]/datetime.timedelta(days=1))

# # plt.plot(np.array(elapsed_time)/datetime.timedelta(days=1))
# # plt.xlabel('Observation')
# # plt.ylabel('Elapsed time (days)')
# # plt.savefig('timeplot.pdf')

# ### Calculating daily mean speed

# data = birddata[birddata.bird_name == 'Eric'] 
# times = data.timestamp
# elapsed_time = [time - times[0] for time in times]
# elapsed_days = np.array(elapsed_time)/datetime.timedelta(days=1)

# next_day = 1
# inds = []
# daily_mean_speed = []

# for (i,t) in enumerate(elapsed_days):
#     if t<next_day:
#         inds.append(i)
#     else:
#         #Compute mean speed
#         daily_mean_speed.append(np.mean(data.speed_2d[inds]))
#         next_day += 1
#         inds = []

# plt.figure(figsize=(8,6))
# plt.plot(daily_mean_speed)
# plt.xlabel('Day')
# plt.ylabel("Mean speed (m/s)")
# plt.savefig('dms.pdf')

# ### Using the cartopy library
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

# proj = ccrs.Mercator()

# plt.figure(figsize=(10,10))
# ax = plt.axes(projection=proj)
# ax.set_extent((-25,20,52,10))
# ax.add_feature(cfeature.LAND)
# ax.add_feature(cfeature.OCEAN)
# ax.add_feature(cfeature.COASTLINE)
# ax.add_feature(cfeature.BORDERS, linestyle=':')

# bird_names = pd.unique(birddata.bird_name)
# for name in bird_names:
#     ix = birddata['bird_name'] == name
#     x, y = birddata.longitude[ix], birddata.latitude[ix]
#     ax.plot(x, y, '.', transform=ccrs.Geodetic(), label=name)
    
# plt.legend(loc='upper left')
# plt.savefig('map.pdf')

# =============================================================================
# 
# =============================================================================
### HomeWork (general file is in jupyter notebook)

import pandas as pd
import numpy as np
birddata = pd.read_csv("https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@bird_tracking.csv", index_col=0)
birddata.head()