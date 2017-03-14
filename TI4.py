# Dota2_PP.py - Analytics of Dota 2 International Prizepool(s)
# C. Casarella Mar 12, 2017

import itertools
import numpy as np
import scipy as sc
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def base_log_func(x,a,b,c):
  """
  Base log function to fit data from day 1. 
  Implies the prizepool starts at 1.6M USD
  """
  return 1.6 + b * np.log(a * x + c)

def log_func(x,a,b,c,d):
  """
  If I need a generalized log function to fit
  """
  return d + b * np.log(a * x + c)

def exp_decay(x,a,b,c,d):
  """
  Exponential decay, intended use:
  on Daily Change
  """
  return b*np.exp(a * x + c) + d

def cache1(x,a,b,c):
  """
  Another log 'on-top-of' the base fit
  Intended use: after the first treasure is released.
  crossover_point is intended to be the new 'y-intercept'
  """
  return turning_point + b * np.log(a * x + c)

def variance(fit,data):
  """
  Calculates the deviation of the model versus the data.
  """
  var=[]
  for ind in list(range(len(data))):
    var.append(fit[ind][1]-data[ind][1])
  return var
    
#TI4_input='TI4_data'

#csv_TI4 = np.genfromtxt(TI4_input,delimiter=",")

#print(csv_TI4)

#TI4_list = []
#with open(TI4_input,'r') as file:
    #for line in file:
        #line = line.strip('') #or someother preprocessing
        #TI4_list.append(line)

#x_list= [list(t) for t in zip(*TI4_input)]

#print(TI4_list)
#print(x_list)

TI4_data = [[0, 1.600000], [1, 2.682060], [2, 3.412390], [3, 3.887100], [4, 4.359370], 
	    [5, 4.751090], [6, 5.032240], [7, 5.293740], [8, 5.556660], [9, 5.757480],
	    [10, 5.92025], [11, 6.07707], [12, 6.23797], [13, 6.36256], [14, 6.49109], 
	    [15, 6.62690], [16, 6.73067], [17, 6.82937], [18, 6.91624], [19, 6.99377], 
	    [20, 7.07259], [21, 7.14956], [22, 7.82060], [23, 8.10829], [24, 8.31126], 
	    [25, 8.48072], [26, 8.61333], [27, 8.72916], [28, 8.84857], [29, 8.94560],
	    [30, 9.04259], [31, 9.12841], [32, 9.21070], [33, 9.28734], [34, 9.34587],
	    [35, 9.40875], [36, 9.47995], [37, 9.53958], [38, 9.59331], [39, 9.60540], 
	    [40, 9.64929], [41, 9.69444], [42, 9.74420], [43, 9.79523], [44, 9.84048], 
	    [45, 9.88047], [46, 9.91621], [47, 9.95143], [48, 9.98317], [49, 10.0193], 
	    [50, 10.0590], [51, 10.0952], [52, 10.1247], [53, 10.2029], [54, 10.2520], 
	    [55, 10.2941], [56, 10.3552], [57, 10.4038], [58, 10.4455], [59, 10.4826], 
	    [60, 10.5254], [61, 10.5669], [62, 10.6049], [63, 10.6454], [64, 10.6846], 
	    [65, 10.7204], [66, 10.7487], [67, 10.7801], [68, 10.8072], [69, 10.8290], 
	    [70, 10.8583], [71, 10.8897], [72, 10.9224], [73, 10.9311]]

#print(TI4_data)

def fix_log_data(TI_data):
  """
  Corrects data if the first entry is an undefined Log.
  (Log[0] -> Undefined)
  Basically, shift days by 1
  """
  if TI_data[0][0]==0:
    for ind in list(range(len(TI_data))):
      TI_data[ind][0] = TI_data[ind][0]+1 
  return TI_data

def daily_growth(TI_data):
  growth = []
  for row in list(range(len(TI_data))):
    growth.append(TI_data[row][1]-TI_data[row-1][1])
  growth[0] = TI_data[0][0]
  return growth

#print(list(range(len(TI4_data))))
#daily_growth(TI4_data)

#print(growth)

#print(TI4_data) 

#fix_log_data(TI4_data)
#xdata,ydata = zip(*TI4_data)
#xdata = np.array(xdata)
#ydata = np.array(ydata)



def prepare_data(dataset):
  """
  Unzips datasets into separate x-y arrays.
  """
  xdata,ydata = zip(*dataset)
  xdata = np.array(xdata)
  ydata = np.array(ydata)
  return xdata,ydata
  
#print('xdata',prepare_data(TI4_data)[0])
#print('ydata',prepare_data(TI4_data)[1])
crossover_point = 21

def base_fit(dataset):
  """
  Fit for the first 'leg' of the data.
  """
  popt, pcov = curve_fit(base_log_func, prepare_data(dataset)[0][0:crossover_point], prepare_data(dataset)[1][0:crossover_point])
  base_fit_fn = np.array(list(zip(prepare_data(dataset)[0],base_log_func(prepare_data(dataset)[0], *popt))))
  base_variance = variance(base_fit_fn,dataset)
  return popt,pcov,base_fit_fn,base_variance

def cache_fit(dataset):
  popt_cache, pcov_cache = curve_fit(cache1, prepare_data(dataset)[0][crossover_point:], prepare_data(dataset)[1][crossover_point:])
  cache_fit_fn = np.array(list(zip(prepare_data(dataset)[0],cache1(prepare_data(dataset)[0], *popt_cache))))
  cache_variance = variance(cache_fit_fn,dataset)
  return popt_cache,pcov_cache,cache_fit_fn,cache_variance


#base_fit(TI4_data)

#def plot_base(dataset):
  #plt.figure('TI4')
  #plt.subplot(2,1,1)
  #plt.scatter(xdata,ydata,color='b',label='TI4 Data')
  #plt.plot(xdata, base_log_func(xdata, *popt), 'r-', label='fit')
  ##plt.plot(xdata, cache1(xdata, *popt_cache), 'r-', label='cache')
  #return 1

turning_point = base_fit(TI4_data)[2][21][1]
#print(cache_fit(TI4_data))
plt.figure('TI4')
plt.subplot(2,1,1)
plt.scatter(prepare_data(TI4_data)[0],prepare_data(TI4_data)[1],color='b',label='TI4 Data')
plt.plot(prepare_data(TI4_data)[0], base_log_func(prepare_data(TI4_data)[0], *base_fit(TI4_data)[0]), 'r-', label='fit')
plt.plot(prepare_data(TI4_data)[0], cache1(prepare_data(TI4_data)[0], *cache_fit(TI4_data)[0]), 'r-', label='cache')
plt.yticks(np.arange(0, max(prepare_data(TI4_data)[1])+1, 1))
plt.tick_params(labelbottom='off')
plt.xticks(np.arange(0, max(prepare_data(TI4_data)[0])+1, 5))
plt.ylabel('Millions of Dollars (USD)')
plt.grid(True)
plt.title('Dota 2 International Prizepool')
plt.legend(loc='lower right')


plt.subplot(2,1,2)
plt.ylabel('Millions of Dollars (USD)')
plt.xlabel('Day')
plt.title('Variance in Model Prediction vs. Data')
plt.scatter(prepare_data(TI4_data)[0][:crossover_point],base_fit(TI4_data)[3][:crossover_point],color='m',label='base')
plt.xticks(np.arange(0, max(prepare_data(TI4_data)[0])+1, 5))
plt.scatter(prepare_data(TI4_data)[0][crossover_point+1:],cache_fit(TI4_data)[3][22:],color='m',label='cache')
plt.grid(True)
plt.legend(loc='best')



#base fit
#popt, pcov = curve_fit(base_log_func, xdata[0:21], ydata[0:21])
#base_fit = np.array(list(zip(xdata,base_log_func(xdata, *popt))))
#base_variance = variance(base_fit,TI4_data)
#crossover_point = base_fit_fn[21][1]

##cache fit
#popt_cache, pcov_cache = curve_fit(cache1, xdata[21:], ydata[21:])
#print('Final Estimate:',round(cache1(xdata, *popt_cache)[-1],2),'Million USD')
#cache_fit = np.array(list(zip(xdata,cache1(xdata, *popt_cache))))
#cache_variance = variance(cache_fit,TI4_data)

##print(cache_variance)

#print('Base fit',popt)
#print('Base covariance',pcov)
#print('Cache fit',popt_cache)
#print('Cache covariance',np.array(pcov_cache))

##plotting
#plt.figure('TI4')
#plt.subplot(2,1,1)
#plt.scatter(xdata,ydata,color='b',label='TI4 Data')
#plt.plot(xdata, base_log_func(xdata, *popt), 'r-', label='fit')
#plt.plot(xdata, cache1(xdata, *popt_cache), 'r-', label='cache')
#plt.yticks(np.arange(0, max(ydata)+1, 1))
#plt.tick_params(labelbottom='off')
#plt.xticks(np.arange(0, max(xdata)+1, 5))
#plt.ylabel('Millions of Dollars (USD)')
##plt.xlabel('Day')
#plt.grid(True)
#plt.title('Dota 2 International Prizepool')
#plt.legend(loc='lower right')

##plt.figure(2)
#plt.subplot(2,1,2)
#plt.ylabel('Millions of Dollars (USD)')
#plt.xlabel('Day')
#plt.title('Variance in Model Prediction vs. Data')
#plt.scatter(xdata[:21],base_variance[:21],color='m',label='base')
#plt.scatter(xdata[22:],cache_variance[22:],color='m',label='cache')
#plt.xticks(np.arange(0, max(xdata)+1, 5))
#plt.grid(True)
#plt.legend(loc='best')


plt.figure('Daily Change')
plt.ylabel('Millions of Dollars (USD)')
plt.xlabel('Day')
plt.title('Daily Change in Prizepool')
plt.scatter(prepare_data(TI4_data)[0],daily_growth(TI4_data),color='c',label='TI4')
plt.legend(loc='best')

plt.show()

