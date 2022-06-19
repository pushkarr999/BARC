from scipy.signal import savgol_filter
import numpy as np
import statsmodels.api as sm

def savgol(data):
    for columns in data.iloc[:, 1:]:
        data[columns] = savgol_filter(data[columns], 101, 2)
    return data

def convolve(data):
    kernel_size = 20
    kernel = np.ones(kernel_size) / kernel_size
    for columns in data.iloc[:,1:]:
        data[columns] = np.convolve(data[columns], kernel, mode='same')
    return data

def smooth_data_np_average(data):  # my original, naive approach
    span = 20
    for columns in data.iloc[:,1:]:
        data[columns] = [np.average(data[columns][val - span:val + span + 1]) for val in range(len(data[columns]))]
    return data

def smooth_data_lowess(data):
    span = 20
    for columns in data.iloc[:,1:]:
        x = np.linspace(0, 1, len(data[columns]))
        data[columns] = sm.nonparametric.lowess(data[columns], x, frac=(5*span / len(data[columns])), return_sorted=False)
    return data

def exp_avg(data):
    span = 20
    for columns in data.iloc[:,1:]:
        data[columns] = data[columns].ewm(span=span).mean()
    return data



