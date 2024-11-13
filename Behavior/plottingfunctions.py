import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from numpy.random import randn
from numpy.random import seed
from numpy import mean
from numpy import var
from math import sqrt

def plot_columns(ax, df, column_names):
    for n, i in enumerate(column_names):
        print(i)
        x1 = df[df['VR']=='VR1'][i]
        x2 = df[df['VR']=='VR2'][i]
        print(x1.shape, x2.shape)
        
        t, p = scipy.stats.ttest_rel(x1, x2)
        print('tstat: %0.3f, p-value: %0.3f' %(t, p))
        
        VR = ['VR1', 'VR2']
        # print(i)
        for n1, x in enumerate([x1, x2]):
            mean, ci = mean_confidence_interval(x)
            print('VR: %s\n Mean: %0.2f\n Median: %0.2f\n STD: %0.2f\n CI: %0.2f' %(VR[n1], mean, np.median(x), np.std(x), ci))
        sns.boxplot(data=df, x='VR', order=['VR1', 'VR2'], y=i, ax=ax[n], showfliers=False, fill=False)
        ax[n].plot([0.25, 1.25], [x1, x2], 'k.-', markersize=6, markerfacecolor='white', alpha=0.5)

def plot_delta_columns(ax, df, column_names):
    #Skip D7, Thy5-3 as it does not have before data
    df = df[~df['Animal'].str.contains('|'.join(['D7', 'Thy5-3']))]
    for n, i in enumerate(column_names):
        sub_data = df[df['Paradigm']=='Before']
        x1 = np.asarray(sub_data[sub_data['VR']=='VR1'][i])
        x2 = np.asarray(sub_data[sub_data['VR']=='VR2'][i])
       
        delta_bef = x2-x1

        sub_data = df[df['Paradigm'].str.contains('RecallDay1')]
        x1 = np.asarray(sub_data[sub_data['VR']=='VR1'][i])
        x2 = np.asarray(sub_data[sub_data['VR']=='VR2'][i])
        
        delta_after = x2-x1
        
        print(np.sort(delta_after))

        ax[n].plot([0, 1], [delta_bef, delta_after], '+-', color='grey', markersize=8, markerfacecolor='white')
        ax[n].axhline(0, linewidth=2, linestyle='--', color='k')
    ax[0].set_ylabel('∆ freezing')
    return delta_after

def plot_freezing_bylap(sub_data, ax, numlaps, legend, colname='Freezingpercent_bylap',heather_data=False):
    array = np.asarray([])
    for n, i in enumerate(sub_data[colname]):
        if heather_data:
            if len(i)>=numlaps+1:
                array = np.vstack((array, np.array(i[1:numlaps+1]))) if array.size else np.array(i[1:numlaps+1])
        else:
            if len(i)>=numlaps:
                array = np.vstack((array, np.array(i[:numlaps]))) if array.size else np.array(i[:numlaps])

    m, ci = [], []
    for i in range(10):
        this_m, this_ci = mean_confidence_interval(array[:, i])
        m.append(this_m)
        ci.append(this_ci)
    m, ci = np.array(m), np.array(ci)
    ax.plot(range(numlaps), m, '.-', label=legend, markersize=7)
    ax.fill_between(range(numlaps), m-ci, m+ci, alpha=0.5)

    return array
# function to calculate Cohen's d for independent samples
def cohend(d1, d2):
	n1, n2 = len(d1), len(d2)
	s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
	s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	u1, u2 = mean(d1), mean(d2)
	return (u1 - u2) / s

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.nanmean(a), scipy.stats.sem(a, nan_policy='omit')
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h
