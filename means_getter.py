#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'tk')


# In[2]:


SPATH = r"E:\NBR07_AutoFluxSweep_160mK\\"
dateOfMeasurement = 20220720
Temp = 160


eccosorb = True
KandL = True
rigidCables = True
JPA = False
TWPA = True
ClearingCoupler = False
sampleRateMHz = 5
durationSeconds = 3


# In[3]:


import numpy as np
import h5py
import fitTools.quasiparticleFunctions as qp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import glob
plt.rcParams.update({'axes.labelpad':0.2,
                     'axes.linewidth':1.0,
                     'figure.dpi':300.0,
                     'figure.figsize':[3.325,3.325],
                     'legend.frameon':True,
                     'legend.handlelength':1.0,
                     'xtick.major.pad':2,
                     'xtick.minor.pad':2,
                     'xtick.major.width':1.0,
                     'ytick.major.pad':2,
                     'ytick.minor.pad':2,
                     'ytick.major.width':1.0,
                     'axes.ymargin':0.01,
                     'axes.xmargin':0.01})
import os
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
from time import perf_counter, sleep
from hmmlearn import hmm
import pickle


# In[4]:



figurepath = os.path.join(SPATH,'autoPhi\Figures')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.size'] = 24

if not os.path.exists(figurepath):
    os.makedirs(figurepath)


# # We have about 200 different fluxes with a power sweep at each one. This may take a while to do manually, can we automate it? 

# In[8]:


MEANS_PHI = []
for phi in np.arange(0.5,0.3,-0.001):
    # grab files
    files = glob.glob(os.path.join(SPATH,f'{phi*1000:3.0f}flux*\*\*.bin'),recursive=True)

    # grab the digital attenuator settings and then make sorted arrays of files/powers
    attensunsorted = [int(fi.split('DA')[1][:2]) for fi in files]
    sortind = np.argsort(attensunsorted)
    attens = np.array(attensunsorted)[sortind]
    print(attens)
    filesH2L = np.array(files)[sortind]
    LOpower = 17 - 3 - 50 - 75 - 3 - np.array(attens)

    # grab information about the 0 peak freq and drive frequency from the VNA fits
    with open(os.path.join(SPATH,f'Figures/PHI_{phi*1000:3.0f}_fit.pkl'),'rb') as f:
        fitResults = pickle.load(f)
    pars = fitResults[0]
    if len(pars) == 6:
        f0 = pars[3]
        LOf = pars[3] - pars[4]
        Delta = pars[4]
    else:
        f0 = pars[1]
        Delta = 1.5*(qp.f_n_phi(phi, 0) - qp.f_n_phi(phi, 1))
        LOf = f0 - Delta
    print('PHI = {:.3f} -- LO is {:.6} Hz from f0'.format(phi,Delta*1e9))

    # import data and try fitting
    data = qp.loadAlazarData(filesH2L[5])
    data, sr = qp.BoxcarDownsample(data,2,sampleRateMHz,returnRate=True)
    data = qp.uint16_to_mV(data)
    
    h = qp.plotComplexHist(data[0],data[1],figsize=[4,4])
    plt.title(f'PHI = {phi:.3f}')
    means_guess = plt.ginput(2)
#     print(means_guess)
    plt.close()
    MEANS_PHI.append(means_guess)


# In[9]:


if not os.path.exists(os.path.join(SPATH,'autoPhi','AnalyisResults')):
    os.makedirs(os.path.join(SPATH,'autoPhi','AnalyisResults'))
with open(os.path.join(SPATH,'autoPhi','AnalyisResults','MeansGuess_DA15.pkl'),'wb') as f:
    pickle.dump(MEANS_PHI,f)


# In[ ]:




