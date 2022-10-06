
from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt
import fitTools.quasiparticleFunctions as qp
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
import glob
import os


def create_HMM_QP_statistics_plots(hdf5_file):
    create_mean_occupation_plot(hdf5_file)
    create_transition_lifetimes_plot(hdf5_file)
    create_transition_probability_plot(hdf5_file)

def create_mean_occupation_plot(hdf5_file):
    raise NotImplementedError()

def create_transition_probability_plot(hdf5_file):
    raise NotImplementedError()

def create_transition_lifetimes_plot(hdf5_file):
    raise NotImplementedError()

def create_path(target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)

def set_plot_style():
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.facecolor'] = 'white'

    plt.rcParams.update({'axes.labelpad':0.2,
                        'axes.linewidth':1.0,
                        'figure.dpi':300.0,
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


def sort_files_descending_attenuation(files):
    attens = []
    for file in files:
        _,s = file.split('_DA')
        atten = s[:2]
        attens.append(int(atten))
    sortind = np.argsort(attens)
    files = np.asarray(files)[sortind[::-1]]
    attens = np.asarray(attens)[sortind[::-1]]
    return files, attens


def sort_files_ascending_attenuation(files):
    files, attens = sort_files_descending_attenuation(files)
    files = np.flip(files,0)
    attens = np.flip(attens,0)
    return files, np.array(attens)



def create_IQ_downsampled_plots(files, avgTime=2, sampleTime=10):
    """
    TO DO: create pdf and show pdf
    """
    for i,file in enumerate(files):
        data = qp.loadAlazarData(file)
        data = qp.BoxcarDownsample(data,avgTime,sampleTime)
        data = qp.uint16_to_mV(data)

        qp.plotComplexHist(data[0],data[1])
        plt.title(f'Atten = {attens[i]} dB')
        plt.show()


def create_IQ_plot(data):
    qp.plotComplexHist(data[0],data[1])
    plt.show()
