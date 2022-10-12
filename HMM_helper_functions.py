import subprocess
from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt
import fitTools.quasiparticleFunctions as qp
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
import h5py
import glob
import os
import pickle
import json
import matplotlib


def set_qt_backend():
    try:
        matplotlib.use('Qt5Agg')
    except:
        try:
            matplotlib.use('Qt4Agg')
        except:
            matplotlib.use('QtAgg')

def convert_to_json(files):
    
    for file in files:
        try:
            
            file = file.replace(".bin",".txt") 
            dict1 = {}
    
            with open(file) as fh:
                for line in fh:
                    try:
                        command, description = line.strip().split(":")
                        command = command.replace(" ","_")
                        value = description.strip().split(" ")[0]
                        dict1[command] = value
                    except:
                        pass
    
            file_name = file.replace(".txt",".json") 
            out_file = open(file_name, "w")
            json.dump(dict1, out_file, indent = 4, sort_keys = False)
            out_file.close()
       
        except:
            pass


def get_all_project_folders(project_path):
    folders = glob.glob(project_path+"\*flux")
    return folders

def pickle_HMM(HMM, fdir):
    for i,M in enumerate(HMM):
        fp = r"{}\\HMM_DA_index_{}.pkl".format(fdir, i)
        path,fname = os.path.split(fp)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(fp,'wb') as f:
            pickle.dump(M,f)


def get_sample_rate_from_run(file):
    file = file.split(".")[0] + ".json"
    return int(float(json.load(open(file))["Sample_Rate_MHz"]))

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
    matplotlib.use('Agg')
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
    try:
        for file in files:
            _,s = file.split('_DA')
            atten = s[:2]
            attens.append(int(atten))
        sortind = np.argsort(attens)
        files = np.asarray(files)[sortind[::-1]]
        attens = np.asarray(attens)[sortind[::-1]]
    except:
        for file in files:
            atten = file.split('DA')[-1].split("_")[0]
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



def create_IQ_downsampled_plots(files, attens, base_dir, avgTime=2, sampleTime=10):
    pdfName = '{}/IQ_downsampled_plots_{}_{}.pdf'.format(base_dir, avgTime, sampleTime)

    pp = PdfPages(pdfName)
    for i,file in enumerate(files):
        data = qp.loadAlazarData(file)
        data = qp.BoxcarDownsample(data,avgTime,sampleTime)
        data = qp.uint16_to_mV(data)

        qp.plotComplexHist(data[0],data[1])
        plt.title(f'Atten = {attens[i]} dB')
        # plt.show()
        pp.savefig(plt.gcf())            #Save each figure in pdf
        plt.close()

    pp.close()                           #close the pdf

    # os.startfile(pdfName)

    subprocess.Popen([pdfName],shell=True)
    print("\nPlease review the downsampled IQ plots - {}".format(pdfName))


def create_IQ_plot(data):
    qp.plotComplexHist(data[0],data[1])
    plt.show()
