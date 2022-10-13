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
import AlazarPowerSweepData as alazar

def get_all_phis_and_sampleRate(project_path):
    flux_sweeps = get_all_project_folders(project_path)

    phi_sweep = []
    for power_sweeps in flux_sweeps:
        power_sweep_obj = alazar.AlazarPowerSweepData(power_sweeps,interactive=False)
        phi, sampleRateMHz = power_sweep_obj.get_phi_sweep_and_sampleRate()
        phi_sweep.append(phi)
    
    return phi_sweep, sampleRateMHz

def set_all_initial_QP_means(project_path, target_device_power=-127, numModes=2, avgTime=2):
    
    phi_sweep, sampleRateMHz = get_all_phis_and_sampleRate(project_path)
    print(phi_sweep)    
    create_QP_means(project_path, phi_sweep, target_device_power, numModes, sampleRateMHz, avgTime)
    print("All initial means have been set!")



def create_dir(path,new_folder):
    figurepath = os.path.join(path,new_folder)
    if not os.path.exists(figurepath):
        os.makedirs(figurepath)


def get_phi_sweep_array():
    raise NotImplementedError()



def get_power_to_device(attens):
    atten_config = json.load(open("attenuation.json"))
    atten_config_value = 0

    #print("Attenuation Configuration:\n\n")
    for key,value in atten_config.items():
        atten_config_value += value
    power_to_device = atten_config_value - attens

    return power_to_device

def get_freqs_from_VNA():
    # grab information about the 0 peak freq and drive frequency from the VNA fits
    with open(os.path.join(project_path,f'Figures/PHI_{phi*1000:3.0f}_fit.pkl'),'rb') as f:
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
    raise DeprecationWarning()


def create_QP_means(project_path, phi_sweep, targetDevPower, numModes=2, sampleRateMHz=10, avgTime=2):
    create_dir(project_path, 'AnalysisResults\guessedMeans\Figures')
    flux_sweeps = get_all_project_folders(project_path)

    means_phi = []

    for i, phi in enumerate(phi_sweep):

        # grab files
        # grab the digital attenuator settings and then make sorted arrays of files/powers

        files = glob.glob(r"{}\**\*.bin".format(flux_sweeps[i]),recursive=True)
        # files = glob.glob(os.path.join(project_path,f'{phi*1000:3.0f}flux*\*\*.bin'),recursive=True)
        files, attens = sort_files_ascending_attenuation(files)
       # print("attens : {}\n".format(attens))

        power_to_device = get_power_to_device(attens)

       # print("\npower_to_device : {}".format(power_to_device))
        index = int(np.where(power_to_device == targetDevPower)[0])

        
        # import data and try fitting
        data = qp.loadAlazarData(files[index])
        data, sr = qp.BoxcarDownsample(data,avgTime,sampleRateMHz,returnRate=True)
        data = qp.uint16_to_mV(data)
        
        set_qt_backend()
        h = qp.plotComplexHist(data[0],data[1],figsize=[4,4])
        plt.title(f'PHI = {phi:.3f}')
        means_guess = plt.ginput(numModes)
        plt.close()
        print(f"Chosen Means:\n{means_guess}")
        means_phi.append((phi,means_guess))

    if not os.path.exists(os.path.join(project_path, 'AnalysisResults', 'guessedMeans')):
        os.makedirs(os.path.join(project_path, 'AnalysisResults', 'guessedMeans'))
    with open(os.path.join(project_path, 'AnalysisResults', 'guessedMeans','QP_init_means.pkl'),'wb') as f:
        pickle.dump(means_phi,f)


def get_QP_means(project_path, phi):
    with open(os.path.join(project_path, 'AnalysisResults', 'guessedMeans','QP_init_means.pkl'), 'rb') as f:
        means_phi = pickle.load(f)
    for i in range(len(means_phi)):
        if means_phi[i][0] == phi:
            print(f"At Phi = {phi}")
            return means_phi[i][1]
        else:
            raise ValueError()

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
            if os.path.isfile(file.replace(".txt",".json")):
                return
            else:
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


def update_metainfo(file):
    file = file.split(".")[0] + ".json"
    data = json.load(open(file))
    LOf = float(data["LO_frequency"])
    temp = float(data["Temperature"])
    sampleRate = int(float(data["Sample_Rate_MHz"]))
    phi = float(data["PHI"])
    durationSeconds = int(data["Acquisition_duration"])
        
    with open("metainfo.json", "r") as jsonFile:
        data = json.load(jsonFile)
    
    data["Temp"] = temp
    data["sampleRateMHz"] = sampleRate
    data["durationSeconds"] = durationSeconds
    data["LOf"] = LOf
    data["phi"] = phi
    
    with open("metainfo.json", "w") as jsonFile:
        json.dump(data, jsonFile,  indent = 4, sort_keys = False)

def get_phi_from_run(file):
    file = file.split(".")[0] + ".json"
    return float(json.load(open(file))["PHI"])

def get_temp_from_run(file):
    file = file.split(".")[0] + ".json"
    return float(json.load(open(file))["Temperature"])

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
