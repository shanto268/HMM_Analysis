# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 09:25:37 2022

@author: shanto
"""

# Standard library imports
import glob
import json
import os
import pickle
import re
import subprocess
import sys

# Custom modules
import fitTools.quasiparticleFunctions as qp
# Third-party imports
import h5py
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from HMM_plotter_functions import *


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
        h = qp.plotComplexHist(data[0],data[1],figsize=[8,8])
        try:
            plt.title(f'PHI = {phi:.3f}')
        except:
            plt.title(f'PHI')
        means_guess = plt.ginput(numModes,timeout=120)
        figname = f"IQ_plot_P{phi:.3f}_M{numModes}_SR{sampleRateMHz}_DP{targetDevPower}_DA{attens[index]}".replace(".","p")
        plt.savefig(os.path.join(project_path, f'AnalysisResults\guessedMeans\Figures\{figname}.png'))
        plt.close()
        print(f"Chosen Means:\n{means_guess}")
        means_phi.append((phi,means_guess))

    if not os.path.exists(os.path.join(project_path, 'AnalysisResults', 'guessedMeans')):
        os.makedirs(os.path.join(project_path, 'AnalysisResults', 'guessedMeans'))
    with open(os.path.join(project_path, 'AnalysisResults', 'guessedMeans',f'QP_init_means_M{numModes}.pkl'),'wb') as f:
        pickle.dump(means_phi,f)


def get_QP_means(project_path, phi, numModes):
    with open(os.path.join(project_path, 'AnalysisResults', 'guessedMeans',f'QP_init_means_M{numModes}.pkl'), 'rb') as f:
        means_phi = pickle.load(f)
    for i in range(len(means_phi)):
        if means_phi[i][0] == phi:
            print(f"At Phi = {phi}")
            return means_phi[i][1]


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
    try:
        temp = float(data["Temperature_MXC"])
    except:
        temp = float(data["Temperature"])
    sampleRate = int(float(data["Sample_Rate_MHz"]))
    try:
        phi = float(data["PHI"])  
    except:
        phi = None
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
    try:
        return float(json.load(open(file))["Flux_bias_(Phi)"])
    except:
        return float(json.load(open(file))["PHI"])
    
def get_temp_from_run(file):
    file = file.split(".")[0] + ".json"
    try:
        return float(json.load(open(file))["Temperature_MXC"])
    except:
        return float(json.load(open(file))["Temperature"])

def get_sample_rate_from_run(file):
    file = file.split(".")[0] + ".json"
    return int(float(json.load(open(file))["Sample_Rate_MHz"]))


def create_path(target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)

def set_plot_style():
    matplotlib.use('Agg')
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.facecolor'] = 'white'
  #  plt.rcParams['figure.constrained_layout.use'] = True
    plt.style.use(['science','no-latex'])
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
   # plt.tight_layout()


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

def get_IQ_data(files, avgTime=2):
    """
    Load and process IQ data from a file without plotting.
    
    Args:
        files (list): List of data files
        avgTime (float): Time in microseconds to average data for downsampling
        
    Returns:
        tuple: (data, sr) where data is the processed IQ data in mV and sr is the sample rate in MHz,
               or (None, None) if an error occurs
    """
    all_data = []
    all_sr = []
    
    for file in files:
        try:
            # Load and downsample data
            data = qp.loadAlazarData(file)
            sampleRateFromData = get_sample_rate_from_run(file)
            data, sr = qp.BoxcarDownsample(data, avgTime, sampleRateFromData, returnRate=True)
            data = qp.uint16_to_mV(data)
            
            all_data.append(data)
            all_sr.append(sr)

        except Exception as e:
            print(f"Error loading or processing data: {str(e)}")

    return all_data, all_sr

def create_IQ_downsampled_plots(files, attens, base_dir, avgTime=2, sampleTime=10):
    """
    Create downsampled IQ plots for each file.
    
    Args:
        files (list): List of data files
        attens (list): List of attenuation values
        base_dir (str): Base directory for saving plots
        avgTime (float): Time in microseconds to average data for downsampling
        sampleTime (float): Sample time in seconds
    """
    try:
        print(f"Creating downsampled IQ plots in {base_dir}...")
        for i, (file, atten) in enumerate(zip(files, attens)):
            try:
                # Load and downsample data
                data = qp.loadAlazarData(file)
                sampleRateFromData = get_sample_rate_from_run(file)
                data, sr = qp.BoxcarDownsample(data, avgTime, sampleRateFromData, returnRate=True)
                data = qp.uint16_to_mV(data)
                
                # Create plot with direct matplotlib commands
                fig, ax = plt.subplots(figsize=(8, 8))
                
                # Create 2D histogram
                h = ax.hist2d(data[0], data[1], bins=80, 
                           norm=matplotlib.colors.LogNorm(), 
                           cmap=plt.cm.Greys)
                plt.colorbar(h[3], ax=ax, shrink=0.9, extend='both')
                
                # Add grid and set aspect ratio
                ax.grid(True)
                ax.set_aspect('equal')
                
                # Add labels and title
                ax.set_xlabel('I [mV]', fontsize=12)
                ax.set_ylabel('Q [mV]', fontsize=12)
                ax.set_title(f'I-Q Plot | Attenuation: {atten} | Sample Rate: {sr:.2f} MHz', fontsize=14)
                
                # Use subplots_adjust instead of tight_layout
                plt.subplots_adjust(right=0.85, top=0.9, bottom=0.1, left=0.1)
                
                # Save the figure
                plt.savefig(os.path.join(base_dir, f'iq_plot_{i}_atten{atten}.png'), 
                          bbox_inches='tight', dpi=150)
                plt.close(fig)
                
                print(f"Created IQ plot for attenuation {atten} ({i+1}/{len(files)})")
            except Exception as e:
                print(f"Error creating plot for attenuation {atten}: {str(e)}")
                plt.close('all')  # Make sure to close any open plots on error
                
    except Exception as e:
        print(f"Error in create_IQ_downsampled_plots: {str(e)}")
        plt.close('all')


def create_IQ_plot(data):
    """Create an IQ plot for manual mean selection with improved error handling.
    
    Args:
        data: IQ data to plot
        
    Returns:
        None, displays a plot for interaction
    """
    try:
        # Create figure with explicit dimensions
        plt.figure(figsize=(10, 10))
        
        # Create a 2D histogram rather than using plotComplexHist
        # This gives us more direct control over the plot
        h = plt.hist2d(data[0], data[1], bins=80, 
                     norm=matplotlib.colors.LogNorm(), 
                     cmap=plt.cm.Greys)
        plt.colorbar(h[3], shrink=0.9, extend='both')
        
        # Add grid and set aspect ratio
        plt.grid(True)
        plt.gca().set_aspect('equal')
        
        # Add labels and title
        plt.xlabel('I [mV]', fontsize=12)
        plt.ylabel('Q [mV]', fontsize=12)
        plt.title('Click to select initial means for each state', fontsize=14)
        
        # Use subplots_adjust instead of tight_layout
        plt.subplots_adjust(right=0.9, top=0.9, bottom=0.1, left=0.1)
        
        # Display the plot
        plt.show()
    except Exception as e:
        print(f"Error creating IQ plot: {str(e)}")
        plt.close('all')
        # If plotting fails, create a minimal fallback plot
        plt.figure(figsize=(10, 10))
        plt.scatter(data[0], data[1], s=1, alpha=0.5)
        plt.xlabel('I [mV]')
        plt.ylabel('Q [mV]')
        plt.title('Fallback plot - click to select initial means')
        plt.grid(True)
        plt.show()
        plt.show()
