# -*- coding: utf-8 -*-
"""
===============================================
Program : HMM_Analysis/AlazarPowerSweep.py
===============================================
Summary:

To Do:
    1) automate choosing of attenuation beyond which the resonator goes non-linear
"""
__author__ =  "Sadman Ahmed Shanto"
__date__ = "10/06/2022"
__email__ = "shanto@usc.edu"

#libraries used

import glob
import json
import multiprocessing
import os
import time
import warnings

import fitTools.quasiparticleFunctions as qp
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import psutil
from hmmlearn import hmm
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable

from HMM_helper_functions import *
from HMM_plotter_functions import *


class AlazarPowerSweepData:

    def __init__(self, project_path, interactive=True, project_root=None):
        self.project_path = project_path
        self.figure_path = r"{}\PowerSweepfigures\\".format(self.project_path)
        self.files = glob.glob(r"{}\**\*.bin".format(self.project_path),recursive=True)
        self.interactive = interactive
        self.project_root = project_root
        self.power_to_device = None
        self.attens = None
        self.index = None
        self.numModes = None
        self.metainfo = None
        self.HMM = None
        self.hdf5_file = None
        self.sampleRateFromData = None
        self.phi = None
        self.temp = None
        self.num_cores = None
        
        # HMM model parameters with defaults
        self.hmm_params = {
            'covariance_type': 'full',
            'n_iter': 500,        # Maximum number of iterations
            'tol': 0.001,         # Convergence tolerance
            'verbose': True,      # Show progress during fitting
        }
        
    def init_message(self):
        if not self.interactive:
            print("Please ensure the attenuation.json file is accurate.")
            print("Please ensure the static experimental parameters of the metadata.json file is accurate.\n\n")
        else:
            pass

    def set_attenuation_configuration(self):
        atten_config = json.load(open("attenuation.json"))
        atten_config_value = 0
        
        if self.interactive:
            
            print("Please Ensure the Following Attenuation Configuration is Correct:\n\n")
            for key,value in atten_config.items():
                print(key + " = " + str(value))
                atten_config_value += value
    
            isCorrect = input("Press Y/y if Correct or N/n if not: ")
    
            if isCorrect in ["y","Y","y\n","Y\n","yes","Yes","YES"]:
                power_to_device = atten_config_value - self.attens
            elif isCorrect in ["n","N","no","NO"]:
                print("Please update the `attenuation.json` file and re-run the function - <AlazarPowerSweepData Object>.start_HMM_fit(*args) .")
                quit()
            else:
                print("Incorrect Input. Try again. \n")
                self.set_attenuation_configuration()
        else:
                        
            print("Attenuation Configuration:\n\n")
            for key,value in atten_config.items():
                print(key + " = " + str(value))
                atten_config_value += value
            power_to_device = atten_config_value - self.attens
            
        return power_to_device

    def get_QP_means_from_IQ(self, avgTime):
        data = qp.loadAlazarData(self.files[self.index])
        data, sr = qp.BoxcarDownsample(data, avgTime, sampleRate=self.sampleRateFromData, returnRate=True)
        data = qp.uint16_to_mV(data)

        create_IQ_plot(data)
        means = plt.ginput(n = self.numModes)
        plt.close()
        return means, data


    def get_initial_QP_means(self, avgTime=3):
        if self.interactive:
            return self.get_QP_means_from_IQ(avgTime)
        else:
            try:
                return get_QP_means(self.project_root, self.phi, self.numModes), None
            except:
                return self.get_QP_means_from_IQ(avgTime)

    def estimate_initial_covariances(self, I, Q, initial_means):
        """
        Estimates initial covariance matrices based on the data and initial means.

        Args:
            I (numpy.ndarray): I quadrature data.
            Q (numpy.ndarray): Q quadrature data.
            initial_means (numpy.ndarray): Initial estimates for the Gaussian mode centers.

        Returns:
            numpy.ndarray: Initial estimates for the covariance matrices.
        """
        n_states = len(initial_means)
        initial_covariances = np.zeros((n_states, 2, 2))
        data = np.column_stack([I, Q])

        for i in range(n_states):
            # Calculate distances to the current mean
            distances = np.linalg.norm(data - initial_means[i], axis=1)

            # Select data points "close" to the mean (e.g., within 1 standard deviation)
            close_indices = np.where(distances < np.std(distances))
            close_data = data[close_indices]

            # If there are enough close points, calculate the covariance
            if len(close_data) > 10:  # Ensure we have enough points for a reasonable estimate
                initial_covariances[i] = np.cov(close_data.T)
            else:
                # Not enough points for this state, create a warning
                warnings.warn(f"Not enough data points close to mean {i+1} to estimate covariance. Using identity matrix.")
                initial_covariances[i] = np.eye(2)  # Identity matrix as neutral default

        return initial_covariances

    def get_initial_QP_covars(self, data=None, means=None):
        """
        Get initial covariance matrices for the HMM model.
        
        Uses data-driven approach to estimate covariances from I/Q data.
        Issues warnings and uses identity matrices if estimation isn't possible.
        
        Args:
            data (numpy.ndarray, optional): IQ data for covariance estimation
            means (numpy.ndarray, optional): Initial means for covariance estimation
            
        Returns:
            numpy.ndarray: Covariance matrices for each mode
        """
        # If we have both data and means, try to estimate covariances
        if data is not None and means is not None:
            try:
                print("Estimating covariances from data...")
                return self.estimate_initial_covariances(data[0], data[1], means)
            except Exception as e:
                warnings.warn(f"Error estimating covariances: {e}")
                warnings.warn("Falling back to identity matrices for covariance initialization")
                
        else:
            warnings.warn("No data or means provided for covariance estimation. Using identity matrices.")
                
        # Use identity matrices if estimation fails or inputs are missing
        dim_list = []
        for i in range(self.numModes):
            dim_list.append(np.eye(2))  # Identity matrix as neutral default
        return np.array(dim_list)


    def set_metadata(self):
        metainfo = json.load(open("metainfo.json"))
        if self.interactive:

            print("Please Ensure the Following MetaData is Correct:\n\n")
            for key,value in metainfo.items():
                print(key + " = " + str(value))
    
            isCorrect = input("Press Y/y if Correct or N/n if not: ")
    
            if isCorrect in ["y","Y","y\n","Y\n","yes","Yes","YES"]:
                pass
    
            elif isCorrect in ["n","N","no","NO"]:
                print("Please update the `metainfo.json` file and re-run the function - <AlazarPowerSweepData Object>.start_HMM_fit(*args) .")
                quit()
    
            else:
                print("Incorrect Input. Try again. \n")
                self.set_metadata()
        else:
            print("MetaData: \n\n")
            for key,value in metainfo.items():
                print(key + " = " + str(value))
            
        return metainfo


    def get_phi_sweep_and_sampleRate(self):
        self.files, self.attens = sort_files_ascending_attenuation(self.files)
        convert_to_json(self.files)
        self.phi = get_phi_from_run(self.files[0])
        self.sampleRateFromData = get_sample_rate_from_run(self.files[0])
        return self.phi, self.sampleRateFromData

        

    def process_Alazar_Data(self, avgTime=2, plots=True):
        print("Creating figure paths.....")
        create_path(self.figure_path)
        print("Reading and sorting data files.....")

        self.files, self.attens = sort_files_ascending_attenuation(self.files)
        convert_to_json(self.files)

        set_plot_style()
        print("Reading and updating the metadata.....")

        update_metainfo(self.files[0])

        self.sampleRateFromData = get_sample_rate_from_run(self.files[0])
        self.phi = get_phi_from_run(self.files[0])
        self.temp = get_temp_from_run(self.files[0])

        if plots:
            print("Creating IQ downsampled plots.....")            
            create_IQ_downsampled_plots(self.files, self.attens, self.project_path, avgTime)
        else:
            print("Data loaded without creating IQ downsampled plots....")            

    def start_HMM_fit(self, intTime=1, SNRmin=3, targetPower=None, numModes=2, n_jobs=None, 
                     covariance_type=None, n_iter=None, tol=None, verbose=None):
        print("\n\n"+"="*10+"\tHMM ANALYSIS STARTED\t"+"="*10)
        
        # Update HMM parameters if provided
        if covariance_type is not None:
            self.hmm_params['covariance_type'] = covariance_type
        if n_iter is not None:
            self.hmm_params['n_iter'] = n_iter
        if tol is not None:
            self.hmm_params['tol'] = tol
        if verbose is not None:
            self.hmm_params['verbose'] = verbose
            
        print(f"HMM parameters: {self.hmm_params}")
        
        # Determine number of cores to use
        if n_jobs is None:
            self.num_cores = psutil.cpu_count(logical=False)  # Use physical cores by default
        else:
            self.num_cores = n_jobs
            
        print(f"Processing will use {self.num_cores} CPU cores")

        if self.interactive:
            chosenAtten = int(input("\nAttenuation below which the system goes non-linear: "))
            self.power_to_device = self.set_attenuation_configuration()

            self.metainfo = self.set_metadata()

            self.index = int(np.where(self.attens == chosenAtten)[0])
            print("\nThe power to the device is {} dBM at the chosen attenuation {}".format(self.power_to_device[self.index], self.attens[self.index]))

            self.numModes = int(input("\nNumber of Modes you want to fit: "))
            set_qt_backend()

            means, data = self.get_initial_QP_means()
            covars = self.get_initial_QP_covars(data, means)
            print(f"Extracted Means:\n{means}\n\nEstimated Covariance:\n{covars}\n")
            print("\nStarting HMM Analysis.....\n\n")
            self.runHMM(means, covars, intTime, SNRmin)
        else:
            self.power_to_device = self.set_attenuation_configuration()
            self.metainfo = self.set_metadata()
            self.index = int(np.where(self.power_to_device == targetPower)[0])

            chosenAtten = self.attens[self.index]
            print("\nThe chosen power to the device is {} dBM at the attenuation {}".format(self.power_to_device[self.index], chosenAtten))

            self.numModes = numModes
            print(f"\nNumber of Modes to be fit: {self.numModes}")
            set_qt_backend()

            means, data = self.get_initial_QP_means()
            covars = self.get_initial_QP_covars(data, means)
            print(f"Extracted Means:\n{means}\n\nEstimated Covariance:\n{covars}\n")
            print("\nStarting HMM Analysis.....\n\n")
            self.runHMM(means, covars, intTime, SNRmin)

    def _process_single_file(self, i, atten, file, means, covars, intTime, SNRmin, skip, savefile, metainfo):
        """Process a single file with HMM analysis - designed for parallel processing"""
        print(f"Starting HMM fit for attenuation {atten} (process {os.getpid()})...")
        start_time = time.time()
        
        n_comp = self.numModes
        figpath = os.path.join(self.figure_path, f'ATTEN{atten}')
        if not os.path.exists(figpath):
            os.makedirs(figpath)
            
        # Load and process data
        data = qp.loadAlazarData(file)
        data, sr = qp.BoxcarDownsample(data, avgTime=intTime, sampleRate=self.sampleRateFromData, returnRate=True) 
        data = qp.uint16_to_mV(data)
        
        # Fit the HMM using the alternative approach with parameterized values
        M = hmm.GaussianHMM(n_components=n_comp, 
                            covariance_type=self.hmm_params['covariance_type'],
                            n_iter=self.hmm_params['n_iter'],
                            tol=self.hmm_params['tol'],
                            init_params="",  # No automatic initialization
                            verbose=self.hmm_params['verbose'])
                            
        # Manual initialization of all parameters
        M.means_ = means
        M.covars_ = covars
        M.startprob_ = np.ones(n_comp) / n_comp  # Equal starting probabilities
        
        # Set transition matrix based on number of components
        if n_comp == 3:
            M.transmat_ = np.array([[0.99, 0.009, 0.001], [0.03, 0.95, 0.02], [0.05, 0.05, 0.9]])
        else:
            M.transmat_ = np.array([[0.99, 0.01], [0.01, 0.99]])

        # Fit the model
        print(f"Fitting HMM for attenuation {atten}...")
        M.fit(data.T)
        print(f"HMM fitting completed for attenuation {atten}")
        
        # Read previous data for comparison
        with h5py.File(savefile, 'r') as ff:
            oldrates = ff[f'ATTEN{self.attens[i+skip-1]}/transitionRatesMHz'][:] if i != 0 else 0.0001*np.ones((n_comp, n_comp))
            lifetimes = np.array([1/oldrates[j,j] for j in range(n_comp)])
            t01 = 1/oldrates[0,1]
            t10 = 1/oldrates[1,0]
            ttimes = np.array([t01, t10])

        # Check SNR
        snr01 = qp.getSNRhmm(M, mode1=0, mode2=1)
        SNRs = np.array([snr01,])

        logprob, Q = M.decode(data.T)
        Qmean = np.mean(Q)
        P0 = np.sum(Q == 0)/Q.size
        P1 = np.sum(Q == 1)/Q.size
        P2 = np.sum(Q == 2)/Q.size if n_comp > 2 else 0
        
        # Increase integration time if needed
        current_intTime = intTime
        if np.min(SNRs) < SNRmin and current_intTime <= np.min(lifetimes)/2 and current_intTime <= np.min(ttimes) and P1 < P0:
            success = False
            srold = np.copy(sr)
            while not success:
                current_intTime *= 1.15*SNRmin/np.min(SNRs)
                print(f'\n\n\nNew integration time = {current_intTime} because SNR was {np.min(SNRs):.6}\n\n\n')
                data = qp.loadAlazarData(file)
                srold = np.copy(sr)
                data, sr = qp.BoxcarDownsample(data, current_intTime, self.sampleRateFromData, returnRate=True)
                data = qp.uint16_to_mV(data)
                
                # Create new model with increased integration time
                M = hmm.GaussianHMM(n_components=n_comp, 
                                   covariance_type=self.hmm_params['covariance_type'],
                                   n_iter=self.hmm_params['n_iter'],
                                   tol=self.hmm_params['tol'],
                                   init_params="",
                                   verbose=self.hmm_params['verbose'])
                
                # Get parameters from previous run if available
                with h5py.File(savefile, 'r') as ff:
                    M.means_ = ff[f'ATTEN{self.attens[i+skip-1]}'].attrs.get('HMMmeans_') if i != 0 else means
                    M.covars_ = ff[f'ATTEN{self.attens[i+skip-1]}'].attrs.get('HMMcovars_') if i != 0 else covars
                
                # Initialize other parameters
                M.startprob_ = np.ones(n_comp) / n_comp
                
                if n_comp == 3:
                    M.transmat_ = np.array([[0.99, 0.009, 0.001], [0.03, 0.95, 0.02], [0.05, 0.05, 0.9]])
                else:
                    M.transmat_ = np.array([[0.99, 0.01], [0.01, 0.99]])
                
                M.fit(data.T)
                
                # Get SNR
                snr01 = qp.getSNRhmm(M, mode1=0, mode2=1)
                SNRs = np.array([snr01,])
                if np.min(SNRs) > SNRmin:
                    success = True
                elif current_intTime > np.min(lifetimes)/2 or current_intTime > np.min(ttimes):
                    print('Integration time has grown too large.')
                    break
                if sr == srold:
                    print('stuck in loop, exiting')
                    break
                srold = np.copy(sr)

        # Check conditions
        if current_intTime > np.min(lifetimes)/2 and i > 3:
            print('Integration time has grown too large.')
            return None
        if P1 > P0:
            print('P1 became larger than P0!')
            return None

        # Plot the fit (offline plotting with Agg backend)
        plt.figure(figsize=[4, 4])
        h = qp.plotComplexHist(data[0], data[1], figsize=[4, 4])
        if n_comp == 3:
            qp.make_ellipsesHMM(M, h, ['purple', 'orange', 'green'])
        else:
            qp.make_ellipsesHMM(M, h, ['purple', 'orange'])
        plt.xlabel('I [mV]')
        plt.ylabel('Q [mV]')
        plt.title('HMM fit | {:.2} MHz | {} dBm'.format(sr, self.power_to_device[i+skip]))
        plt.savefig(os.path.join(figpath, 'HMMfits_{}_{}dBm_{}modes.png'.format(i+skip, self.power_to_device[i+skip], self.numModes)))
        plt.close()

        # Plot time series
        fig, ax = qp.plotTimeSeries(data, Q, np.arange(Q.size)/sr, 1500, 2000, zeroTime=True)
        plt.title('{:.2} MHz | {} dBm'.format(sr, self.power_to_device[i+skip]))
        plt.savefig(os.path.join(figpath, 'TimeSeries__{}_{}dBm_{}modes.png'.format(i+skip, self.power_to_device[i+skip], self.numModes)))
        plt.close()

        # Get transition rates
        rates = qp.getTransRatesFromProb(sr, M.transmat_)

        # Save results to HDF5
        with h5py.File(savefile, 'a') as ff:
            fp = ff.require_group(f'ATTEN{atten}')
            fp.create_dataset('Q', data=Q)
            fp.create_dataset('data', data=data)
            fp.create_dataset('transitionRatesMHz', data=rates)
            fp.attrs.create('logprobQ', logprob)
            fp.attrs.create('mean', Qmean)
            fp.attrs.create('P0', P0)
            fp.attrs.create('P1', P1)
            try:
                fp.attrs.create('P2', P2)
            except:
                pass
            fp.attrs.create('SNRs', SNRs)
            fp.attrs.create('downsampleRateMHz', sr)
            fp.attrs.create('HMMmeans_', M.means_)
            fp.attrs.create('HMMstartprob_', M.startprob_)
            fp.attrs.create('HMMcovars_', M.covars_)
            fp.attrs.create('HMMtransmat_', M.transmat_)
            fp.attrs.create('LOpower', self.power_to_device[i+skip])
            fp.attrs.create('DAsetting', self.attens[i+skip])

            for key in metainfo:
                fp.attrs.create(key, metainfo[key])

        elapsed = time.time() - start_time
        print(f"Completed HMM fit for attenuation {atten} in {elapsed:.2f} seconds")
        return M

    def runHMM(self, means, covars, intTime=1, SNRmin=3):
        matplotlib.use('Agg')  # Use non-interactive backend for parallel processing

        skip = np.copy(self.index)
        n_comp = self.numModes
        
        # Create PDF files
        hmm_fits_pdf = PdfPages('{}/HMM_IQ_fits_{}modes.pdf'.format(self.project_path, self.numModes))
        hmm_time_series_pdf = PdfPages('{}/HMM_time_series_{}modes.pdf'.format(self.project_path, self.numModes))
        
        # Set up HDF5 file
        savefile = os.path.join(self.project_path, 'AnalyisResults', 'FullDataset_M{}_T{}_PHI{}_.hdf5'.format(self.numModes, self.temp, str(self.phi).replace(".", "p")[:5]))
        self.hdf5_file = savefile
        if not os.path.exists(os.path.split(savefile)[0]):
            os.makedirs(os.path.split(savefile)[0])
            
        # Set up initial HDF5 structure
        with h5py.File(savefile, 'a') as f:
            for atten in self.attens[self.index:]:
                g = f.require_group(f'ATTEN{atten}')
                for key in self.metainfo:
                    g.attrs.create(key, self.metainfo[key])
        
        # Prepare data for parallel processing
        files_to_process = []
        for i, atten, file in zip(np.arange(len(self.attens[self.index:])), self.attens[self.index:], self.files[self.index:]):
            files_to_process.append((i, atten, file))
        
        print(f"Starting parallel HMM processing on {self.num_cores} cores for {len(files_to_process)} files...")
        
        # Process files sequentially but with parallel inner operations 
        # (full parallelization might not work well due to dependencies between iterations)
        HMM = []
        current_means = means
        current_covars = covars
        
        for i, atten, file in files_to_process:
            print(f"Processing file {i+1}/{len(files_to_process)}: attenuation {atten}")
            result = self._process_single_file(i, atten, file, current_means, current_covars, intTime, SNRmin, skip, savefile, self.metainfo)
            
            if result is not None:
                HMM.append(result)
                # Update for next iteration
                current_means = result.means_
                current_covars = result.covars_
            else:
                print(f"Stopping at attenuation {atten} due to conditions not met")
                break
        
        print("All HMM fits created.....")
        hmm_fits_pdf.close()
        hmm_time_series_pdf.close()
        self.HMM = HMM
        
        print("Starting post-HMM analysis plots.....")
        create_HMM_QP_statistics_plots(self.hdf5_file, self.figure_path, self.numModes)
        print("="*10+"\tHMM ANALYSIS CONCLUDED\t"+"="*10+"\n\n")
