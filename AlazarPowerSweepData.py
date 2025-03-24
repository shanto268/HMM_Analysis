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
import sys
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

# GPU acceleration is disabled by default - set to True only if you have cuML properly installed
USE_GPU = False

# Check if hmmlearn supports n_jobs parameter (newer versions do)
HMM_SUPPORTS_PARALLEL = False
try:
    hmm.GaussianHMM(n_components=2, n_jobs=1)
    HMM_SUPPORTS_PARALLEL = True
    print("Using parallel-enabled HMM implementation")
except TypeError:
    print("Your hmmlearn version doesn't support parallel processing. Using single-core HMM.")

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
            'n_iter': 150,        # Maximum number of iterations
            'tol': 0.001,         # Convergence tolerance
            'verbose': True,      # Show progress during fitting
            'transition_model': 'physics'  # 'simple' or 'physics'
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

    def set_method_attributes(self):
        print("Setting method attributes.....")
        self.files, self.attens = sort_files_ascending_attenuation(self.files)
        convert_to_json(self.files)

        self.sampleRateFromData = get_sample_rate_from_run(self.files[0])
        self.phi = get_phi_from_run(self.files[0])
        self.temp = get_temp_from_run(self.files[0])

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

    def _create_transition_matrix(self, n_components):
        """
        Create an appropriate transition matrix based on the selected model.
        
        Args:
            n_components (int): Number of states/components in the HMM
            
        Returns:
            numpy.ndarray: Transition matrix with probabilities
        """
        if self.hmm_params['transition_model'] == 'simple':
            # Simple uniform transition matrix
            transmat = np.ones((n_components, n_components)) * 0.01 / (n_components - 1)  # Small transition probability
            np.fill_diagonal(transmat, 0.99)  # High probability to stay in the same state
            return transmat
            
        elif self.hmm_params['transition_model'] == 'physics':
            # Physics-informed transition matrix that models quasiparticle dynamics
            transmat = np.zeros((n_components, n_components))
            
            # Diagonal elements (staying in the same state) have high probability
            for i in range(n_components):
                # Probability of staying in the same state decreases as state number increases
                # Higher states (more quasiparticles) are less stable
                transmat[i, i] = 0.99 - (i * 0.01)
            
            # Off-diagonal elements (transitions to other states)
            for i in range(n_components):
                remaining_prob = 1.0 - transmat[i, i]
                for j in range(n_components):
                    if i != j:
                        # Higher probability to transition to adjacent states
                        # and lower probability for distant states
                        if j > i:
                            # Transition to higher states (more QPs) - less likely for higher states
                            transmat[i, j] = remaining_prob * (0.8 / (j - i)) / (n_components - 1)
                        else:
                            # Transition to lower states (fewer QPs) - more likely for higher states
                            transmat[i, j] = remaining_prob * (1.2 / (i - j + 1)) / (n_components - 1)
            
            # Normalize rows to ensure each row sums to 1
            for i in range(n_components):
                if np.sum(transmat[i, :]) > 0:  # Avoid division by zero
                    transmat[i, :] = transmat[i, :] / np.sum(transmat[i, :])
                else:
                    # Fallback to uniform distribution if all zeros
                    transmat[i, :] = 1.0 / n_components
                    
            return transmat
        else:
            # Fallback to simple model with warning
            warnings.warn(f"Unknown transition model '{self.hmm_params['transition_model']}'. Using 'simple' model instead.")
            transmat = np.ones((n_components, n_components)) * 0.01 / (n_components - 1)
            np.fill_diagonal(transmat, 0.99)
            return transmat

    def _handle_covariance(self, covars, n_components):
        """
        Handle covariance matrices based on the covariance type.
        
        Args:
            covars (numpy.ndarray): Initial covariance estimates
            n_components (int): Number of states/components
            
        Returns:
            numpy.ndarray: Processed covariance matrices
        """
        covariance_type = self.hmm_params['covariance_type']
        
        if covariance_type == "full":
            # Each state has its own covariance matrix
            return covars
            
        elif covariance_type == "tied":
            # For tied covariance, we need a single (n_dim, n_dim) matrix
            # We'll use the average of all the individual covariances
            n_dim = covars.shape[1]
            tied_covar = np.zeros((n_dim, n_dim))
            for i in range(n_components):
                tied_covar += covars[i]
            tied_covar /= n_components
            return tied_covar
            
        elif covariance_type in ["diag", "spherical"]:
            # These types aren't directly handled here but will be processed by hmmlearn
            warnings.warn(f"Covariance type '{covariance_type}' might require further processing by hmmlearn.")
            return covars
            
        else:
            # Fallback to full covariance with warning
            warnings.warn(f"Unknown covariance type '{covariance_type}'. Using 'full' covariance instead.")
            return covars

    def start_HMM_fit(self, intTime=1, SNRmin=3, targetPower=None, numModes=2, n_jobs=None, 
                     covariance_type=None, n_iter=None, tol=None, verbose=None, transition_model=None):
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
        if transition_model is not None:
            self.hmm_params['transition_model'] = transition_model
            
        print(f"HMM parameters: {self.hmm_params}")
        
        # Determine number of cores to use
        if n_jobs is None or n_jobs == -1:
            self.num_cores = psutil.cpu_count(logical=True)  # Use all logical cores
        else:
            self.num_cores = n_jobs
            
        print(f"Using {self.num_cores} CPU cores for HMM parallelization")

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

    def _process_single_file(self, i, atten, file, means, covars, intTime, SNRmin, skip, savefile, metainfo, hmm_n_jobs=1):
        """Process a single file with HMM analysis - optimized for parallel HMM fitting"""
        print(f"Starting HMM fit for attenuation {atten}...")
        start_time = time.time()
        
        n_comp = self.numModes
        figpath = os.path.join(self.figure_path, f'ATTEN{atten}')
        if not os.path.exists(figpath):
            os.makedirs(figpath)
            
        # Load and process data
        data = qp.loadAlazarData(file)
        data, sr = qp.BoxcarDownsample(data, avgTime=intTime, sampleRate=self.sampleRateFromData, returnRate=True) 
        data = qp.uint16_to_mV(data)
        
        # Process covariance matrices based on the covariance type
        processed_covars = self._handle_covariance(covars, n_comp)
        
        # 1. Plot initial guessed centers and covariance on IQ histogram (before HMM)
        plt.figure(figsize=[6, 6])
        h = qp.plotComplexHist(data[0], data[1], figsize=[6, 6])
        
        # Create a custom function to plot the initial guess ellipses
        def make_ellipses_for_initial_guess(means, covars, ax, colors):
            for i, (mean, covar) in enumerate(zip(means, covars)):
                v, w = np.linalg.eigh(covar)
                v = 2. * np.sqrt(2.) * np.sqrt(v)
                u = w[0] / np.linalg.norm(w[0])
                
                # Plot an ellipse to show the Gaussian component
                angle = np.arctan(u[1] / u[0])
                angle = 180. * angle / np.pi  # Convert to degrees
                ell = plt.matplotlib.patches.Ellipse(mean, v[0], v[1], 180. + angle, 
                                                   color=colors[i])
                ell.set_clip_box(ax.bbox)
                ell.set_alpha(0.5)
                ax.add_artist(ell)
                ax.scatter(mean[0], mean[1], s=100, c=colors[i], marker='x')
                ax.text(mean[0], mean[1], f'State {i}', fontsize=10, 
                       color=colors[i], ha='center', va='bottom')
        
        # Use colormap to create colors for any number of states
        colormap = plt.cm.tab10
        colors = [colormap(j/n_comp) for j in range(n_comp)]
        
        make_ellipses_for_initial_guess(means, covars, plt.gca(), colors)
        plt.xlabel('I [mV]')
        plt.ylabel('Q [mV]')
        plt.title(f'Initial Guess for {n_comp} states | {self.power_to_device[i+skip]} dBm')
        plt.savefig(os.path.join(figpath, f'Initial_Guess_IQ_Histogram_{i+skip}_{self.power_to_device[i+skip]}dBm_{n_comp}modes.png'))
        plt.close()
        
        # Use optimized parallel HMM fitting if supported
        if HMM_SUPPORTS_PARALLEL:
            print(f"Fitting HMM for attenuation {atten} using {hmm_n_jobs} CPU cores...")
            M = hmm.GaussianHMM(n_components=n_comp, 
                            covariance_type=self.hmm_params['covariance_type'],
                            n_iter=self.hmm_params['n_iter'],
                            tol=self.hmm_params['tol'],
                            init_params="",  # No automatic initialization
                            verbose=self.hmm_params['verbose'],
                            n_jobs=hmm_n_jobs)  # Use parameter for HMM-level parallelism
        else:
            print(f"Fitting HMM for attenuation {atten} using single-core mode...")
            M = hmm.GaussianHMM(n_components=n_comp, 
                            covariance_type=self.hmm_params['covariance_type'],
                            n_iter=self.hmm_params['n_iter'],
                            tol=self.hmm_params['tol'],
                            init_params="",  # No automatic initialization
                            verbose=self.hmm_params['verbose'])
                          
        # Manual initialization of all parameters
        M.means_ = means
        
        # Set covariance based on the processed value
        if self.hmm_params['covariance_type'] == 'tied':
            M.covars_ = processed_covars
        else:
            M.covars_ = processed_covars
            
        # Equal starting probabilities
        M.startprob_ = np.ones(n_comp) / n_comp
        
        # Create transition matrix based on the selected model
        M.transmat_ = self._create_transition_matrix(n_comp)

        # Fit the model
        print(f"Fitting HMM for attenuation {atten}...")
        M.fit(data.T)
        print(f"HMM fitting completed for attenuation {atten}")
        
        # Read previous data for comparison
        with h5py.File(savefile, 'r') as ff:
            oldrates = ff[f'ATTEN{self.attens[i+skip-1]}/transitionRatesMHz'][:] if i != 0 else 0.0001*np.ones((n_comp, n_comp))
            lifetimes = np.array([1/oldrates[j,j] for j in range(n_comp)])
            
            # For SNR check, we'll use transitions between ground state (0) and first excited state (1)
            t01 = 1/oldrates[0,1]
            t10 = 1/oldrates[1,0]
            ttimes = np.array([t01, t10])

        # Check SNR - always use the first two states for SNR check
        snr01 = qp.getSNRhmm(M, mode1=0, mode2=1)
        SNRs = np.array([snr01,])

        # Get state estimates
        logprob, Q = M.decode(data.T)
        Qmean = np.mean(Q)
        
        # Calculate state occupations for all states
        state_occupations = {}
        for j in range(n_comp):
            state_occupations[f'P{j}'] = np.sum(Q == j) / Q.size
        
        # For backwards compatibility, maintain P0 and P1 variables
        P0 = state_occupations['P0']
        P1 = state_occupations['P1'] if n_comp > 1 else 0
        
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
                if HMM_SUPPORTS_PARALLEL:
                    M = hmm.GaussianHMM(n_components=n_comp, 
                                      covariance_type=self.hmm_params['covariance_type'],
                                      n_iter=self.hmm_params['n_iter'],
                                      tol=self.hmm_params['tol'],
                                      init_params="",
                                      verbose=self.hmm_params['verbose'],
                                      n_jobs=hmm_n_jobs)
                else:
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
                
                # Create transition matrix based on the selected model
                M.transmat_ = self._create_transition_matrix(n_comp)
                
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
                
                # Update state estimates
                logprob, Q = M.decode(data.T)

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
        
        # Use a colormap to generate colors for any number of states
        colormap = plt.cm.viridis  # viridis is a good colormap that's distinguishable even with many colors
        colors = [colormap(j/n_comp) for j in range(n_comp)]
        qp.make_ellipsesHMM(M, h, colors)
        
        plt.xlabel('I [mV]')
        plt.ylabel('Q [mV]')
        plt.title('HMM fit | {:.2} MHz | {} dBm'.format(sr, self.power_to_device[i+skip]))
        plt.savefig(os.path.join(figpath, 'HMMfits_{}_{}dBm_{}modes.png'.format(i+skip, self.power_to_device[i+skip], self.numModes)))
        plt.close()
        
        # 2. Plot I-Q histogram colored by state after HMM analysis
        plt.figure(figsize=[6, 6])
        unique_states = np.unique(Q)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_states)))
        
        for state_idx, state in enumerate(unique_states):
            mask = Q == state
            plt.scatter(data[0][mask], data[1][mask], s=1, c=[colors[state_idx]], label=f'State {state}')
            
        plt.xlabel('I [mV]')
        plt.ylabel('Q [mV]')
        plt.title('I-Q Data Colored by HMM State | {:.2} MHz | {} dBm'.format(sr, self.power_to_device[i+skip]))
        plt.legend()
        plt.savefig(os.path.join(figpath, 'IQ_by_state_{}_{}dBm_{}modes.png'.format(i+skip, self.power_to_device[i+skip], self.numModes)))
        plt.close()
        
        # 3. Individual and cumulative state IQ plots
        n_states = len(unique_states)
        
        # Individual state plots
        fig, axes = plt.subplots(1, n_states, figsize=(5*n_states, 5), squeeze=False)
        
        for state_idx, state in enumerate(unique_states):
            ax = axes[0, state_idx]
            mask = Q == state
            ax.scatter(data[0][mask], data[1][mask], s=1, c=[colors[state_idx]])
            ax.set_title(f"State {state}")
            ax.set_xlabel("I [mV]")
            ax.set_ylabel("Q [mV]")
            ax.grid(True)
            
            # Add the fitted mean for this state
            ax.scatter(M.means_[state_idx, 0], M.means_[state_idx, 1], color='red', s=100, marker='x')
        
        plt.tight_layout()
        plt.savefig(os.path.join(figpath, 'individual_state_IQ_{}_{}dBm_{}modes.png'.format(i+skip, self.power_to_device[i+skip], self.numModes)))
        plt.close()
        
        # Cumulative state plots
        fig, axes = plt.subplots(1, n_states, figsize=(5*n_states, 5), squeeze=False)
        
        for i_state in range(n_states):
            ax = axes[0, i_state]
            
            # Plot states from 0 to i_state
            for j in range(i_state+1):
                state = unique_states[j]
                mask = Q == state
                ax.scatter(data[0][mask], data[1][mask], s=1, c=[colors[j]], label=f"State {state}")
            
            ax.set_title(f"States 0-{i_state}")
            ax.set_xlabel("I [mV]")
            ax.set_ylabel("Q [mV]")
            ax.grid(True)
            ax.legend()
            
            # Add all relevant means
            for j in range(i_state+1):
                ax.scatter(M.means_[j, 0], M.means_[j, 1], color='red', s=100, marker='x')
        
        plt.tight_layout()
        plt.savefig(os.path.join(figpath, 'cumulative_state_IQ_{}_{}dBm_{}modes.png'.format(i+skip, self.power_to_device[i+skip], self.numModes)))
        plt.close()
        
        # 4. 1D distributions of I and Q for each state
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # I distribution
        for state_idx, state in enumerate(unique_states):
            mask = Q == state
            if np.sum(mask) > 0:  # Only plot if there are points in this state
                ax1.hist(data[0][mask], bins=50, alpha=0.7, color=colors[state_idx], label=f"State {state}")
        
        ax1.set_title(f"I Distribution by State - {self.power_to_device[i+skip]} dBm")
        ax1.set_xlabel("I [mV]")
        ax1.set_ylabel("Count")
        ax1.grid(True)
        ax1.legend()
        
        # Q distribution
        for state_idx, state in enumerate(unique_states):
            mask = Q == state
            if np.sum(mask) > 0:  # Only plot if there are points in this state
                ax2.hist(data[1][mask], bins=50, alpha=0.7, color=colors[state_idx], label=f"State {state}")
        
        ax2.set_title(f"Q Distribution by State - {self.power_to_device[i+skip]} dBm")
        ax2.set_xlabel("Q [mV]")
        ax2.set_ylabel("Count")
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(figpath, 'IQ_1D_distributions_{}_{}dBm_{}modes.png'.format(i+skip, self.power_to_device[i+skip], self.numModes)))
        plt.close()
        
        # Individual I and Q distributions for each state (separate subplots)
        fig, axes = plt.subplots(2, n_states, figsize=(5*n_states, 10), squeeze=False)
        
        # Top row: I distributions for each state
        for state_idx, state in enumerate(unique_states):
            ax = axes[0, state_idx]
            mask = Q == state
            if np.sum(mask) > 0:  # Only plot if there are points in this state
                ax.hist(data[0][mask], bins=50, alpha=0.7, color=colors[state_idx])
                
                # Add a vertical line at the mean
                ax.axvline(x=M.means_[state_idx, 0], color='red', linestyle='--', linewidth=2)
                
                # Add statistics
                mean_val = np.mean(data[0][mask])
                std_val = np.std(data[0][mask])
                ax.text(0.05, 0.95, f"Mean: {mean_val:.3f}\nStd: {std_val:.3f}", 
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            ax.set_title(f"I Distribution - State {state}")
            ax.set_xlabel("I [mV]")
            ax.set_ylabel("Count")
            ax.grid(True)
        
        # Bottom row: Q distributions for each state
        for state_idx, state in enumerate(unique_states):
            ax = axes[1, state_idx]
            mask = Q == state
            if np.sum(mask) > 0:  # Only plot if there are points in this state
                ax.hist(data[1][mask], bins=50, alpha=0.7, color=colors[state_idx])
                
                # Add a vertical line at the mean
                ax.axvline(x=M.means_[state_idx, 1], color='red', linestyle='--', linewidth=2)
                
                # Add statistics
                mean_val = np.mean(data[1][mask])
                std_val = np.std(data[1][mask])
                ax.text(0.05, 0.95, f"Mean: {mean_val:.3f}\nStd: {std_val:.3f}", 
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            ax.set_title(f"Q Distribution - State {state}")
            ax.set_xlabel("Q [mV]")
            ax.set_ylabel("Count")
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(figpath, 'individual_IQ_distributions_{}_{}dBm_{}modes.png'.format(i+skip, self.power_to_device[i+skip], self.numModes)))
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
            
            # Save occupation probabilities for all states
            for j in range(n_comp):
                fp.attrs.create(f'P{j}', state_occupations[f'P{j}'])
                
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
        
        # Save HMM model parameters as .npz file
        npz_path = os.path.join(figpath, f'HMM_params_{i+skip}_{self.power_to_device[i+skip]}dBm_{n_comp}modes.npz')
        np.savez(npz_path, 
                 means=M.means_, 
                 covars=M.covars_,
                 transmat=M.transmat_,
                 startprob=M.startprob_,
                 Q=Q,
                 logprob=logprob,
                 SNRs=SNRs,
                 occupation=state_occupations,
                 rates=rates,
                 power=self.power_to_device[i+skip],
                 attenuation=atten,
                 sampleRate=sr,
                 num_modes=n_comp)

        elapsed = time.time() - start_time
        print(f"Completed HMM fit for attenuation {atten} in {elapsed:.2f} seconds")
        return M

    def runHMM(self, means, covars, intTime=1, SNRmin=3):
        matplotlib.use('Agg')  # Use non-interactive backend for plotting
        
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
        
        # Prepare data for processing
        files_to_process = []
        for i, atten, file in zip(np.arange(len(self.attens[self.index:])), self.attens[self.index:], self.files[self.index:]):
            files_to_process.append((i, atten, file))
        
        print(f"Starting HMM processing for {len(files_to_process)} files...")
        print(f"Using {self.num_cores} CPU cores for HMM-level parallelism")
        
        # Process files sequentially with optimized HMM-level parallelism
        HMM = []
        current_means = means
        current_covars = covars
        
        for i, atten, file in files_to_process:
            print(f"Processing file {i+1}/{len(files_to_process)}: attenuation {atten}")
            result = self._process_single_file(i, atten, file, current_means, current_covars, 
                                            intTime, SNRmin, skip, savefile, self.metainfo,
                                            hmm_n_jobs=self.num_cores)  # Use all cores for each HMM fit
            
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
        
        # Save all HMM models to a single NPZ file
        if len(HMM) > 0:
            hmm_models_data = {
                'num_models': len(HMM),
                'numModes': self.numModes,
                'phi': self.phi,
                'temp': self.temp,
                'attens': self.attens[self.index:self.index+len(HMM)],
                'powers': self.power_to_device[self.index:self.index+len(HMM)]
            }
            
            # Add data for each model
            for i, model in enumerate(HMM):
                hmm_models_data[f'model{i}_means'] = model.means_
                hmm_models_data[f'model{i}_covars'] = model.covars_
                hmm_models_data[f'model{i}_transmat'] = model.transmat_
                hmm_models_data[f'model{i}_startprob'] = model.startprob_
            
            # Save to NPZ file
            np.savez(
                os.path.join(self.project_path, 'AnalyisResults', f'HMM_models_M{self.numModes}_T{self.temp}_PHI{str(self.phi).replace(".","p")[:5]}.npz'),
                **hmm_models_data
            )
        
        print("Starting post-HMM analysis plots.....")
        create_HMM_QP_statistics_plots(self.hdf5_file, self.figure_path, self.numModes)
        print("="*10+"\tHMM ANALYSIS CONCLUDED\t"+"="*10+"\n\n")
