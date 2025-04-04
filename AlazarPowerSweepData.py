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

import datetime
import glob
import json
import multiprocessing
import os
import re
import sys
import time
import warnings

import h5py
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import psutil
from hmmlearn import hmm
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm, trange

import quasiparticleFunctions as qp
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
        # Create a timestamp string for unique file identification
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Add timestamp to figure path
        self.figure_path = os.path.join(project_path, f"PowerSweepfigures")
        self.figure_path = os.path.join(self.figure_path, f"{self.timestamp}")
        # Create the figure path directory immediately
        if not os.path.exists(self.figure_path):
            os.makedirs(self.figure_path)
            print(f"Created figure directory: {self.figure_path}")
            
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
            
            print("Please Ensure the Following is Correct:\n\n")
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

    def get_automatic_QP_means(self, avgTime=3, sort_states=True):
        """
        Automatically estimate initial HMM state means using K-means clustering.
        
        This method replaces manual selection of means with automated clustering,
        which is faster and more reproducible than manual selection.
        
        The states are ordered clockwise in the I-Q plane, with:
        - State 0 typically having lower I values
        - Subsequent states increasing in order from left to right on the I axis
        
        Args:
            avgTime (float): Time in microseconds to average data for downsampling
            sort_states (bool): Whether to sort states by I-coordinate and clockwise order
            
        Returns:
            tuple: (means, data) where means is array of state centers and data is the IQ data
        """
        try:
            print("Automatically determining initial means using K-means clustering...")
            
            # Check if sklearn is available for K-means clustering
            try:
                from sklearn.cluster import KMeans
            except ImportError:
                print("Warning: sklearn not installed. Falling back to manual means selection.")
                return self.get_QP_means_from_IQ(avgTime)
                
            # Make sure we have a figure path
            try:
                if not os.path.exists(self.figure_path):
                    os.makedirs(self.figure_path)
                    print(f"Created figure directory: {self.figure_path}")
            except Exception as e:
                print(f"Warning: Could not create figure directory: {str(e)}")
            
            # Load and process the data
            try:
                data = qp.loadAlazarData(self.files[self.index])
                data, sr = qp.BoxcarDownsample(data, avgTime, sampleRate=self.sampleRateFromData, returnRate=True)
                data = qp.uint16_to_mV(data)
            except Exception as e:
                print(f"Error loading or processing data: {str(e)}")
                print("Falling back to manual means selection.")
                return self.get_QP_means_from_IQ(avgTime)
            
            # Prepare data for K-means (reshape to [n_samples, n_features])
            iq_data = np.vstack([data[0], data[1]]).T
            
            # Run K-means clustering
            kmeans = KMeans(
                n_clusters=self.numModes,
                init='k-means++',  # Smart initialization for faster convergence
                n_init=10,         # Run multiple initializations and pick best
                max_iter=300,      # Maximum iterations for each initialization
                tol=1e-4,          # Convergence tolerance
                random_state=42    # For reproducibility
            )
            
            # Fit K-means model
            kmeans.fit(iq_data)
            
            # Get cluster centers as initial means
            means = kmeans.cluster_centers_
            
            # Count points in each cluster for reference
            labels = kmeans.labels_
            cluster_populations = np.zeros(self.numModes, dtype=int)
            for i in range(self.numModes):
                cluster_populations[i] = np.sum(labels == i)
            
            # Save original unsorted means for verification
            original_means = means.copy()
            original_labels = labels.copy()
            
            if sort_states:
                try:
                    # Sort states based on I-coordinate (left to right)
                    # This ensures that states are ordered clockwise around the I-Q plane
                    i_sorted_indices = np.argsort(means[:, 0])
                    means = means[i_sorted_indices]
                    
                    # Create a mapping from old cluster indices to new sorted indices
                    idx_map = {i_sorted_indices[i]: i for i in range(len(i_sorted_indices))}
                    
                    # Remap the cluster labels based on the new ordering
                    remapped_labels = np.zeros_like(labels)
                    for i in range(len(labels)):
                        remapped_labels[i] = idx_map[labels[i]]
                    
                    # Recalculate populations after remapping
                    new_populations = np.zeros(self.numModes, dtype=int)
                    for i in range(self.numModes):
                        new_populations[i] = np.sum(remapped_labels == i)
                        
                    print(f"Sorted means by I-coordinate (left to right, clockwise in I-Q plane)")
                    
                    # Create a verification plot showing before and after sorting
                    plt.close('all')  # Close any existing plots to avoid confusion
                    
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    
                    # Before sorting plot
                    ax1 = axes[0]
                    
                    # Calculate the center for reference
                    center = np.mean(original_means, axis=0)
                    
                    # Create 2D histogram
                    h1 = ax1.hist2d(data[0], data[1], bins=80, norm=matplotlib.colors.LogNorm(), 
                                   cmap=plt.cm.Greys)
                    plt.colorbar(h1[3], ax=ax1, shrink=0.9, extend='both')
                    
                    # Plot center and means
                    ax1.scatter(center[0], center[1], c='blue', s=150, marker='+', label='Center')
                    for i, mean in enumerate(original_means):
                        pop = cluster_populations[i]
                        ax1.scatter(mean[0], mean[1], c=f'C{i}', s=150, marker='x')
                        ax1.text(mean[0], mean[1], f'S{i}\n({pop})', fontsize=10, 
                                color=f'C{i}', ha='center', va='bottom', fontweight='bold')
                    
                    ax1.set_title('Before Sorting', fontsize=14)
                    ax1.set_xlabel('I [mV]', fontsize=12)
                    ax1.set_ylabel('Q [mV]', fontsize=12)
                    ax1.grid(True, alpha=0.3)
                    ax1.set_aspect('equal')
                    
                    # After sorting plot
                    ax2 = axes[1]
                    
                    # Create 2D histogram
                    h2 = ax2.hist2d(data[0], data[1], bins=80, norm=matplotlib.colors.LogNorm(), 
                                   cmap=plt.cm.Greys)
                    plt.colorbar(h2[3], ax=ax2, shrink=0.9, extend='both')
                    
                    # Plot center and sorted means
                    ax2.scatter(center[0], center[1], c='blue', s=150, marker='+', label='Center')
                    for i, mean in enumerate(means):
                        pop = new_populations[i]
                        ax2.scatter(mean[0], mean[1], c=f'C{i}', s=150, marker='x')
                        ax2.text(mean[0], mean[1], f'S{i}\n({pop})', fontsize=10, 
                                color=f'C{i}', ha='center', va='bottom', fontweight='bold')
                        
                        # Draw a line from center to each point to show the clockwise ordering
                        ax2.plot([center[0], mean[0]], [center[1], mean[1]], 'k--', alpha=0.5)
                        
                        # Draw arrows connecting points in order
                        if i < len(means) - 1:
                            dx = means[i+1, 0] - mean[0]
                            dy = means[i+1, 1] - mean[1]
                            ax2.arrow(mean[0], mean[1], dx*0.7, dy*0.7, 
                                     head_width=0.05, head_length=0.1, fc='green', ec='green', alpha=0.7)
                    
                    ax2.set_title('After Sorting (Clockwise by I-coordinate)', fontsize=14)
                    ax2.set_xlabel('I [mV]', fontsize=12)
                    ax2.set_ylabel('Q [mV]', fontsize=12)
                    ax2.grid(True, alpha=0.3)
                    ax2.set_aspect('equal')
                    
                    # Adjust subplot spacing
                    plt.subplots_adjust(wspace=0.3)
                    
                    # Save verification plot
                    try:
                        verification_path = os.path.join(self.figure_path, 
                                                  f'KMeans_Sorting_Verification_{self.numModes}modes_{self.timestamp}.png')
                        fig.savefig(verification_path, bbox_inches='tight', dpi=150)
                        print(f"Saved state sorting verification plot to: {verification_path}")
                    except Exception as e:
                        print(f"Failed to save verification plot: {str(e)}")
                    finally:
                        plt.close(fig)  # Ensure figure is closed
                        
                except Exception as e:
                    print(f"Error in state sorting: {str(e)}")
                    print("Using unsorted K-means states.")
                    plt.close('all')
            else:
                print("Skipping state sorting as requested.")
            
            # Create a final visualization of the sorted states
            try:
                plt.close('all')  # Close any existing plots
                
                fig, ax = plt.subplots(figsize=(8, 8))
                
                # Create 2D histogram
                h = ax.hist2d(data[0], data[1], bins=80, norm=matplotlib.colors.LogNorm(), 
                             cmap=plt.cm.Greys)
                plt.colorbar(h[3], ax=ax, shrink=0.9, extend='both')
                
                # Find the center for plotting
                center = np.mean(means, axis=0)
                
                # Plot the center and means
                ax.scatter(center[0], center[1], c='blue', s=150, marker='+', label='Center')
                
                # Plot cluster centers with arrows showing trajectory
                for i in range(len(means)):
                    ax.scatter(means[i, 0], means[i, 1], c=f'C{i}', s=150, marker='x')
                    ax.text(means[i, 0], means[i, 1], f'State {i}', fontsize=14, 
                           color=f'C{i}', ha='center', va='bottom', fontweight='bold')
                    
                    # Draw a line from center to each point
                    ax.plot([center[0], means[i, 0]], [center[1], means[i, 1]], 'k--', alpha=0.5)
                    
                    # Draw an arrow from current state to next state (if not the last state)
                    if i < len(means) - 1:
                        ax.arrow(means[i, 0], means[i, 1], 
                                (means[i+1, 0] - means[i, 0])*0.7, (means[i+1, 1] - means[i, 1])*0.7,
                                head_width=0.05, head_length=0.1, fc='green', ec='green', alpha=0.7)
                
                # Draw an arrow from the last state back to the first to complete the circle
                if len(means) > 2:
                    ax.arrow(means[-1, 0], means[-1, 1], 
                           (means[0, 0] - means[-1, 0])*0.7, (means[0, 1] - means[-1, 1])*0.7,
                           head_width=0.05, head_length=0.1, fc='red', ec='red', alpha=0.7)
                
                ax.set_title(f'K-means Clustering: {self.numModes} States\nStates sorted left-to-right (clockwise in I-Q)', fontsize=14)
                ax.set_xlabel('I [mV]', fontsize=12)
                ax.set_ylabel('Q [mV]', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
                ax.legend()
                
                # Save the figure
                try:
                    fig_path = os.path.join(self.figure_path, f'KMeans_Initial_Means_{self.numModes}modes_{self.timestamp}.png')
                    fig.savefig(fig_path, bbox_inches='tight', dpi=150)
                    print(f"Saved K-means clustering figure to: {fig_path}")
                except Exception as e:
                    print(f"Failed to save K-means clustering figure: {str(e)}")
                finally:
                    plt.close(fig)  # Ensure figure is closed
                    
            except Exception as e:
                print(f"Error creating final visualization: {str(e)}")
                plt.close('all')
                
            print(f"K-means clustering complete. Found {self.numModes} cluster centers.")
            return means, data
            
        except Exception as e:
            print(f"Unhandled error in K-means clustering: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Falling back to manual means selection.")
            plt.close('all')
            return self.get_QP_means_from_IQ(avgTime)

    def get_initial_QP_means(self, avgTime=3):
        if self.interactive:
            return self.get_QP_means_from_IQ(avgTime)
        else:
            try:
                return get_QP_means(self.project_root, self.phi, self.numModes), None
            except:
                # Use automatic means detection instead of manual selection in non-interactive mode
                return self.get_automatic_QP_means(avgTime)

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
        """
        Process Alazar data files and optionally create IQ plots.
        
        Args:
            avgTime (float): Time in microseconds to average data for downsampling
            plots (bool): Whether to create IQ plots of the data
        """
        print("Setting up directories and paths...")
        
        # Ensure figure path exists
        if not os.path.exists(self.figure_path):
            os.makedirs(self.figure_path)
            print(f"Created figure directory: {self.figure_path}")
            
        print("Reading and sorting data files...")
        self.files, self.attens = sort_files_ascending_attenuation(self.files)
        convert_to_json(self.files)

        set_plot_style()
        print("Reading and updating the metadata...")

        update_metainfo(self.files[0])

        self.sampleRateFromData = get_sample_rate_from_run(self.files[0])
        self.phi = get_phi_from_run(self.files[0])
        self.temp = get_temp_from_run(self.files[0])

        if plots:
            print(f"Creating IQ downsampled plots with timestamp {self.timestamp}...")            
            # Create a timestamped directory for IQ plots
            iq_plot_dir = os.path.join(self.project_path, f"IQ_Plots_{self.timestamp}")
            if not os.path.exists(iq_plot_dir):
                os.makedirs(iq_plot_dir)
                print(f"Created IQ plots directory: {iq_plot_dir}")
                
            try:
                create_IQ_downsampled_plots(self.files, self.attens, iq_plot_dir, avgTime)
                print(f"Successfully created IQ plots in {iq_plot_dir}")
            except Exception as e:
                print(f"Warning: Failed to create IQ plots: {str(e)}")
        else:
            print("Skipping IQ downsampled plots as requested.")

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
            
        elif covariance_type == "diag":
            # For diagonal covariance, extract the diagonals from each covariance matrix
            n_dim = covars.shape[1]
            diag_covars = np.zeros((n_components, n_dim))
            for i in range(n_components):
                diag_covars[i] = np.diag(covars[i])
            return diag_covars
            
        elif covariance_type == "spherical":
            # For spherical covariance, use the average of diagonal elements
            n_dim = covars.shape[1]
            spherical_covars = np.zeros(n_components)
            for i in range(n_components):
                spherical_covars[i] = np.mean(np.diag(covars[i]))
            return spherical_covars
            
        else:
            # Fallback to full covariance with warning
            warnings.warn(f"Unknown covariance type '{covariance_type}'. Using 'full' covariance instead.")
            return covars

    def _ensure_positive_definite(self, covariance, min_eigenval=1e-6):
        """
        Ensure covariance matrix is positive definite by adding small values to diagonal if needed.
        
        Args:
            covariance: The covariance matrix to check/fix
            min_eigenval: Minimum eigenvalue to ensure positive definiteness
            
        Returns:
            Modified covariance matrix that is positive definite
        """
        eigenvals, eigenvecs = np.linalg.eigh(covariance)
        if np.min(eigenvals) < min_eigenval:
            # Add small positive value to diagonal to ensure positive definiteness
            delta = min_eigenval - np.min(eigenvals)
            covariance += np.eye(covariance.shape[0]) * (delta + 1e-6)
        return covariance

    def start_HMM_fit(self, intTime=1, SNRmin=3, targetPower=None, numModes=2, n_jobs=None, 
                     covariance_type=None, n_iter=None, tol=None, verbose=None, transition_model=None,
                     fast_mode=False, auto_means=False, sort_states=True):
        """
        Start the HMM fitting process with optimized performance options.
        
        Args:
            intTime (float): Integration time in microseconds
            SNRmin (float): Minimum SNR threshold
            targetPower (float): Target power to device in dB
            numModes (int): Number of HMM states to fit
            n_jobs (int): Number of CPU cores to use (-1 for all cores)
            covariance_type (str): Covariance type ('full', 'diag', 'spherical', 'tied')
            n_iter (int): Maximum number of EM iterations
            tol (float): Convergence tolerance
            verbose (bool): Whether to print progress during fitting
            transition_model (str): Model for transition matrix ('simple' or 'physics')
            fast_mode (bool): Whether to use optimized parameters for faster fitting
            auto_means (bool): Whether to use automatic mean estimation with K-means
            sort_states (bool): Whether to sort states by proximity to origin and clockwise order
        """
        try:
            print("\n\n"+"="*10+"\tHMM ANALYSIS STARTED\t"+"="*10)
            
            # Make sure figure directory exists
            if not os.path.exists(self.figure_path):
                os.makedirs(self.figure_path)
                print(f"Created figure directory: {self.figure_path}")
            
            # Apply fast mode settings if requested
            if fast_mode:
                print("Fast mode enabled. Using optimized parameters for speed.")
                if covariance_type is None:
                    covariance_type = 'diag'  # Diagonal covariance is faster than full
                if n_iter is None:
                    n_iter = 50  # Fewer iterations for faster convergence
                if tol is None:
                    tol = 1e-2  # Higher tolerance for earlier stopping
                if transition_model is None:
                    transition_model = 'simple'  # Simpler transition model is faster
                if verbose is None:
                    verbose = False  # Less output for faster processing
            
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
            
            # Check if any data files were found
            if len(self.files) == 0:
                raise FileNotFoundError(f"No .bin files found in {self.project_path}. Please check the path.")

            if self.interactive:
                try:
                    chosenAtten = int(input("\nAttenuation below which the system goes non-linear: "))
                    # self.power_to_device = self.set_attenuation_configuration()
                    self.power_to_device = float(input("\nInput the Power to the device (in dB): "))

                    self.metainfo = self.set_metadata()

                    self.index = int(np.where(self.attens == chosenAtten)[0])
                    print("\nThe power to the device is {} dBM at the chosen attenuation {}".format(self.power_to_device - chosenAtten, chosenAtten))

                    self.numModes = int(input("\nNumber of Modes you want to fit: "))
                    set_qt_backend()

                    # Choose automatic or manual means selection
                    if auto_means:
                        means, data = self.get_automatic_QP_means(sort_states=sort_states)
                    else:
                        means, data = self.get_initial_QP_means()
                        
                    covars = self.get_initial_QP_covars(data, means)
                    print(f"Extracted Means:\n{means}\n\nEstimated Covariance:\n{covars}\n")
                    print("\nStarting HMM Analysis.....\n\n")
                    self.runHMM(means, covars, intTime, SNRmin)
                except ValueError as e:
                    print(f"Error during interactive setup: {str(e)}")
                    print("Please check your input values and try again.")
                    raise
            else:
                try:
                    # self.power_to_device = self.set_attenuation_configuration()
                    # Ask user to input the power to the device
                    self.power_to_device = float(input("\nInput the Power to the device (in dB): "))
                    self.metainfo = self.set_metadata()
                    try:
                        if targetPower is None:
                            self.index = 0
                            print("No target power specified. Using the first file.")
                        else:
                            # Try to find the exact match first
                            match_idx = np.where(self.power_to_device == targetPower)[0]
                            if len(match_idx) > 0:
                                self.index = int(match_idx[0])
                            else:
                                # If no exact match, find closest
                                print(f"No exact match for {targetPower} dB. Finding closest value...")
                                self.index = np.argmin(np.abs(self.power_to_device - targetPower))
                    except Exception as e:
                        print(f"Error finding power match: {str(e)}. Using index 0.")
                        self.index = 0
                        
                    if self.index >= len(self.attens):
                        print(f"Warning: Index {self.index} is out of range for attenuations array. Using index 0.")
                        self.index = 0
                        
                    chosenAtten = self.attens[self.index]
                    print("\nThe chosen power to the device is {} dBM at the attenuation {}".format(self.power_to_device - chosenAtten, chosenAtten))

                    self.numModes = numModes
                    print(f"\nNumber of Modes to be fit: {self.numModes}")
                    set_qt_backend()

                    # Always use automatic means in non-interactive mode when fast_mode is enabled
                    if auto_means or fast_mode:
                        means, data = self.get_automatic_QP_means(sort_states=sort_states)
                    else:
                        means, data = self.get_initial_QP_means()
                        
                    covars = self.get_initial_QP_covars(data, means)
                    print(f"Extracted Means:\n{means}\n\nEstimated Covariance:\n{covars}\n")
                    print("\nStarting HMM Analysis.....\n\n")
                    self.runHMM(means, covars, intTime, SNRmin)
                except Exception as e:
                    print(f"Error during non-interactive setup: {str(e)}")
                    raise

            print(f"Analysis completed with timestamp: {self.timestamp}")
            
        except Exception as e:
            print(f"Error in HMM analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            print("\nTry using the following options for better compatibility:")
            print("1. Set auto_means=True to use K-means clustering instead of manual mean selection")
            print("2. Use fast_mode=True for more reliable performance settings")
            print("3. Check that your project_path contains valid .bin files")
            raise

    def _process_single_file(self, i, atten, file, means, covars, intTime, SNRmin, skip, savefile, metainfo, hmm_n_jobs=1):
        """Process a single file with HMM analysis - optimized for parallel HMM fitting"""
        print(f"Starting HMM fit for attenuation {atten}...")
        start_time = time.time()
        
        try:
            n_comp = self.numModes
            # Create attenuation subfolder with timestamp
            figpath = os.path.join(self.figure_path, f'ATTEN{atten}')
            if not os.path.exists(figpath):
                os.makedirs(figpath)
                
            # Load and process data
            try:
                data = qp.loadAlazarData(file)
                data, sr = qp.BoxcarDownsample(data, avgTime=intTime, sampleRate=self.sampleRateFromData, returnRate=True) 
                data = qp.uint16_to_mV(data)
                n_dim = data.shape[0]
            except Exception as e:
                print(f"Error loading or processing data: {str(e)}")
                return None
            
            # Prepare data for HMM (transpose to [n_samples, n_features])
            hmm_data = data.T
            
            # Process covariance matrices based on the covariance type
            processed_covars = self._handle_covariance(covars, n_comp)
            
            # 1. Plot initial guessed centers and covariance on IQ histogram (before HMM)
            try:
                fig = plt.figure(figsize=(6, 6))
                h = qp.plotComplexHist(data[0], data[1], figsize=(6, 6))
                
                # Create a custom function to plot the initial guess ellipses
                def make_ellipses_for_initial_guess(means, covars, colors):
                    for i, (mean, covar) in enumerate(zip(means, covars)):
                        v, w = np.linalg.eigh(covar)
                        v = 2. * np.sqrt(2.) * np.sqrt(v)
                        u = w[0] / np.linalg.norm(w[0])
                        
                        # Plot an ellipse to show the Gaussian component
                        angle = np.arctan(u[1] / u[0])
                        angle = 180. * angle / np.pi  # Convert to degrees
                        ell = plt.matplotlib.patches.Ellipse(mean, v[0], v[1], 180. + angle, 
                                                           color=colors[i])
                        ell.set_clip_box(plt.gca().bbox)
                        ell.set_alpha(0.5)
                        plt.gca().add_artist(ell)
                        plt.scatter(mean[0], mean[1], s=100, c=colors[i], marker='x')
                        plt.text(mean[0], mean[1], f'State {i}', fontsize=10, 
                               color=colors[i], ha='center', va='bottom')
                
                # Use colormap to create colors for any number of states
                colormap = plt.cm.tab10
                colors = [colormap(j/n_comp) for j in range(n_comp)]
                
                make_ellipses_for_initial_guess(means, covars, colors)
                plt.xlabel('I [mV]')
                plt.ylabel('Q [mV]')
                plt.title(f'Initial Guess for {n_comp} states | {self.power_to_device - self.attens[i+skip]} dBm')
                
                # Avoid tight_layout - use subplots_adjust instead
                plt.subplots_adjust(right=0.85, top=0.9, bottom=0.1, left=0.1)
                
                # Add timestamp to file name
                initial_guess_path = os.path.join(figpath, f'Initial_Guess_IQ_Histogram_{i+skip}_{self.power_to_device - self.attens[i+skip]}dBm_{n_comp}modes_{self.timestamp}.png')
                fig.savefig(initial_guess_path, bbox_inches='tight', dpi=150)
                plt.close(fig)
                print(f"Saved initial guess plot to: {initial_guess_path}")
            except Exception as e:
                print(f"Warning: Error creating initial guess plot: {str(e)}")
                plt.close('all')  # Close any open figures
            
            # Use optimized parallel HMM fitting if supported
            try:
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
                    # For tied covariance, we need a single (n_dim, n_dim) matrix
                    # We'll use the average of all the individual covariances
                    tied_covar = np.zeros((n_dim, n_dim))
                    for i in range(n_comp):
                        tied_covar += processed_covars[i]
                    tied_covar /= n_comp
                    M.covars_ = tied_covar
                elif self.hmm_params['covariance_type'] in ["full", "diag", "spherical"]:
                    M.covars_ = processed_covars
                    
                # Equal starting probabilities
                M.startprob_ = np.ones(n_comp) / n_comp
                
                # Create transition matrix based on the selected model
                M.transmat_ = self._create_transition_matrix(n_comp)

                # Fit the model
                print(f"Fitting HMM for attenuation {atten}...")
                fit_start_time = time.time()
                try:
                    M.fit(hmm_data)
                    fit_success = True
                except Exception as e:
                    print(f"Error in HMM fitting: {str(e)}")
                    # Try to recover by adding regularization to covariances
                    try:
                        if self.hmm_params['covariance_type'] == 'full':
                            for i in range(n_comp):
                                M.covars_[i] = self._ensure_positive_definite(M.covars_[i], min_eigenval=1e-4)
                        M.fit(hmm_data)
                        fit_success = True
                        print("Successfully recovered from fitting error with regularization")
                    except Exception as e2:
                        print(f"Could not recover from fitting error: {str(e2)}")
                        fit_success = False
                        return None
                
                fit_duration = time.time() - fit_start_time
                print(f"The HMM fitting has been completed for attenuation {atten} in {fit_duration:.2f} seconds")
            except Exception as e:
                print(f"Error fitting HMM for attenuation {atten}: {str(e)}")
                return None
            
            # Read previous data for comparison
            try:
                with h5py.File(savefile, 'r') as ff:
                    oldrates = ff[f'ATTEN{self.attens[i+skip-1]}/transitionRatesMHz'][:] if i != 0 else 0.0001*np.ones((n_comp, n_comp))
                    lifetimes = np.array([1/oldrates[j,j] for j in range(n_comp)])
                    
                    # For SNR check, we'll use transitions between ground state (0) and first excited state (1)
                    t01 = 1/oldrates[0,1]
                    t10 = 1/oldrates[1,0]
                    ttimes = np.array([t01, t10])
            except Exception as e:
                print(f"Warning: Error reading previous data: {str(e)}")
                # Use reasonable defaults if can't read previous data
                oldrates = 0.0001*np.ones((n_comp, n_comp))
                lifetimes = np.array([1000.0 for j in range(n_comp)])
                t01 = t10 = 1000.0
                ttimes = np.array([t01, t10])

            # Check SNR - always use the first two states for SNR check
            try:
                snr01 = qp.getSNRhmm(M, mode1=0, mode2=1)
                print(f"SNR01: {snr01}")
                SNRs = np.array([snr01,])
            except Exception as e:
                print(f"Warning: Error calculating SNR: {str(e)}")
                snr01 = 0
                SNRs = np.array([snr01,])

            # Get state estimates
            try:
                logprob, Q = M.decode(hmm_data)
                Qmean = np.mean(Q)
                
                # Calculate state occupations for all states
                state_occupations = {}
                for j in range(n_comp):
                    state_occupations[f'P{j}'] = np.sum(Q == j) / Q.size
                
                # For backwards compatibility, maintain P0 and P1 variables
                P0 = state_occupations['P0']
                P1 = state_occupations['P1'] if n_comp > 1 else 0
            except Exception as e:
                print(f"Error decoding HMM states: {str(e)}")
                return None
            
            # Increase integration time if needed
            current_intTime = intTime
            if np.min(SNRs) < SNRmin and current_intTime <= np.min(lifetimes)/2 and current_intTime <= np.min(ttimes) and P1 < P0:
                try:
                    success = False
                    srold = np.copy(sr)
                    max_attempts = 5  # Limit number of attempts to avoid infinite loops
                    attempt = 0
                    
                    while not success and attempt < max_attempts:
                        attempt += 1
                        current_intTime *= 1.15*SNRmin/np.min(SNRs)
                        print(f'\n\n\nNew integration time = {current_intTime} because SNR was {np.min(SNRs):.6}\n\n\n')
                        
                        try:
                            data = qp.loadAlazarData(file)
                            srold = np.copy(sr)
                            data, sr = qp.BoxcarDownsample(data, current_intTime, self.sampleRateFromData, returnRate=True)
                            data = qp.uint16_to_mV(data)
                        except Exception as e:
                            print(f"Error reloading data with new integration time: {str(e)}")
                            break
                        
                        # Create new model with increased integration time
                        try:
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
                        except Exception as e:
                            print(f"Error creating new HMM with increased integration time: {str(e)}")
                            break
                        
                        # Get parameters from previous run if available
                        try:
                            with h5py.File(savefile, 'r') as ff:
                                M.means_ = ff[f'ATTEN{self.attens[i+skip-1]}'].attrs.get('HMMmeans_') if i != 0 else means
                                M.covars_ = ff[f'ATTEN{self.attens[i+skip-1]}'].attrs.get('HMMcovars_') if i != 0 else covars
                        except Exception as e:
                            print(f"Warning: Error getting parameters from previous run: {str(e)}")
                            M.means_ = means
                            M.covars_ = covars
                        
                        # Initialize other parameters
                        M.startprob_ = np.ones(n_comp) / n_comp
                        
                        # Create transition matrix based on the selected model
                        M.transmat_ = self._create_transition_matrix(n_comp)
                        
                        try:
                            M.fit(data.T)
                        except Exception as e:
                            print(f"Error fitting HMM with increased integration time: {str(e)}")
                            break
                        
                        # Get SNR
                        try:
                            snr01 = qp.getSNRhmm(M, mode1=0, mode2=1)
                            SNRs = np.array([snr01,])
                        except Exception as e:
                            print(f"Warning: Error calculating SNR with increased integration time: {str(e)}")
                            break
                            
                        if np.min(SNRs) > SNRmin:
                            success = True
                        elif current_intTime > np.min(lifetimes)/2 or current_intTime > np.min(ttimes):
                            print('Integration time has grown too large.')
                            break
                        if sr == srold:
                            print('Stuck in loop, exiting')
                            break
                        srold = np.copy(sr)
                        
                        # Update state estimates
                        try:
                            logprob, Q = M.decode(data.T)
                        except Exception as e:
                            print(f"Error decoding HMM states with increased integration time: {str(e)}")
                            break
                except Exception as e:
                    print(f"Error during integration time adjustment: {str(e)}")

            # Check conditions
            if current_intTime > np.min(lifetimes)/2 and i > 3:
                print('Integration time has grown too large.')
                return None
            if P1 > P0:
                print('P1 became larger than P0!')
                return None

            # Plot the fit (offline plotting with Agg backend)
            try:
                # Create figure without constrained_layout
                fig = plt.figure(figsize=(4, 4), constrained_layout=False)
                ax = plt.axes()
                
                # Create 2D histogram manually
                hist_data = plt.hist2d(data[0], data[1], bins=80, 
                                   norm=matplotlib.colors.LogNorm(), 
                                   cmap=plt.cm.Greys)
                plt.colorbar(hist_data[3], shrink=0.9, extend='both')
                plt.grid(True)
                plt.gca().set_aspect('equal')
                
                # Use a colormap to generate colors for any number of states
                colormap = plt.cm.viridis  # viridis is a good colormap that's distinguishable even with many colors
                colors = [colormap(j/n_comp) for j in range(n_comp)]
                
                # Create ellipses manually similar to make_ellipsesHMM
                for n, color in enumerate(colors):
                    try:
                        # get the covariance matrix for this state
                        covariances = M.covars_[n][:2,:2]
                        # get eigenvalues and eigenvectors
                        v, w = np.linalg.eigh(covariances)
                        # normalize the eigenvector
                        u = w[0] / np.linalg.norm(w[0])
                        # get the angle
                        angle = 180*np.arctan2(u[1],u[0])/np.pi
                        # scale the eigenvalues for the ellipse
                        v = 2. * np.sqrt(v)
                        # create the ellipse
                        ell = matplotlib.patches.Ellipse(M.means_[n,:2], v[0], v[1], 180+angle, 
                                                      color=color, fill=False)
                        ell.set_clip_box(plt.gca().bbox)
                        ell.set_alpha(0.8)
                        plt.gca().add_artist(ell)
                    except Exception as e:
                        print(f"Warning: Error creating ellipse for state {n}: {str(e)}")
                        continue
                
                plt.xlabel('I [mV]')
                plt.ylabel('Q [mV]')
                plt.title('HMM fit | {:.2} MHz | {} dBm'.format(sr, self.power_to_device - self.attens[i+skip]))
                
                # Use subplots_adjust safely
                plt.subplots_adjust(right=0.85, top=0.9, bottom=0.1, left=0.1)
                
                hmm_fit_path = os.path.join(figpath, f'HMMfits_{i+skip}_{self.power_to_device - self.attens[i+skip]}dBm_{n_comp}modes_{self.timestamp}.png')
                fig.savefig(hmm_fit_path, bbox_inches='tight', dpi=150)
                plt.close(fig)
                print(f"Saved HMM fit plot to: {hmm_fit_path}")
            except Exception as e:
                print(f"Warning: Error creating HMM fit plot: {str(e)}")
                plt.close('all')  # Close any open figures
            
            # 2. Plot I-Q histogram colored by state after HMM analysis
            try:
                # Create figure without constrained_layout
                fig = plt.figure(figsize=(6, 6), constrained_layout=False)
                ax = plt.axes()
                
                # Get unique states and create color map
                unique_states = np.unique(Q)
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_states)))
                
                # Plot each state with a different color
                for state_idx, state in enumerate(unique_states):
                    mask = Q == state
                    plt.scatter(data[0][mask], data[1][mask], s=1, color=colors[state_idx], 
                               label=f'State {state}')
                    
                plt.xlabel('I [mV]')
                plt.ylabel('Q [mV]')
                plt.title('I-Q Data Colored by HMM State | {:.2} MHz | {} dBm'.format(sr, self.power_to_device - self.attens[i+skip]))
                plt.grid(True)
                plt.gca().set_aspect('equal')
                plt.legend()
                
                # Use subplots_adjust safely
                plt.subplots_adjust(right=0.85, top=0.9, bottom=0.1, left=0.1)
                
                iq_by_state_path = os.path.join(figpath, f'IQ_by_state_{i+skip}_{self.power_to_device - self.attens[i+skip]}dBm_{n_comp}modes_{self.timestamp}.png')
                fig.savefig(iq_by_state_path, bbox_inches='tight', dpi=150)
                plt.close(fig)
                print(f"Saved I-Q by state plot to: {iq_by_state_path}")
            except Exception as e:
                print(f"Warning: Error creating I-Q by state plot: {str(e)}")
                plt.close('all')  # Close any open figures
            
            # Calculate transition rates BEFORE saving to HDF5
            try:
                rates = qp.getTransRatesFromProb(sr, M.transmat_)
                print(f"Calculated transition rates for attenuation {atten}")
            except Exception as e:
                print(f"Error calculating transition rates: {str(e)}")
                rates = np.ones((n_comp, n_comp)) * 0.001  # Default rates if calculation fails
            
            # Now save results to HDF5
            try:
                with h5py.File(savefile, 'a') as ff:
                    fp = ff.require_group(f'ATTEN{atten}')
                    fp.create_dataset('Q', data=Q)
                    fp.create_dataset('data', data=data)
                    fp.create_dataset('transitionRatesMHz', data=rates)  # Now rates is defined
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
                    fp.attrs.create('LOpower', self.power_to_device - self.attens[i+skip])
                    fp.attrs.create('DAsetting', self.attens[i+skip])

                    for key in metainfo:
                        fp.attrs.create(key, metainfo[key])
            except Exception as e:
                print(f"Error saving results to HDF5: {str(e)}")
            
            # Save HMM model parameters as .npz file
            try:
                npz_path = os.path.join(figpath, f'HMM_params_{i+skip}_{self.power_to_device - self.attens[i+skip]}dBm_{n_comp}modes_{self.timestamp}.npz')
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
                         power=self.power_to_device - self.attens[i+skip],
                         attenuation=atten,
                         sampleRate=sr,
                         num_modes=n_comp,
                         timestamp=self.timestamp)
                print(f"Saved HMM parameters to: {npz_path}")
            except Exception as e:
                print(f"Error saving HMM parameters to NPZ file: {str(e)}")

            elapsed = time.time() - start_time
            print(f"Completed HMM fit for attenuation {atten} in {elapsed:.2f} seconds")
            return M
            
        except Exception as e:
            print(f"Unhandled error in _process_single_file for attenuation {atten}: {str(e)}")
            import traceback
            traceback.print_exc()
            plt.close('all')  # Make sure to close all figures
            return None

    def runHMM(self, means, covars, intTime=1, SNRmin=3):
        matplotlib.use('Agg')  # Use non-interactive backend for plotting
        
        skip = np.copy(self.index)
        n_comp = self.numModes
        
        # Create PDF files with timestamp
        hmm_fits_pdf = PdfPages('{}/HMM_IQ_fits_{}modes_{}.pdf'.format(self.project_path, self.numModes, self.timestamp))
        hmm_time_series_pdf = PdfPages('{}/HMM_time_series_{}modes_{}.pdf'.format(self.project_path, self.numModes, self.timestamp))
        
        # Set up HDF5 file with timestamp
        analysis_dir = os.path.join(self.project_path, 'AnalysisResults')
        analysis_dir = os.path.join(analysis_dir, f"{self.timestamp}")
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)
            
        savefile = os.path.join(analysis_dir, 'FullDataset_M{}_T{}_PHI{}_{}.hdf5'.format(
            self.numModes, self.temp, str(self.phi).replace(".", "p")[:5], self.timestamp))
        self.hdf5_file = savefile
        
        # Set up initial HDF5 structure
        with h5py.File(savefile, 'a') as f:
            for atten in self.attens[self.index:]:
                g = f.require_group(f'ATTEN{atten}')
                for key in self.metainfo:
                    g.attrs.create(key, self.metainfo[key])
                # Add timestamp as attribute
                g.attrs.create('timestamp', self.timestamp)
        
        # Prepare data for processing
        files_to_process = []
        for i, atten, file in zip(np.arange(len(self.attens[self.index:])), self.attens[self.index:], self.files[self.index:]):
            files_to_process.append((i, atten, file))
        
        print(f"Starting HMM processing for {len(files_to_process)} files with timestamp {self.timestamp}...")
        
        if HMM_SUPPORTS_PARALLEL:
            print(f"Using {self.num_cores} CPU cores for HMM-level parallelism")
        else:
            print("Parallel processing not supported by your hmmlearn version. Running in single-core mode.")
        
        # Process files sequentially with optimized HMM-level parallelism if available
        HMM = []
        current_means = means
        current_covars = covars
        
        for i, atten, file in files_to_process:
            print(f"Processing file {i+1}/{len(files_to_process)}: attenuation {atten}")
            result = self._process_single_file(i, atten, file, current_means, current_covars, 
                                            intTime, SNRmin, skip, savefile, self.metainfo,
                                            hmm_n_jobs=self.num_cores)  # This parameter will be ignored if not supported
            
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
        
        # Save all HMM models to a single NPZ file with timestamp
        if len(HMM) > 0:
            try:
                hmm_models_data = {
                    'num_models': len(HMM),
                    'numModes': self.numModes,
                    'phi': self.phi,
                    'temp': self.temp,
                    'attens': self.attens[self.index:self.index+len(HMM)],
                    'powers': self.power_to_device[self.index:self.index+len(HMM)],
                    'timestamp': self.timestamp
                }
            except:
                hmm_models_data = {
                    'num_models': len(HMM),
                    'numModes': self.numModes,
                    'phi': self.phi,
                    'temp': self.temp,
                    'attens': self.attens[self.index:self.index+len(HMM)],
                    'power': self.power_to_device,
                    'timestamp': self.timestamp
                }
            
            # Add data for each model
            for i, model in enumerate(HMM):
                hmm_models_data[f'model{i}_means'] = model.means_
                hmm_models_data[f'model{i}_covars'] = model.covars_
                hmm_models_data[f'model{i}_transmat'] = model.transmat_
                hmm_models_data[f'model{i}_startprob'] = model.startprob_
            
            # Save to NPZ file with timestamp
            np.savez(
                os.path.join(analysis_dir, f'HMM_models_M{self.numModes}_T{self.temp}_PHI{str(self.phi).replace(".","p")[:5]}_{self.timestamp}.npz'),
                **hmm_models_data
            )
        
        print("Starting post-HMM analysis plots.....")
        # Pass timestamp to ensure statistics plots are also uniquely named
        create_HMM_QP_statistics_plots(self.hdf5_file, self.figure_path, self.numModes)
        print(f"Analysis completed with timestamp: {self.timestamp}")
        print("="*10+"\tHMM ANALYSIS CONCLUDED\t"+"="*10+"\n\n")

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
        
        # Draw the plot
        plt.draw()
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
        plt.draw()

