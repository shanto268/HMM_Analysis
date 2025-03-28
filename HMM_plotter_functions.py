import glob
import os
import subprocess

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit, leastsq
from scipy.signal import oaconvolve, savgol_filter, windows

import HMM_helper_functions as hmm_func
import quasiparticleFunctions as qp


def weightedExp(t,a,tau):
    return a*(t/tau)*np.exp(-t/tau)

def fitWeightedExpDecay(dist,t,cut=0,returnTauDetector=True,returnSGDIST=False):
    '''estimates the detection rate and fits the distribution to exponential.
    
    returns pars, cov from scipy.optimize.curve_fit and optionally the detector timescale.
    --------------------------------------
    dist:   data, presumably lifetimes between events
    t:      times corresponding to dist, must have same size
    returnTauDetector:  Boolean, if False, only returns pars, cov.
    '''
    cutmask = t >= cut
    window = max(int(len(dist)*0.04),5)
    window += 0 if window%2 else 1 # ensure window is odd
    sgdist = savgol_filter(dist,window,3)
    tdetInd = np.argmax(sgdist)
    tauDetector = t[tdetInd]
    ampGuess = 1.2*sgdist[tdetInd]
    mask = sgdist[tdetInd:] < ampGuess/np.e
    tauGuess = t[tdetInd:][mask][0]
    # tauGuess = cut
    pars, cov = curve_fit(weightedExp,t[cutmask],dist[cutmask],p0=[ampGuess,tauGuess])
    
    if returnSGDIST and returnTauDetector:
        return pars, cov, tauDetector, sgdist
    elif returnTauDetector:
        return pars, cov, tauDetector
    elif returnSGDIST:
        return pars, cov, sgdist
    else:
        return pars, cov

def getWeightedTauDist(dist,bins=80,color='grey',alpha=0.3,figsize=[9,6]):
    '''Creates new figure with given distribution as a histogram and returns nonzero bins with centers.
    
    returns pyplot subplot, nonzero bin counts, nonzero bin centers
    ---------------------------
    dist:   data to histogram
    bins:   passed to pyplot.hist
    color:  passed to pyplot.hist
    alpha:  passed to pyplot.hist
    figsize:    passed to pyplot.figure
    '''
    fig = plt.figure(figsize=figsize,constrained_layout=True)
    h = fig.add_subplot()
    hi = h.hist(dist,weights=dist,bins=bins,color=color,alpha=alpha,density=True)
    # h.set_xlim(hi[1][1],hi[1][-1])
    # h.set_ylim(0,1.5*np.max(hi[0][3:]))
    binmask = np.array(hi[0] > 0,dtype=bool)
    BinCenters = (hi[1][1:] + hi[1][:-1])/2
    plt.close()
    return hi[0][binmask], BinCenters[binmask]


def PlotWeightedExpDecay(dist,bins=100):
    plt.hist(dist,bins,density=True, weights=dist, color="grey", alpha=0.3)
    plt.xlabel('Time [$\\mu$s]')
    plt.ylabel("density * $\\overline{\\mu}$")


def create_lifetime_distribution(hdf5_file, figpath):
    try:
        # Get the last key to use most recent data
        with h5py.File(hdf5_file, 'r') as fb:
            keys = list(fb.keys())
            if not keys:
                print("No data found in HDF5 file for lifetime distribution")
                return
                
            # Try to get data from the last key (most recent attenuation)
            key = keys[-1]
            try:
                nEst = fb[key]['Q'][:]
                sampleRate = fb[key].attrs.get('downsampleRateMHz')
                
                # Verify we have enough data
                if len(nEst) == 0:
                    print(f"Empty Q data found in {key}")
                    return
                    
                print(f"Using data from {key} for lifetime distribution with {len(nEst)} points")
            except Exception as e:
                print(f"Error reading data from {key}: {str(e)}")
                return
                
        # Proceed with lifetime calculation
        time = np.arange(len(nEst)) / sampleRate
        lifetimes_dict = qp.extractLifetimes(nEst, time)
        
        # Check if we got any lifetimes
        if not lifetimes_dict:
            print("No lifetimes extracted from data")
            return
            
        # Create the plots for each mode
        for key, value in lifetimes_dict.items():
            # Skip if no transitions were found for this mode
            if len(value) == 0:
                print(f"No transitions found for mode {key}")
                continue
                
            # Create the plot
            plt.figure(figsize=(8, 6))
            qp.fitAndPlotExpDecay(value)
            plt.title(f"QP Mode: {key}")
            plt.grid(True, alpha=0.3)
            
            # Ensure directory exists
            os.makedirs(figpath, exist_ok=True)
            
            # Save the figure
            save_path = os.path.join(figpath, f"lifetime_of_{key}_qp_distribution.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved lifetime distribution for mode {key} to {save_path}")
    except Exception as e:
        print(f"Error in create_lifetime_distribution: {str(e)}")
        import traceback
        traceback.print_exc()

def fitAndPlotWeightedExpDecay(dist,key,cut=None,bins=100,figsize=[3.325,3.325]):
    if cut is None:
        cut = np.mean(dist)
        
    hi,bc = getWeightedTauDist(dist,bins=bins,figsize=figsize)
    
    pars,cov,taud,sgdist = fitWeightedExpDecay(hi,bc,cut=cut,returnSGDIST=True)
    perr = np.sqrt(np.diag(cov)) # 1 sigma error on fit parameters
    taus = [pars[0],perr[0],taud]
    
    fit = weightedExp(bc,*pars)

    lowb = weightedExp(bc,pars[0]-perr[0],pars[1]-perr[1])
    uppb = weightedExp(bc,pars[0]+perr[0],pars[1]+perr[1])
    
    plt.hist(dist,bins, weights=dist, color="grey", alpha=0.3,density=True)
    labels = "fit: $ \\tau = {:6.1f} \pm{:6.1f} \\mu s $".format(pars[1],perr[1])
    plt.plot(bc,fit,color='darkgreen',label=labels)
    plt.fill_between(bc,uppb,lowb,color='lightgreen')
    plt.ylabel("density * $\\overline{\\mu}$")
    plt.legend()
    plt.xlabel('Time [$\\mu$s]')    
    fitstring = r"$\frac{AT}{\tau}e^{\frac{T}{\tau}}$"
    plt.title("QP Mode: {} | {}".format(key, fitstring))

def create_weighted_lifetime_distribution(hdf5_file, figpath, numModes):
    try:
        # Get the last key to use most recent data
        with h5py.File(hdf5_file, 'r') as fb:
            keys = list(fb.keys())
            if not keys:
                print("No data found in HDF5 file for weighted lifetime distribution")
                return
                
            # Try to get data from the last key (most recent attenuation)
            key = keys[-1]
            try:
                nEst = fb[key]['Q'][:]
                sampleRate = fb[key].attrs.get('downsampleRateMHz')
                
                # Verify we have enough data
                if len(nEst) == 0:
                    print(f"Empty Q data found in {key}")
                    return
                    
                print(f"Using data from {key} for weighted lifetime distribution with {len(nEst)} points")
            except Exception as e:
                print(f"Error reading data from {key}: {str(e)}")
                return
        
        # Proceed with lifetime calculation
        time = np.arange(len(nEst)) / sampleRate
        lifetimes_dict = qp.extractLifetimes(nEst, time)
        
        # Check if we got any lifetimes
        if not lifetimes_dict:
            print("No lifetimes extracted from data")
            return
            
        # Create the plots for each mode
        for key, value in lifetimes_dict.items():
            # Skip if no transitions were found for this mode
            if len(value) == 0:
                print(f"No transitions found for mode {key}")
                continue
                
            # Create the plot
            plt.figure(figsize=(8, 6))
            fitAndPlotWeightedExpDecay(value, key)
            plt.grid(True, alpha=0.3)
            
            # Ensure directory exists
            os.makedirs(figpath, exist_ok=True)
            
            # Save the figure
            save_path = os.path.join(figpath, f"weighted_lifetime_of_{key}_qp_distribution.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved weighted lifetime distribution for mode {key} to {save_path}")
    except Exception as e:
        print(f"Error in create_weighted_lifetime_distribution: {str(e)}")
        import traceback
        traceback.print_exc()

def create_HMM_QP_statistics_plots(hdf5_file, figpath, numModes):
    """
    Create a set of plots for analyzing HMM QP statistics.
    
    Parameters:
    -----------
    hdf5_file : str
        Path to the HDF5 file containing the HMM results
    figpath : str
        Path to save the figures
    numModes : int
        Number of modes in the HMM model (2 or 3)
    """
    hmm_func.set_plot_style()
    figpath = figpath + f"/post_HMM_fit_plots_M{numModes}"
    hmm_func.create_path(figpath)
    
    # Use a dictionary to store function names and references for cleaner code
    plot_functions = {
        "mean_occupation": create_mean_occupation_plot,
        "transition_probability": create_transition_probability_plot,
        "transition_rate": create_transition_rate_plot,
        "transition_lifetimes": create_transition_lifetimes_plot,
        "lifetime_distribution": create_lifetime_distribution,
        "weighted_lifetime_distribution": create_weighted_lifetime_distribution
    }
    
    # Check if file exists
    if not os.path.exists(hdf5_file):
        print(f"Error: HDF5 file '{hdf5_file}' does not exist")
        return
    
    # Check numModes is valid
    if numModes not in [2, 3]:
        print(f"Error: Unsupported number of modes: {numModes}. Only 2 or 3 modes are supported.")
        return
    
    print(f"Creating HMM QP statistics plots from {hdf5_file}")
    print(f"Saving plots to {figpath}")
    
    # Create each plot with error handling
    for name, func in plot_functions.items():
        print(f"Creating {name} plot...")
        try:
            # For functions that need numModes
            if name in ["transition_probability", "transition_rate", "transition_lifetimes", "weighted_lifetime_distribution"]:
                func(hdf5_file, figpath, numModes)
            else:
                func(hdf5_file, figpath)
            print(f"Successfully created {name} plot")
        except Exception as err:
            print(f"Error in {name} plot generation:")
            print(f"  - Error message: {str(err)}")
            import traceback
            traceback.print_exc()
            print(f"  - Continuing with next plot...")
    
    print(f"Completed HMM QP statistics plots generation")

def create_summary_plot_pdf(figpath):
    raise NotImplementedError()

def create_mean_occupation_plot(hdf5_file, figpath):
    """
    Create a plot showing mean occupation vs. LO power.
    
    Parameters:
    -----------
    hdf5_file : str
        Path to the HDF5 file containing the HMM results
    figpath : str
        Path to save the figures
    """
    LOps = []
    Qmeans = []
    
    try:
        with h5py.File(hdf5_file, 'r') as fb:
            for key in list(fb.keys()):
                try:
                    LOp = fb[key].attrs.get('LOpower')
                    Qmean = fb[key].attrs.get('mean')
                    
                    if LOp is None:
                        print(f"Warning: LOpower not found for {key}")
                        continue
                        
                    if Qmean is None:
                        print(f"Warning: Mean occupation not found for {key}")
                        continue
                        
                    LOps.append(LOp)
                    Qmeans.append(Qmean)
                except Exception as e:
                    print(f"Error processing {key}: {str(e)}")
                    continue
    except Exception as e:
        print(f"Error opening HDF5 file {hdf5_file}: {str(e)}")
        return
    
    # Check if we have any data
    if not LOps or not Qmeans:
        print("No mean occupation data found")
        return
        
    # Create the figure directory if it doesn't exist
    os.makedirs(os.path.dirname(figpath), exist_ok=True)
    
    figname = figpath + "/" + 'meanOccupation.png'
    create_2_scale_scatter_plots(LOps, Qmeans, 'LO power [dBm]', 
                               'QP Occupation Number', 
                               "Mean Occupation",
                               figname)


def create_2_scale_scatter_plots(x, y, xlabel, ylabel, title, figname):
    """
    Create scatter plots with both linear and log scales.
    
    Parameters:
    -----------
    x : array-like
        X values for the scatter plot
    y : array-like
        Y values for the scatter plot
    xlabel : str
        Label for the x-axis
    ylabel : str
        Label for the y-axis
    title : str
        Title for the plots
    figname : str
        Path to save the figure
    """
    plt.figure(figsize=(10, 8))
    plt.suptitle(title)
    
    # Linear scale plot
    plt.subplot(211)
    plt.scatter(x, y, color="red")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)

    # Log scale plot - handle non-positive values
    plt.subplot(212)
    try:
        # Filter out non-positive values for log scale
        valid_indices = []
        valid_y_values = []
        valid_x_values = []
        
        for i, val in enumerate(y):
            if val > 0 and i < len(x):  # Ensure we have a matching x value
                valid_indices.append(i)
                valid_y_values.append(val)
                valid_x_values.append(x[i])
        
        # Only plot if we have valid positive values
        if len(valid_y_values) > 0:
            plt.scatter(valid_x_values, valid_y_values, color="red")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            
            if len(valid_y_values) < len(y):
                plt.figtext(0.1, 0.01, f"Note: {len(y) - len(valid_y_values)} non-positive values not shown in log scale", 
                           fontsize=8, style='italic')
        else:
            plt.figtext(0.5, 0.5, "No positive values available for log scale plotting", 
                       ha='center', fontsize=10, color='red')
    except Exception as e:
        print(f"Warning: Error creating log scale plot: {str(e)}")
        # Create a message in the plot to indicate the issue
        plt.figtext(0.5, 0.5, f"Log scale unavailable: {str(e)}", 
                   ha='center', fontsize=10, color='red')
    
    # Save the figure
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(figname), exist_ok=True)
        plt.savefig(figname, bbox_inches='tight', dpi=150)
        print(f"Saved figure to {figname}")
    except Exception as e:
        print(f"Error saving figure: {str(e)}")
    
    plt.close()



def create_2_scale_plots(x, y, xlabel, ylabel, title, figname):
    """
    Create line plots with both linear and log scales.
    
    Parameters:
    -----------
    x : array-like
        X values for the line plot
    y : array-like
        Y values for the line plot
    xlabel : str
        Label for the x-axis
    ylabel : str
        Label for the y-axis
    title : str
        Title for the plots
    figname : str
        Path to save the figure
    """
    plt.figure(figsize=(10, 8))
    plt.suptitle(title)
    
    # Linear scale plot
    plt.subplot(211)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)

    # Log scale plot - handle non-positive values
    plt.subplot(212)
    try:
        # Filter out non-positive values for log scale
        valid_indices = []
        valid_y_values = []
        valid_x_values = []
        
        for i, val in enumerate(y):
            if val > 0 and i < len(x):  # Ensure we have a matching x value
                valid_indices.append(i)
                valid_y_values.append(val)
                valid_x_values.append(x[i])
        
        # Only plot if we have valid positive values
        if len(valid_y_values) > 0:
            plt.plot(valid_x_values, valid_y_values)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            
            if len(valid_y_values) < len(y):
                plt.figtext(0.1, 0.01, f"Note: {len(y) - len(valid_y_values)} non-positive values not shown in log scale", 
                           fontsize=8, style='italic')
        else:
            plt.figtext(0.5, 0.5, "No positive values available for log scale plotting", 
                       ha='center', fontsize=10, color='red')
    except Exception as e:
        print(f"Warning: Error creating log scale plot: {str(e)}")
        # Create a message in the plot to indicate the issue
        plt.figtext(0.5, 0.5, f"Log scale unavailable: {str(e)}", 
                   ha='center', fontsize=10, color='red')
    
    # Save the figure
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(figname), exist_ok=True)
        plt.savefig(figname, bbox_inches='tight', dpi=150)
        print(f"Saved figure to {figname}")
    except Exception as e:
        print(f"Error saving figure: {str(e)}")
    
    plt.close()


def create_transition_probability_plot(hdf5_file, figpath, numModes):
    """
    Create a plot showing transition probabilities between states.
    
    Parameters:
    -----------
    hdf5_file : str
        Path to the HDF5 file containing the HMM results
    figpath : str
        Path to save the figures
    numModes : int
        Number of modes in the HMM model (2 or 3)
    """
    if numModes == 2:
        P0s = []
        P1s = []
        ys = [P0s, P1s]
        labels = ["P0","P1"]
    elif numModes == 3:
        P0s = []
        P1s = []
        P2s = []
        ys = [P0s, P1s, P2s]
        labels = ["P0","P1", "P2"]
    else:
        raise ValueError(f"Unsupported numModes: {numModes}")
        
    LOps = []

    with h5py.File(hdf5_file, 'r') as fb:
        for key in list(fb.keys()):
            try:
                LOp = fb[key].attrs.get('LOpower')
                if LOp is None:
                    print(f"Warning: LOpower not found for {key}")
                    continue
                
                LOps.append(LOp)
                
                if numModes == 2:
                    try:
                        P0 = fb[key].attrs.get('P0')
                        P1 = fb[key].attrs.get('P1')
                        
                        if P0 is None or P1 is None:
                            print(f"Warning: Missing probability data for {key}")
                            continue
                            
                        P0s.append(P0)
                        P1s.append(P1)
                    except Exception as e:
                        print(f"Error reading probabilities for {key}: {str(e)}")
                        continue
                        
                elif numModes == 3:
                    try:
                        P0 = fb[key].attrs.get('P0')
                        P1 = fb[key].attrs.get('P1')
                        P2 = fb[key].attrs.get('P2')
                        
                        if P0 is None or P1 is None or P2 is None:
                            print(f"Warning: Missing probability data for {key}")
                            continue
                            
                        P0s.append(P0)
                        P1s.append(P1)
                        P2s.append(P2)
                    except Exception as e:
                        print(f"Error reading probabilities for {key}: {str(e)}")
                        continue
                else:
                    raise ValueError(f"Unsupported numModes: {numModes}")
            except Exception as e:
                print(f"Error processing {key}: {str(e)}")
                continue

    # Filter out None values and check if we have any data
    LOps = list(filter(lambda item: item is not None, LOps))
    
    if not LOps:
        print(f"No valid data found for transition probability plot")
        return
        
    # Verify we have matching data sizes
    min_length = min(len(LOps), min([len(y) for y in ys]))
    if min_length < len(LOps):
        print(f"Warning: Data sizes don't match. Truncating to {min_length} points")
        LOps = LOps[:min_length]
        for i in range(len(ys)):
            ys[i] = ys[i][:min_length]

    figname = figpath + "/" + "probabilities.png"
    create_2_scale_multiple_scatter_plots(LOps, ys, "LO Power [dBm]", 
                                         'Probability of mode', 
                                         labels, 
                                         "Transition Probabilities", 
                                         figname)


def create_2_scale_multiple_scatter_plots(x, ys, xlabel, ylabel, labels, title, figname):
    plt.figure(1, figsize=(10, 8))
    plt.suptitle(title)
    
    # Linear scale plot
    plt.subplot(211)
    for i, y in enumerate(ys):
        plt.scatter(x, y, label=labels[i])
    plt.xlabel(xlabel)
    plt.legend()
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)

    # Log scale plot - handle non-positive values
    plt.subplot(212)
    try:
        for i, y in enumerate(ys):
            # Filter out non-positive values for log scale
            valid_indices = []
            valid_y_values = []
            valid_x_values = []
            
            for j, val in enumerate(y):
                if val > 0 and j < len(x):  # Ensure we have a matching x value
                    valid_indices.append(j)
                    valid_y_values.append(val)
                    valid_x_values.append(x[j])
            
            # Only plot if we have valid positive values
            if len(valid_y_values) > 0:
                plt.scatter(valid_x_values, valid_y_values, label=labels[i])
            else:
                print(f"Warning: No positive values for {labels[i]} - skipping log plot for this series")
                
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.figtext(0.1, 0.01, "Note: Only positive values shown in log scale", fontsize=8, style='italic')
    except Exception as e:
        print(f"Warning: Error creating log scale plot: {str(e)}")
        # Create a message in the plot to indicate the issue
        plt.figtext(0.5, 0.5, f"Log scale unavailable: {str(e)}", 
                   ha='center', fontsize=10, color='red')
    
    # Use bbox_inches='tight' to ensure everything fits
    plt.savefig(figname, bbox_inches='tight', dpi=150)
    plt.close()



def create_transition_rate_plot(hdf5_file, figpath, numModes):
    """
    Create a plot showing transition rates between states.
    
    Parameters:
    -----------
    hdf5_file : str
        Path to the HDF5 file containing the HMM results
    figpath : str
        Path to save the figures
    numModes : int
        Number of modes in the HMM model (2 or 3)
    """
    if numModes == 2:
        rate01 = []
        rate10 = []
        ys = [rate01, rate10]
        labels = ['$\\Gamma_{01}$', '$\\Gamma_{10}$']
    elif numModes == 3:
        rate01 = []
        rate10 = []
        rate02 = []
        rate20 = []
        rate12 = []
        rate21 = []
        ys = [rate01, rate10, rate02, rate20, rate12, rate21]
        labels = ['$\\Gamma_{01}$', '$\\Gamma_{10}$', '$\\Gamma_{02}$', '$\\Gamma_{20}$', '$\\Gamma_{12}$', '$\\Gamma_{21}$']
    else:
        raise ValueError(f"Unsupported numModes: {numModes}")
        
    LOps = []

    with h5py.File(hdf5_file, 'r') as fb:
        for key in list(fb.keys()):
            try:
                LOp = fb[key].attrs.get('LOpower')
                if LOp is None:
                    print(f"Warning: LOpower not found for {key}")
                    continue
                    
                LOps.append(LOp)
                
                # Safely get the transition rates
                try:
                    rates = fb[key]['transitionRatesMHz'][:]
                except Exception as e:
                    print(f"Error reading transition rates for {key}: {str(e)}")
                    continue

                if numModes == 2:
                    rate01.append(rates[0, 1])
                    rate10.append(rates[1, 0])
                elif numModes == 3:
                    rate01.append(rates[0, 1])
                    rate10.append(rates[1, 0])
                    rate02.append(rates[0, 2])
                    rate20.append(rates[2, 0])
                    rate12.append(rates[1, 2])
                    rate21.append(rates[2, 1])
                else:
                    raise ValueError(f"Unsupported numModes: {numModes}")
            except Exception as e:
                print(f"Error processing {key}: {str(e)}")
                continue

    # Filter out None values
    LOps = list(filter(lambda item: item is not None, LOps))
    
    if not LOps:
        print(f"No valid data found for transition rate plot")
        return
        
    figname = figpath + "/" + "transitionRatesMHz.png"
    create_2_scale_multiple_scatter_plots(LOps, ys, "LO Power [dBm]", 
                                         'Transition Rate [MHz]', 
                                         labels, 
                                         "Transition Rates", 
                                         figname)



def create_transition_lifetimes_plot(hdf5_file, figpath, numModes):
    if numModes == 2:
        tau0 = []
        tau1 = []
        ys = [tau0, tau1]
        labels = ['$\\tau_{0}$', '$\\tau_{1}$']
    elif numModes == 3:
        tau0 = []
        tau1 = []
        tau2 = []
        ys = [tau0, tau1, tau2]
        labels = ['$\\tau_{0}$', '$\\tau_{1}$', '$\\tau_{2}$']
    else:
        raise ValueError(f"Unsupported numModes: {numModes}")
    
    LOps = []

    with h5py.File(hdf5_file, 'r') as fb:
        for key in list(fb.keys()):
            try:
                LOp = fb[key].attrs.get('LOpower')
                if LOp is None:
                    print(f"Warning: LOpower not found for {key}")
                    continue
                    
                LOps.append(LOp)
                
                # Safely get the transition rates
                try:
                    rates = fb[key]['transitionRatesMHz'][:]
                except Exception as e:
                    print(f"Error reading transition rates for {key}: {str(e)}")
                    continue

                # Calculate lifetimes as 1/rate for each state
                # But skip if the rate is 0 or negative to avoid division by zero or negative lifetimes
                if numModes == 2:
                    # The lifetimes are the inverse of the sum of transition rates out of each state
                    # For each state i, get the sum of all rates i,j where j≠i
                    rate_out_0 = np.sum(rates[0, 1:])  # Sum of rates from state 0 to all other states
                    rate_out_1 = np.sum(rates[1, :1])   # Sum of rates from state 1 to all other states
                    
                    # Only append if rates are positive to avoid division by zero or negative lifetimes
                    tau0.append(1/rate_out_0 if rate_out_0 > 0 else np.nan)
                    tau1.append(1/rate_out_1 if rate_out_1 > 0 else np.nan)
                    
                elif numModes == 3:
                    # Calculate sum of rates out of each state
                    rate_out_0 = rates[0, 1] + rates[0, 2]  # Sum of rates from state 0 to states 1 and 2
                    rate_out_1 = rates[1, 0] + rates[1, 2]  # Sum of rates from state 1 to states 0 and 2
                    rate_out_2 = rates[2, 0] + rates[2, 1]  # Sum of rates from state 2 to states 0 and 1
                    
                    # Only append if rates are positive
                    tau0.append(1/rate_out_0 if rate_out_0 > 0 else np.nan)
                    tau1.append(1/rate_out_1 if rate_out_1 > 0 else np.nan)
                    tau2.append(1/rate_out_2 if rate_out_2 > 0 else np.nan)
                else:
                    raise ValueError(f"Unsupported numModes: {numModes}")
            except Exception as e:
                print(f"Error processing {key}: {str(e)}")
                continue

    # Filter out None values
    LOps = list(filter(lambda item: item is not None, LOps))
    
    # Replace NaN values with zeros for plotting (they'll be filtered in log scale)
    for y_list in ys:
        for i in range(len(y_list)):
            if np.isnan(y_list[i]):
                y_list[i] = 0
    
    figname = figpath + "/" + "transition_lifetimes.png"
    create_2_scale_multiple_scatter_plots(LOps, ys, "LO Power [dBm]", 'Lifetimes [$\\mu$s]', 
                                        labels, "Lifetimes from HMM", figname)


