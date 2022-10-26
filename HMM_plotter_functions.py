import numpy as np
import matplotlib.pyplot as plt
import fitTools.quasiparticleFunctions as qp
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
import HMM_helper_functions as hmm_func
import subprocess
import glob
from scipy.signal import windows, oaconvolve, savgol_filter
from scipy.optimize import curve_fit,leastsq


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
    with h5py.File(hdf5_file,'r') as fb:
        for key in list(fb.keys()):
            try:
                nEst = fb[key]['Q'][:]
                sampleRate = fb[key].attrs.get('downsampleRateMHz')
            except:
                pass
    time = np.arange(len(nEst)) / sampleRate
    lifetimes_dict = qp.extractLifetimes(nEst,time)
    for key,value in lifetimes_dict.items():
        qp.fitAndPlotExpDecay(value)
        plt.title(f"QP Mode: {key}")
        plt.savefig(figpath+"/"+f"lifetime_of_{key}_qp_distribution.png")
        plt.close()

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
    with h5py.File(hdf5_file,'r') as fb:
        for key in list(fb.keys()):
            try:           
                nEst = fb[key]['Q'][:]
                sampleRate = fb[key].attrs.get('downsampleRateMHz')
            except:
                pass
    time = np.arange(len(nEst)) / sampleRate
    lifetimes_dict = qp.extractLifetimes(nEst,time)
    for key,value in lifetimes_dict.items():
        fitAndPlotWeightedExpDecay(value,key)
        plt.savefig(figpath+"/"+f"weighted_lifetime_of_{key}_qp_distribution.png")
        plt.close()
def create_HMM_QP_statistics_plots(hdf5_file, figpath, numModes):
    hmm_func.set_plot_style()
    figpath = figpath + f"/post_HMM_fit_plots_M{numModes}"
    hmm_func.create_path(figpath)
    try:
        create_mean_occupation_plot(hdf5_file, figpath)
    except Exception as err:
        print("Issue with \"create_mean_occupation_plot\" \nError message {}".format(err))
    try:
        create_transition_probability_plot(hdf5_file, figpath, numModes)
    except Exception as err:
        print("Issue with \"create_transition_probability_plot\" \nError message {}".format(err))
    try:
        create_transition_rate_plot(hdf5_file, figpath, numModes)
    except Exception as err:
        print("Issue with \"create_transition_rate_plot\" \nError message {}".format(err))
    try:
        create_transition_lifetimes_plot(hdf5_file, figpath, numModes)
    except Exception as err:
        print("Issue with \"create_transition_lifetimes_plot\" \nError message {}".format(err))
    try:
        create_lifetime_distribution(hdf5_file, figpath)
    except Exception as err:
        print("Issue with \"create_lifetime_distribution\" \nError message {}".format(err))
    try:
        create_weighted_lifetime_distribution(hdf5_file, figpath, numModes)
    except Exception as err:
        print("Issue with \"create_weighted_lifetime_distribution\" \nError message {}".format(err))


def create_summary_plot_pdf(figpath):
    raise NotImplementedError()

def create_mean_occupation_plot(hdf5_file, figpath):
    LOps = []
    Qmeans = []
    with h5py.File(hdf5_file,'r') as fb:
        for key in list(fb.keys()):
            LOp = fb[key].attrs.get('LOpower')
            Qmean = fb[key].attrs.get('mean')
            LOps.append(LOp)
            Qmeans.append(Qmean)
    figname = figpath + "/" 'meanOccupation.png'
    create_2_scale_scatter_plots(LOps, Qmeans, 'LO power [dBm]', 'QP Occupation Number', "Mean Occupation" ,figname)



def create_2_scale_scatter_plots(x,y,xlabel,ylabel,title,figname):
    plt.figure(1)
    plt.suptitle(title)
    plt.subplot(211)
    plt.scatter(x, y, color="red")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    try:
        plt.subplot(212)
        plt.scatter(x, y, color="red")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yscale('log')
        plt.savefig(figname, bbox_inches='tight')
    except:
        pass
    plt.close()



def create_2_scale_plots(x,y,xlabel,ylabel,title,figname):
    plt.figure(1)
    plt.suptitle(title)
    plt.subplot(211)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    try:
        plt.subplot(212)
        plt.plot(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yscale('log')
        plt.savefig(figname, bbox_inches='tight')
    except:
        pass
    plt.close()


def create_transition_probability_plot(hdf5_file, figpath, numModes):
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
        raise ValueError()
    LOps = []

    with h5py.File(hdf5_file,'r') as fb:
        for key in list(fb.keys()):
            LOp = fb[key].attrs.get('LOpower')

            LOps.append(LOp)
            if numModes == 2:
                P0 = fb[key].attrs.get('P0')
                P1 = fb[key].attrs.get('P1')
                P0s.append(P0)
                P1s.append(P1)
            elif numModes == 3:
                P0 = fb[key].attrs.get('P0')
                P1 = fb[key].attrs.get('P1')
                P2 = fb[key].attrs.get('P2')
                P0s.append(P0)
                P1s.append(P1)
                P2s.append(P2)
            else:
                raise ValueError()

    figname = figpath + "/" + "probabilities.png"
    create_2_scale_multiple_scatter_plots(LOps, ys, "LO Power [dBm]", 'Probability of mode', labels, "Transition Probabilities", figname)


def create_2_scale_multiple_scatter_plots(x,ys,xlabel,ylabel,labels,title,figname):
    plt.figure(1)
    plt.suptitle(title)
    plt.subplot(211)
    for i,y in enumerate(ys):
        plt.scatter(x, ys[i],label=labels[i])
    plt.xlabel(xlabel)
    plt.legend()
    plt.ylabel(ylabel)

    plt.subplot(212)
    try:
        for i,y in enumerate(ys):
            plt.scatter(x, ys[i],label=labels[i])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yscale('log')
        plt.legend()
    except:
        pass
    plt.savefig(figname, bbox_inches='tight')
    plt.close()



def create_transition_rate_plot(hdf5_file, figpath, numModes):
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
        raise ValueError()
    LOps = []

    with h5py.File(hdf5_file,'r') as fb:
        for key in list(fb.keys()):
            try:
                LOp = fb[key].attrs.get('LOpower')

                LOps.append(LOp)
                rates = fb[key]['transitionRatesMHz'][:]

                if numModes == 2:
                    rate01.append(rates[0,1])
                    rate10.append(rates[1,0])
                elif numModes == 3:
                    rate01.append(rates[0,1])
                    rate10.append(rates[1,0])

                    rate02.append(rates[0,2])
                    rate20.append(rates[2,0])

                    rate12.append(rates[1,2])
                    rate21.append(rates[2,1])

                else:
                    raise ValueError()
            except:
                pass

    LOps = list(filter(lambda item: item is not None, LOps))
    figname = figpath + "/" + "transitionRatesMHz.png"
    create_2_scale_multiple_scatter_plots(LOps, ys, "LO Power [dBm]", 'Transition Rate [MHz]', labels, "Transition Rates", figname)



def create_transition_lifetimes_plot(hdf5_file, figpath, numModes):
    if numModes == 2:
        tau0 = []
        tau1 = []
        ys = [tau0, tau1]
        labels = ['$\\tau_{0}$', '$\\tau_{1}$']
    elif numModes == 3:
        tau0 = []
        tau1 = []
        tau3 = []
        ys = [tau0, tau1, tau3]
        labels = ['$\\tau_{0}$', '$\\tau_{1}$', '$\\tau_{2}$']
    else:
        raise ValueError()
    LOps = []

    with h5py.File(hdf5_file,'r') as fb:
        for key in list(fb.keys()):
            try:
                LOp = fb[key].attrs.get('LOpower')

                LOps.append(LOp)
                rates = fb[key]['transitionRatesMHz'][:]

                if numModes == 2:
                    tau0.append(rates[0,0])
                    tau1.append(rates[1,1])
                elif numModes == 3:
                    tau0.append(rates[0,0])
                    tau1.append(rates[1,1])
                    tau3.append(rates[2,2])

                else:
                    raise ValueError()
            except:
                pass

    LOps = list(filter(lambda item: item is not None, LOps))
    figname = figpath + "/" + "transition_lifetimes.png"
    create_2_scale_multiple_scatter_plots(LOps, ys, "LO Power [dBm]", 'Lifetimes [$\\mu$]', labels, "Lifetimes from HMM", figname)


