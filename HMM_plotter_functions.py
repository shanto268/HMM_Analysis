import numpy as np
import matplotlib.pyplot as plt
import fitTools.quasiparticleFunctions as qp
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
import HMM_helper_functions as hmm_func
import subprocess
import glob

"""
TO DO:
- pdf making
"""

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
        PlotWeightedExpDecay(value)
        plt.title(f"QP Mode: {key}")
        plt.savefig(figpath+"/"+f"weighted_lifetime_of_{key}_qp_distribution.png")
        plt.close()

def create_HMM_QP_statistics_plots(hdf5_file, figpath, numModes):
    hmm_func.set_plot_style()
    figpath = figpath + f"/post_HMM_fit_plots_M{numModes}"
    hmm_func.create_path(figpath)
    create_mean_occupation_plot(hdf5_file, figpath)
    create_transition_probability_plot(hdf5_file, figpath, numModes)
    create_transition_rate_plot(hdf5_file, figpath, numModes)
    create_transition_lifetimes_plot(hdf5_file, figpath, numModes)
    create_lifetime_distribution(hdf5_file, figpath)
    create_weighted_lifetime_distribution(hdf5_file, figpath, numModes)
    # create_summary_plot_pdf(figpath)


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

    plt.subplot(212)
    plt.scatter(x, y, color="red")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale('log')
    plt.savefig(figname, bbox_inches='tight')
    plt.close()



def create_2_scale_plots(x,y,xlabel,ylabel,title,figname):
    plt.figure(1)
    plt.suptitle(title)
    plt.subplot(211)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.subplot(212)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale('log')
    plt.savefig(figname, bbox_inches='tight')
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
    for i,y in enumerate(ys):
        plt.scatter(x, ys[i],label=labels[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale('log')
    plt.legend()
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


