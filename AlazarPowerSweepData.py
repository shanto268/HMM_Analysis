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

from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt
import fitTools.quasiparticleFunctions as qp
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
import h5py
import glob
import json
import os
import matplotlib
from HMM_helper_functions import *


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
        return means


    def get_initial_QP_means(self, avgTime=3):
        if self.interactive:
            return self.get_QP_means_from_IQ(avgTime)
        else:
            try:
                return get_QP_means(self.project_root, self.phi, self.numModes)
            except:
                return self.get_QP_means_from_IQ(avgTime)




    def get_initial_QP_covars(self, r_unit=[[5,1],[1,5]]):
        dim_list = []
        for i in range(self.numModes):
            dim_list.append(r_unit)
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

    def start_HMM_fit(self,  r_unit=[[5,1],[1,5]], intTime=1,SNRmin=3, targetPower=None, numModes=2):
        print("\n\n"+"="*10+"\tHMM ANALYSIS STARTED\t"+"="*10)

        if self.interactive:
            chosenAtten = int(input("\nAttenuation below which the system goes non-linear: "))
            self.power_to_device = self.set_attenuation_configuration()

            self.metainfo = self.set_metadata()

            self.index = int(np.where(self.attens == chosenAtten)[0])
            print("\nThe power to the device is {} dBM at the chosen attenuation {}".format(self.power_to_device[self.index], self.attens[self.index]))

            self.numModes = int(input("\nNumber of Modes you want to fit: "))
            set_qt_backend()

            means = self.get_initial_QP_means()
            covars = self.get_initial_QP_covars(r_unit)
            print(f"Extracted Means:\n{means}\n\nGussed Covariance:\n{covars}\n")
            print("\nStarting HMM Analysis.....\n\n")
            self.runHMM(means, covars,intTime, SNRmin)
        else:
            self.power_to_device = self.set_attenuation_configuration()
            self.metainfo = self.set_metadata()
            self.index = int(np.where(self.power_to_device == targetPower)[0])

            chosenAtten = self.attens[self.index]
            print("\nThe chosen power to the device is {} dBM at the attenuation {}".format(self.power_to_device[self.index], chosenAtten))

            self.numModes = numModes
            print(f"\nNumber of Modes to be fit: {self.numModes}")
            set_qt_backend()

            means = self.get_initial_QP_means()
            covars = self.get_initial_QP_covars(r_unit)
            print(f"Extracted Means:\n{means}\n\nGussed Covariance:\n{covars}\n")
            print("\nStarting HMM Analysis.....\n\n")
            self.runHMM(means, covars,intTime, SNRmin)


    def restart_HMM_fit(self, r_unit=[[5,1],[1,5]], intTime=1,SNRmin=3):
        raise NotImplementedError()

    def runHMM(self, means, covars,intTime=1,SNRmin=3):
        matplotlib.use('Agg')

        HMM = []
        skip = np.copy(self.index)
        n_comp = self.numModes
        #######################################3
        # original code used i,file in enumerate(files). There are still some dependencies on i in the code below
        #########################################
        hmm_fits_pdf = PdfPages('{}/HMM_IQ_fits.pdf'.format(self.project_path))
        hmm_time_series_pdf = PdfPages('{}/HMM_time_series.pdf'.format(self.project_path))

        for i,atten,file in zip(np.arange(len(self.attens[self.index:])),self.attens[self.index:],self.files[self.index:]):

            metainfo = self.metainfo

            savefile = os.path.join(self.project_path, 'AnalyisResults','FullDataset_M{}_T{}_PHI{}_.hdf5'.format(self.numModes, self.temp,str(self.phi).replace(".","p")[:5]))
            self.hdf5_file = savefile
            if not os.path.exists(os.path.split(savefile)[0]):
                os.makedirs(os.path.split(savefile)[0])
            figpath = os.path.join(self.figure_path, f'ATTEN{atten}')
            if not os.path.exists(figpath):
                os.makedirs(figpath)
            with h5py.File(savefile,'a') as f:
                g = f.require_group(f'ATTEN{atten}')
                for key in metainfo:
                    g.attrs.create(key,metainfo[key])
            ###########################################################
            # In boxcar downsampling below, avgTime is hard coded to always start at 1.
            # At each iteration of this loop, we expect the results to change only by a small amount;
            # this is why we use the previous values of means and covars as initial guesses.
            # Similarly, we should use the previous avgTime so as to avoid unecessary fits (we're decreasing
            # our readout power at each iteration. avgTime should only increase from one iteration to the next,
            # never get smaller, so if it's increased due to failing SNR check, there's no reason to return to 1)
            #############################################################
            # oops, accidentaly hit shift+enter. Hopefully stopped it before overwriting anything!
            data = qp.loadAlazarData(file)
        #     data, sr = qp.BoxcarDownsample(data, avgTime=1,sampleRate=self.sampleRateFromData, returnRate=True)
            data, sr = qp.BoxcarDownsample(data, avgTime=intTime,sampleRate=self.sampleRateFromData, returnRate=True) 
            data = qp.uint16_to_mV(data)
            # fit the HMM
            M = hmm.GaussianHMM(n_components=n_comp,covariance_type='full',init_params='s',n_iter=500,tol=0.001)
            M.means_ = means
            M.covars_ = covars

            if n_comp == 3:
                M.transmat_ = np.array([[0.99, 0.009,0.001],[0.03, 0.95,0.02],[0.05,0.05,0.9]])
            else:
                M.transmat_ = np.array([[0.99, 0.01],[0.01, 0.99]])

            M.fit(data.T)
            HMM.append(M)
            #########################################
            # the block below uses loop index i to get the previous HMM results. i is now static, so 
            # our failure condition comparing intTime to lifetimes is broken
            #################3#######################3
            with h5py.File(savefile,'r') as ff:
        #         f = ff[f'ATTEN{attens[i+skip-1]}'] 
                # altered here because we want the previous iteration
                oldrates = ff[f'ATTEN{self.attens[i+skip-1]}/transitionRatesMHz'][:] if i != 0 else 0.0001*np.ones((n_comp,n_comp))
                lifetimes = np.array([1/oldrates[j,j] for j in range(n_comp)])
                t01 = 1/oldrates[0,1]
                t10 = 1/oldrates[1,0]
                ttimes = np.array([t01,t10])

            # get SNR
            snr01 = qp.getSNRhmm(M,mode1=0,mode2=1)
            SNRs = np.array([snr01,])

            logprob,Q = M.decode(data.T)
            Qmean = np.mean(Q)
            P0 = np.sum(Q == 0)/Q.size
            P1 = np.sum(Q == 1)/Q.size
            P2 = np.sum(Q == 2)/Q.size

            if np.min(SNRs) < SNRmin and intTime <= np.min(lifetimes)/2 and intTime <= np.min(ttimes) and P1 < P0:
                success = False
                srold = np.copy(sr)
                while not success:
                    intTime *= 1.15*SNRmin/np.min(SNRs)
                    print(f'\n\n\nNew integration time = {intTime} because SNR was {np.min(SNRs):.6}\n\n\n')
                    data = qp.loadAlazarData(file)
                    srold = np.copy(sr)
                    data, sr = qp.BoxcarDownsample(data,intTime,self.sampleRateFromData,returnRate = True)
                    data = qp.uint16_to_mV(data)
                    M = hmm.GaussianHMM(n_components=n_comp,covariance_type='full',init_params='s',n_iter=500,tol=0.001)
                    ###################################################
                    # The block below also has dependency on loop index i. 
                    # Now when the SNR is small and we try fitting with increased intTime, it will pull
                    # less than ideal initial values for means and covars.
                    ###################################################
                    with h5py.File(savefile,'r') as ff:
        #                 f = ff[f'ATTEN{attens[i+skip-1]}'] 
                        # altered here because we want the previous iteration
                        M.means_ = ff[f'ATTEN{self.attens[i+skip-1]}'].attrs.get('HMMmeans_') if i != 0 else means
                        M.covars_ = ff[f'ATTEN{self.attens[i+skip-1]}'].attrs.get('HMMcovars_') if i != 0 else covars
                    if n_comp == 3:
                        M.transmat_ = np.array([[0.99, 0.009,0.001],[0.03, 0.95,0.02],[0.05,0.05,0.9]])
                    else:
                        M.transmat_ = np.array([[0.99, 0.01],[0.01, 0.99]])
                    M.fit(data.T)
                    # get SNR
                    snr01 = qp.getSNRhmm(M,mode1=0,mode2=1)
                    SNRs = np.array([snr01,])
                    if np.min(SNRs) > SNRmin:
                        success = True
                    elif intTime > np.min(lifetimes)/2 or intTime > np.min(ttimes):
                        print('Integration time has grown too large.')
                        break
                    if sr == srold:
                        print('stuck in loop, exiting')
                        break
                    srold = np.copy(sr)

            ###################3
            # i > 3 condition can never be met with static i = 0
            #####################
            if intTime > np.min(lifetimes)/2 and i > 3:
                print('Integration time has grown too large.')
                break
            if P1 > P0:
                print('P1 became larger than P0!')
                break

            # plot the fit
            ##################
            # i is also used in several places for plotting here
            ####################
            h = qp.plotComplexHist(data[0],data[1],figsize=[4,4])
            if n_comp == 3:
                qp.make_ellipsesHMM(M,h,['purple','orange','green'])
            else:
                qp.make_ellipsesHMM(M,h,['purple','orange'])
            plt.xlabel('I [mV]')
            plt.ylabel('Q [mV]')
            plt.title('HMM fit | {:.2} MHz | {} dBm'.format(sr,self.power_to_device[i+skip]))
            plt.savefig(os.path.join(figpath,'HMMfits_{}_{}dBm.png'.format(i+skip,self.power_to_device[i+skip])))
            hmm_fits_pdf.savefig(plt.gcf())
            plt.close()

            # extract occupation
            logprob,Q = M.decode(data.T)
            Qmean = np.mean(Q)
            P0 = np.sum(Q == 0)/Q.size
            P1 = np.sum(Q == 1)/Q.size
            P2 = np.sum(Q == 2)/Q.size

            # plot a section of time series
            fig, ax = qp.plotTimeSeries(data,Q,np.arange(Q.size)/sr,1500,2000,zeroTime=True)
            plt.title('{:.2} MHz | {} dBm'.format(sr,self.power_to_device[i+skip]))
            plt.savefig(os.path.join(figpath,'TimeSeries__{}_{}dBm.png'.format(i+skip,self.power_to_device[i+skip])))
            hmm_time_series_pdf.savefig(plt.gcf())
            plt.close()

            # get the transition rates
            rates = qp.getTransRatesFromProb(sr,M.transmat_)

            # save all the information to aggregate file
            ##################
            # i is used in a few places here, in particular the last attributes LOpower and attens are incorrect
            # also note that skip has no meaning anymore. the variable index found from power = -127 has taken on the same role
            ##################
            with h5py.File(savefile,'a') as ff:
                fp = ff.require_group(f'ATTEN{atten}')
                fp.create_dataset('Q',data = Q)
                fp.create_dataset('data',data = data)
                fp.create_dataset('transitionRatesMHz',data = rates)
                fp.attrs.create('logprobQ',logprob)
                fp.attrs.create('mean',Qmean)
                fp.attrs.create('P0',P0)
                fp.attrs.create('P1',P1)
                try:
                    fp.attrs.create('P2',P2)
                except:
                    pass
                fp.attrs.create('SNRs',SNRs)
                fp.attrs.create('downsampleRateMHz',sr)
                fp.attrs.create('HMMmeans_',M.means_)
                fp.attrs.create('HMMstartprob_',M.startprob_)
                fp.attrs.create('HMMcovars_',M.covars_)
                fp.attrs.create('HMMtransmat_',M.transmat_)
                fp.attrs.create('LOpower',self.power_to_device[i+skip])
                fp.attrs.create('DAsetting',self.attens[i+skip])

                for key in metainfo:
                    fp.attrs.create(key,metainfo[key])

            # save a copy of current means to use as estimate for next power
            means= np.copy(M.means_)
            covars= np.copy(M.covars_)
            print(f"HMM fitting concluded on Alazar Data at attenuation: {atten} ...")

        
        print("All HMM fits created.....")
        hmm_fits_pdf.close()
        hmm_time_series_pdf.close()
        self.HMM = HMM

        try:
            fdir = os.path.join(self.project_path,'AnalyisResults')
            pickle_HMM(HMM,fdir)
            print("Pickled the HMM Object to disk.....")
        except:
            pass
        
        print("Starting post-HMM analysis plots.....")
        #create_HMM_QP_statistics_plots(self.hdf5_file)
        print("="*10+"\tHMM ANALYSIS CONCLUDED\t"+"="*10)
