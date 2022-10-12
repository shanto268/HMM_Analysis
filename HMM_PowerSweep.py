from AlazarPowerSweepData import *

if __name__ == "__main__":
    createPdf = True
    intTime=1
    SNRmin=3
    
    project_path = str(input(r"Path to Measurement Run Root: "))
    power_sweep_obj = AlazarPowerSweepData(project_path)
    
    power_sweep_obj.process_Alazar_Data(avgTime=2,plots=createPdf)
    
    power_sweep_obj.start_HMM_fit(intTime=intTime, SNRmin= SNRmin)