from AlazarPowerSweepData import *

if __name__ == "__main__":
    project_path = str(input("Path to Measurement Run Root: "))
    power_sweep_obj = AlazarPowerSweepData(project_path)
    power_sweep_obj.start_HMM_fit()
