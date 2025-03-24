from AlazarPowerSweepData import *

if __name__ == "__main__":
    createPdf = True
    intTime=1
    SNRmin=3
    
    project_path = str(input(r"Path to Measurement Run Root: "))
    power_sweep_obj = AlazarPowerSweepData(project_path)
    
    power_sweep_obj.process_Alazar_Data(avgTime=2,plots=createPdf)
    
    # Using new parameters for more control
    power_sweep_obj.start_HMM_fit(
        intTime=intTime, 
        SNRmin=SNRmin,
        n_jobs=8,                      # Use 8 CPU cores
        covariance_type="full",        # Use full covariance matrices
        n_iter=250,                    # Increase max iterations
        tol=1e-3,                    # Tighter convergence tolerance
        verbose=True,                  # Show detailed progress
        transition_model="physics"     # Use physics-informed transition matrix
    )