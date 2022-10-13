# Usage for Power Sweep Data:

1. **ONLY** Update the following in `metainfo.json` 
```json
    "dateOfMeasurement" : 20221005,
    "eccosorb": "True",
    "KandL": "True",
    "rigidCables": "True",
    "JPA": "True",
    "TWPA": "True",
    "ClearingCoupler": "False",
    "Device_id": "NBR07",
    "f0": 4.2727,
```
2. Verify and update all the info in `attenuation.json`
3. Run `python HMM_PowerSweep.py` or `jupyter-notebook HMM_PowerSweep.ipynb`

---

# Usage for Flux-Power Swepp Data:
1. **ONLY** Update the following in `metainfo.json` 
```json
    "dateOfMeasurement" : 20221005,
    "eccosorb": "True",
    "KandL": "True",
    "rigidCables": "True",
    "JPA": "True",
    "TWPA": "True",
    "ClearingCoupler": "False",
    "Device_id": "NBR07",
    "f0": 4.2727,
```

---

## To Do:

- [ ] Handle multiple modes
    - [ ] Automate extraction of means
    - [ ] P0, P1, P2
- [ ] Automate the determination for attenuation below which the system is non-linear
- [ ] Better algorithm for `covariances`
