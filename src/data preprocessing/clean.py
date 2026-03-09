"""
data preprocessing/clean.py
============================
Converts the raw fixed-width TPM text file to a clean CSV.

Run from anywhere after `pip install -e .`:
    python "src/data preprocessing/clean.py"
"""
# converts raw data from .txt to .csv 
# runs from anywhere after 'pip install -e .' for better path finding 

import pandas as pd
from spatialmt.config import Paths, validate_raw_inputs


def convert_raw_tpm() -> None:
    validate_raw_inputs()

    tpm_data = pd.read_fwf(Paths.raw_tpm_txt)
    tpm_data.to_csv(Paths.raw_tpm, index=False)

    print(f"Saved: {Paths.raw_tpm}  ({len(tpm_data):,} rows)")


if __name__ == "__main__":
    convert_raw_tpm()